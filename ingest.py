"""
ingest.py
---------
Loads the COLD Cases dataset from HuggingFace, chunks opinion text,
generates embeddings using sentence-transformers, and stores everything
in a local Qdrant vector database.

Run once before starting the API:
    python ingest.py              # full dataset (run overnight)
    python ingest.py --limit 500  # quick test run
"""

import os
import re
import argparse
from tqdm import tqdm

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "legal_cases"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
VECTOR_DIM      = 768
CHUNK_SIZE      = 400
CHUNK_OVERLAP   = 50
BATCH_SIZE      = 32


# ── HTML stripper ─────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """
    Remove HTML tags and clean up whitespace.
    The COLD Cases opinions field contains raw HTML from CourtListener —
    stripping it gives clean plain text for embedding.
    """
    if not text:
        return ""
    # Remove script and style blocks entirely
    text = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") \
               .replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_opinion_text(opinions_field) -> str:
    """
    The 'opinions' column is a list of dicts with keys:
      'type', 'text', 'author_str', 'per_curiam', etc.
    Concatenates all opinion texts, majority opinions first.
    HTML is stripped from each piece of text.
    """
    if not opinions_field:
        return ""

    majority_texts = []
    other_texts    = []

    for op in opinions_field:
        if not isinstance(op, dict):
            continue
        raw = op.get("text", "") or op.get("html", "") or ""
        text = strip_html(raw).strip()
        if not text:
            continue
        op_type = str(op.get("type", "")).lower()
        if "majority" in op_type or op_type == "010combined":
            majority_texts.append(text)
        else:
            other_texts.append(text)

    return "\n\n".join(majority_texts + other_texts)


def extract_author(opinions_field, judges_field) -> str:
    if opinions_field:
        for op in opinions_field:
            if isinstance(op, dict):
                author = op.get("author_str", "") or ""
                if author.strip():
                    return author.strip()
    return str(judges_field) if judges_field else "Unknown"


def extract_category(opinions_field) -> str:
    if not opinions_field:
        return "Unknown"
    types = {str(op.get("type", "")) for op in opinions_field if isinstance(op, dict) and op.get("type")}
    return ", ".join(types) if types else "Unknown"


def chunk_text(text: str) -> list:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start:start + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def make_payload(row: dict, chunk: str, chunk_index: int, author: str, category: str) -> dict:
    slug = row.get("slug", "")
    url  = f"https://www.courtlistener.com/opinion/{slug}/" if slug else ""
    return {
        "chunk_index"  : chunk_index,
        "chunk_text"   : chunk,
        "case_name"    : row.get("case_name", "") or row.get("case_name_short", "Unknown"),
        "author_name"  : author,
        "category"     : category,
        "date_filed"   : str(row.get("date_filed", "")),
        "absolute_url" : url,
        "court"        : row.get("court_short_name", ""),
        "jurisdiction" : row.get("court_jurisdiction", ""),
        "syllabus"     : strip_html(row.get("syllabus", "") or "")[:500],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(limit=None):
    print("Loading COLD Cases dataset from HuggingFace...")
    print("(Cached from first run — loading from disk)")
    dataset = load_dataset("harvard-lil/cold-cases", split="train")

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    print(f"Loaded {len(dataset):,} cases")

    print(f"\nConnecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, timeout=60)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Deleting existing '{COLLECTION_NAME}' collection...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection: '{COLLECTION_NAME}'")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("\nIngesting cases (HTML stripping + chunking + embedding)...")
    point_id    = 0
    batch_texts = []
    batch_meta  = []
    skipped     = 0

    def flush_batch():
        nonlocal point_id
        if not batch_texts:
            return
        vectors = embedder.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)
        points  = [
            PointStruct(id=point_id + i, vector=vectors[i].tolist(), payload=batch_meta[i])
            for i in range(len(batch_texts))
        ]
        for attempt in range(3):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"\nUpsert failed (attempt {attempt+1}/3): {e}. Retrying...")
                import time; time.sleep(5)
        point_id += len(batch_texts)
        batch_texts.clear()
        batch_meta.clear()

    for row in tqdm(dataset, desc="Ingesting"):
        opinions_field = row.get("opinions") or []
        opinion_text   = extract_opinion_text(opinions_field)

        # Fallback: syllabus/summary (also HTML-stripped)
        if len(opinion_text.strip()) < 100:
            opinion_text = strip_html(
                row.get("syllabus", "") or
                row.get("summary", "") or
                row.get("headmatter", "") or ""
            )

        if len(opinion_text.strip()) < 100:
            skipped += 1
            continue

        author   = extract_author(opinions_field, row.get("judges"))
        category = extract_category(opinions_field)
        chunks   = chunk_text(opinion_text)

        for idx, chunk in enumerate(chunks):
            batch_texts.append(chunk)
            batch_meta.append(make_payload(row, chunk, idx, author, category))
            if len(batch_texts) >= BATCH_SIZE:
                flush_batch()

    flush_batch()

    info = client.get_collection(COLLECTION_NAME)
    print(f"\nIngestion complete.")
    print(f"Cases skipped (no usable text) : {skipped:,}")
    print(f"Total vectors stored           : {info.points_count:,}")
    print(f"\nNext: uvicorn api:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(limit=args.limit)
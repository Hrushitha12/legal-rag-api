"""
retriever.py
------------
Legal case retrieval using LangChain + Qdrant + sentence-transformers.

LangChain provides:
  - HuggingFaceEmbeddings  : wraps sentence-transformers for query embedding
  - QdrantVectorStore      : registered as the vector store (used for writes/info)

The search itself uses the raw Qdrant client so we get our full flat payload
back (case_name, author_name, court, etc.) — LangChain's default reader
expects a nested "metadata" key which our ingest.py didn't produce.

Usage (standalone test):
    python retriever.py
"""

import os
from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "legal_cases"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K_CHUNKS    = 20
TOP_K_CASES     = 5
COHERE_API_KEY  = os.getenv("COHERE_API_KEY", "")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedCase:
    case_name    : str
    author_name  : str
    category     : str
    date_filed   : str
    absolute_url : str
    court        : str
    jurisdiction : str
    syllabus     : str
    best_chunk   : str
    score        : float


# ── Core retriever ────────────────────────────────────────────────────────────

class LegalRetriever:

    def __init__(self):
        print("Loading embedding model via LangChain + HuggingFace...")

        # LangChain HuggingFaceEmbeddings — wraps sentence-transformers
        self.embeddings = HuggingFaceEmbeddings(
            model_name    = EMBEDDING_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )

        # LangChain QdrantVectorStore — registered against our collection
        # Used here as the canonical LangChain vector store interface
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding           = self.embeddings,
            collection_name     = COLLECTION_NAME,
            url                 = QDRANT_URL,
            content_payload_key = "chunk_text",
        )

        # Raw Qdrant client for search — gives us the full flat payload
        # Our ingest.py stored fields flat (case_name, author_name, etc.)
        # rather than nested under "metadata", so we query directly
        self.qdrant_client = QdrantClient(url=QDRANT_URL, timeout=30)

        # Optional Cohere reranker
        if COHERE_AVAILABLE and COHERE_API_KEY:
            self.cohere_client = cohere.Client(COHERE_API_KEY)
            print("Cohere reranker enabled.")
        else:
            self.cohere_client = None
            print("Cohere reranker not configured — using score-based deduplication.")

    def retrieve(self, query: str, top_k_cases: int = TOP_K_CASES) -> list:
        """
        Main retrieval method.
        1. LangChain embeds the query via HuggingFaceEmbeddings
        2. Raw Qdrant client searches and returns full flat payload
        3. Deduplicate by case name, optional Cohere rerank
        Returns a list of RetrievedCase objects, best match first.
        """

        # Step 1 — Embed query using LangChain's embedding interface
        query_vector = self.embeddings.embed_query(query)

        # Step 2 — Search Qdrant with raw client to get full flat payload
        response = self.qdrant_client.query_points(
            collection_name = COLLECTION_NAME,
            query           = query_vector,
            limit           = TOP_K_CHUNKS,
            with_payload    = True,
        )
        hits = response.points

        if not hits:
            return []

        # Step 3 — Deduplicate: keep best-scoring chunk per unique case
        best_by_case: dict = {}
        for hit in hits:
            name = hit.payload.get("case_name", "Unknown")
            if name not in best_by_case or hit.score > best_by_case[name].score:
                best_by_case[name] = hit

        # Step 4 — Sort by score, optionally rerank with Cohere
        top_hits = sorted(best_by_case.values(), key=lambda h: h.score, reverse=True)

        if self.cohere_client and len(top_hits) > 1:
            top_hits = self._cohere_rerank(query, top_hits)

        # Step 5 — Build output objects from full payload
        cases = []
        for hit in top_hits[:top_k_cases]:
            p = hit.payload
            cases.append(RetrievedCase(
                case_name    = p.get("case_name", "Unknown"),
                author_name  = p.get("author_name", "Unknown"),
                category     = p.get("category", "Unknown"),
                date_filed   = p.get("date_filed", ""),
                absolute_url = p.get("absolute_url", ""),
                court        = p.get("court", ""),
                jurisdiction = p.get("jurisdiction", ""),
                syllabus     = p.get("syllabus", ""),
                best_chunk   = p.get("chunk_text", ""),
                score        = round(hit.score, 4),
            ))

        return cases

    def _cohere_rerank(self, query: str, hits: list) -> list:
        """Use Cohere Rerank API to re-order results by relevance."""
        docs     = [h.payload.get("chunk_text", "") for h in hits]
        response = self.cohere_client.rerank(
            model     = "rerank-english-v3.0",
            query     = query,
            documents = docs,
            top_n     = len(docs),
        )
        return [hits[r.index] for r in response.results]


# ── Singleton ─────────────────────────────────────────────────────────────────

_retriever_instance = None

def get_retriever() -> LegalRetriever:
    """Cached retriever — model loads once, reused for every API request."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = LegalRetriever()
    return _retriever_instance


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = get_retriever()

    test_queries = [
        "First Amendment freedom of speech government restriction",
        "Fourth Amendment unlawful search and seizure",
        "equal protection racial discrimination employment",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        results = retriever.retrieve(query, top_k_cases=3)
        if not results:
            print("No results found.")
            continue
        for i, case in enumerate(results, 1):
            print(f"\n[{i}] {case.case_name}")
            print(f"    Author : {case.author_name}")
            print(f"    Date   : {case.date_filed}")
            print(f"    Court  : {case.court}")
            print(f"    Score  : {case.score}")
            print(f"    URL    : {case.absolute_url}")
            print(f"    Preview: {case.best_chunk[:150]}...")
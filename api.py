"""
api.py
------
FastAPI application exposing the Legal RAG pipeline as a REST API.

Endpoints:
  GET  /health            — confirm service is running
  POST /query             — main RAG endpoint (retrieve + generate memo)
  POST /retrieve          — retrieval only, no LLM generation
  GET  /collection/info   — Qdrant collection stats

Run locally:
    uvicorn api:app --reload --port 8000

Then test at:
    http://localhost:8000/docs   (auto-generated Swagger UI)
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from retriever import get_retriever, LegalRetriever
from generator import generate_memo

load_dotenv()

# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query    : str  = Field(..., min_length=10, max_length=2000,
                            example="Fourth Amendment search and seizure warrant requirement")
    top_k    : int  = Field(default=5, ge=1, le=10,
                            description="Number of cases to retrieve")
    generate : bool = Field(default=True,
                            description="If False, returns retrieved cases only (no LLM call)")


class CaseResult(BaseModel):
    case_name    : str
    author_name  : str
    category     : str
    date_filed   : str
    absolute_url : str
    court        : str
    jurisdiction : str
    score        : float
    chunk_preview: str


class QueryResponse(BaseModel):
    query          : str
    cases          : list[CaseResult]
    memo           : str | None   = None
    model_used     : str | None   = None
    retrieval_ms   : int
    generation_ms  : int | None   = None
    total_ms       : int


class HealthResponse(BaseModel):
    status         : str
    qdrant         : str
    ollama_model   : str
    collection     : str
    vectors_count  : int | None


class CollectionInfoResponse(BaseModel):
    collection     : str
    vectors_count  : int
    vector_dim     : int
    distance       : str


# ── App lifecycle ─────────────────────────────────────────────────────────────

retriever: LegalRetriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model once at startup — shared across all requests."""
    global retriever
    print("Starting Legal RAG API...")
    print("Loading retriever (embedding model + Qdrant connection)...")
    retriever = get_retriever()
    print("Retriever ready.")
    yield
    print("Shutting down.")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Legal Precedent RAG API",
    description = (
        "Retrieval-Augmented Generation API for US legal case research. "
        "Given a legal query, retrieves relevant precedents from a Qdrant vector database "
        "and generates a structured legal memo using a local LLM (Ollama). "
        "Built on top of published research: arXiv:2406.01609"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check that all components are reachable and the collection exists."""
    from qdrant_client import QdrantClient
    qdrant_url  = os.getenv("QDRANT_URL", "http://localhost:6333")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

    qdrant_status  = "unreachable"
    vectors_count  = None
    collection     = "legal_cases"

    try:
        client     = QdrantClient(url=qdrant_url)
        info       = client.get_collection(collection)
        qdrant_status = "ok"
        vectors_count = info.points_count
    except Exception as e:
        qdrant_status = f"error: {str(e)}"

    return HealthResponse(
        status        = "ok",
        qdrant        = qdrant_status,
        ollama_model  = ollama_model,
        collection    = collection,
        vectors_count = vectors_count,
    )


@app.get("/collection/info", response_model=CollectionInfoResponse, tags=["System"])
def collection_info():
    """Return stats about the Qdrant vector collection."""
    from qdrant_client import QdrantClient
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        client = QdrantClient(url=qdrant_url)
        info   = client.get_collection("legal_cases")
        config = info.config.params.vectors
        return CollectionInfoResponse(
            collection    = "legal_cases",
            vectors_count = info.points_count,
            vector_dim    = config.size,
            distance      = str(config.distance),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant error: {str(e)}")


@app.post("/retrieve", response_model=QueryResponse, tags=["RAG"])
def retrieve_only(req: QueryRequest):
    """
    Retrieve relevant cases for a query without calling the LLM.
    Faster — useful for testing retrieval quality independently.
    """
    t0 = time.time()

    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised yet.")

    cases = retriever.retrieve(req.query, top_k_cases=req.top_k)
    retrieval_ms = int((time.time() - t0) * 1000)

    case_results = [
        CaseResult(
            case_name     = c.case_name,
            author_name   = c.author_name,
            category      = c.category,
            date_filed    = c.date_filed,
            absolute_url  = c.absolute_url,
            court         = c.court,
            jurisdiction  = c.jurisdiction,
            score         = c.score,
            chunk_preview = c.best_chunk[:300],
        )
        for c in cases
    ]

    return QueryResponse(
        query        = req.query,
        cases        = case_results,
        memo         = None,
        model_used   = None,
        retrieval_ms = retrieval_ms,
        total_ms     = retrieval_ms,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(req: QueryRequest):
    """
    Main RAG endpoint.
    Retrieves relevant precedents and generates a legal research memo.

    - Set generate=false to skip LLM and return cases only.
    - top_k controls how many cases are retrieved (1-10, default 5).
    """
    t0 = time.time()

    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised yet.")

    # Step 1 — Retrieve
    t_ret = time.time()
    cases = retriever.retrieve(req.query, top_k_cases=req.top_k)
    retrieval_ms = int((time.time() - t_ret) * 1000)

    if not cases:
        raise HTTPException(
            status_code=404,
            detail="No relevant cases found for this query. Try rephrasing or broadening your search."
        )

    case_results = [
        CaseResult(
            case_name     = c.case_name,
            author_name   = c.author_name,
            category      = c.category,
            date_filed    = c.date_filed,
            absolute_url  = c.absolute_url,
            court         = c.court,
            jurisdiction  = c.jurisdiction,
            score         = c.score,
            chunk_preview = c.best_chunk[:300],
        )
        for c in cases
    ]

    # Step 2 — Generate (optional)
    memo_text     = None
    model_used    = None
    generation_ms = None

    if req.generate:
        t_gen = time.time()
        result = generate_memo(req.query, cases)
        generation_ms = int((time.time() - t_gen) * 1000)
        memo_text  = result.memo
        model_used = result.model_used

    total_ms = int((time.time() - t0) * 1000)

    return QueryResponse(
        query         = req.query,
        cases         = case_results,
        memo          = memo_text,
        model_used    = model_used,
        retrieval_ms  = retrieval_ms,
        generation_ms = generation_ms,
        total_ms      = total_ms,
    )


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
# Legal Precedent RAG API

A production-grade Retrieval-Augmented Generation (RAG) system for US legal case research. Given a legal query, the system retrieves semantically relevant precedents from a vector database and generates a structured legal research memo using a local LLM.

**Built on published research:** [arXiv:2406.01609](https://arxiv.org/abs/2406.01609) — extended from a Streamlit prototype into a deployable REST API.

---

## What it does

A lawyer or researcher submits a query like:

> *"Fourth Amendment unlawful search and seizure warrant requirement"*

The API returns:
1. The top-k most semantically relevant precedent cases from US courts
2. A structured legal research memo with citations, written by a local LLM
3. Relevance scores, court names, dates, and direct CourtListener URLs for every case

---

## Architecture

```
User Query
    │
    ▼
FastAPI (api.py)
    │
    ├── Embed query with all-mpnet-base-v2 (sentence-transformers)
    │
    ├── ANN search → Qdrant vector DB (legal_cases collection)
    │         └── 768-dim cosine similarity over chunked opinion text
    │
    ├── Deduplicate + score-rank retrieved chunks by case
    │
    └── Ollama LLM (llama3.2:1b) → generates legal memo with citations
```

**Tech stack:**

| Layer | Tool |
|---|---|
| Embedding model | `sentence-transformers/all-mpnet-base-v2` |
| Vector database | Qdrant (local Docker / AWS) |
| LLM | Ollama — `llama3.2:1b` (runs locally, no API key needed) |
| API framework | FastAPI + Uvicorn |
| Dataset | [COLD Cases — Harvard LIL](https://huggingface.co/datasets/harvard-lil/cold-cases) (~8M US court opinions, current through 2024) |
| Containerisation | Docker (multi-stage build) |
| Deployment | AWS EC2 + ECR |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status, Qdrant connection, vector count |
| `GET` | `/collection/info` | Vector DB stats (dimensions, distance metric) |
| `POST` | `/query` | Full RAG — retrieve cases + generate legal memo |
| `POST` | `/retrieve` | Retrieval only — no LLM call, faster |

Interactive docs available at **`/docs`** (Swagger UI) when running locally.

### Example request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "First Amendment freedom of speech government censorship",
    "top_k": 5,
    "generate": true
  }'
```

### Example response

```json
{
  "query": "First Amendment freedom of speech government censorship",
  "cases": [
    {
      "case_name": "Washington Legal Foundation v. Texas Equal Access to Justice Foundation",
      "author_name": "Wisdom",
      "category": "majority",
      "date_filed": "1996-09-12",
      "court": "Fifth Circuit",
      "score": 0.4024,
      "absolute_url": "https://www.courtlistener.com/opinion/..."
    }
  ],
  "memo": "## Legal Research Memo\n\nThe query concerns First Amendment...",
  "model_used": "llama3.2:1b",
  "retrieval_ms": 38,
  "generation_ms": 4200,
  "total_ms": 4238
}
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com) with `llama3.2:1b` pulled

### 1. Clone and install

```bash
git clone https://github.com/HrushithaTigulla/legal-rag-api.git
cd legal-rag-api
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker run -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 3. Start Ollama

```bash
ollama serve
ollama pull llama3.2:1b
```

### 4. Ingest the dataset

```bash
# Quick test (500 cases, ~3 min)
python ingest.py --limit 500

# Full dataset (~8M cases, runs overnight)
python ingest.py
```

### 5. Start the API

```bash
uvicorn api:app --reload --port 8000
```

Open **`http://localhost:8000/docs`** for the interactive Swagger UI.

---

## Run with Docker

```bash
# Build
docker build -t legal-rag-api .

# Run (Qdrant and Ollama must be running on host)
docker run -p 8000:8000 legal-rag-api
```

---

## Evaluation

A custom evaluation harness runs 10 legal test queries and measures retrieval quality, keyword coverage, and LLM memo quality.

```bash
# Retrieval only (fast)
python evaluate.py --no-generate

# Full evaluation including LLM generation
python evaluate.py
```

Outputs a `evaluation_report.json` with per-query scores, keyword hit rates, latency, and court diversity metrics.

---

## Dataset

**COLD Cases** — Harvard Library Innovation Lab  
`harvard-lil/cold-cases` on HuggingFace (~8.3M US court opinions)

- Full majority and dissenting opinion text
- Metadata: case name, author/justice, court, date filed, CourtListener URL
- Coverage: federal circuit courts, Supreme Court, state supreme courts through 2024
- Loaded directly via `datasets.load_dataset()` — no manual download

**Original research dataset:** Kaggle SCOTUS opinions (Supreme Court only, pre-2023) — replaced by COLD Cases for broader coverage and current data.

---

## Project background

This project extends published research on legal precedent retrieval ([arXiv:2406.01609](https://arxiv.org/abs/2406.01609), presented at MIWAI 2024). The original system used:
- Google's Universal Sentence Encoder (2019) for embeddings
- KMeans clustering + SVM classifier for retrieval routing
- Flat CSV file (35MB) as the vector store
- Streamlit UI with no API layer

This version replaces each component with production-grade tooling while preserving the core retrieval objective.

---

## Repository structure

```
legal-rag-api/
├── api.py          # FastAPI application — all endpoints
├── retriever.py    # Embedding + Qdrant search + deduplication
├── generator.py    # Ollama LLM prompt builder and caller
├── ingest.py       # Dataset loading, chunking, embedding, Qdrant upload
├── evaluate.py     # Evaluation harness — retrieval and generation quality
├── Dockerfile      # Multi-stage Docker build
├── .dockerignore
├── requirements.txt
└── .env            # Local config (not committed)
```

---

## License

MIT
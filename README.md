# Legal Precedent RAG API

A production-grade Retrieval-Augmented Generation (RAG) system for US legal case research. Given a legal query, the system retrieves semantically relevant precedents from a vector database and generates a structured legal research memo using a local LLM.

**Built on published research:** [arXiv:2406.01609](https://arxiv.org/abs/2406.01609) — extended from a Streamlit prototype into a deployable REST API with evaluation framework.

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
User Query (Streamlit UI)
    │
    ▼
FastAPI (api.py)
    │
    ├── LangChain HuggingFaceEmbeddings → embed query (all-mpnet-base-v2)
    │
    ├── ANN search → Qdrant vector DB (legal_cases collection)
    │         └── 768-dim cosine similarity over chunked opinion text
    │
    ├── Cohere Rerank v3 → neural reranking of top results
    │
    └── Ollama LLM (llama3.2:1b) → generates legal memo with citations
```

**Tech stack:**

| Layer | Tool |
|---|---|
| RAG Orchestration | LangChain (HuggingFaceEmbeddings + QdrantVectorStore) |
| Embedding model | `sentence-transformers/all-mpnet-base-v2` (768-dim) |
| Vector database | Qdrant (cosine similarity, HNSW index) |
| Reranking | Cohere Rerank v3 |
| LLM | Ollama — `llama3.2:1b` (runs locally, no API key needed) |
| API framework | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Experiment tracking | MLflow |
| Dataset | [COLD Cases — Harvard LIL](https://huggingface.co/datasets/harvard-lil/cold-cases) (~8.3M US court opinions, current through 2024) |
| Containerisation | Docker |

---

## Evaluation Results

Custom evaluation harness (`evaluate.py`) with MLflow experiment tracking across 10 standardised legal queries:

| Metric | Score |
|---|---|
| Coverage | 10/10 (100%) |
| Avg cosine similarity | 0.5899 |
| Keyword hit rate | 73.5% |
| Avg retrieval latency | 96ms |
| Memo structure (4-check) | 4/4 on all queries |

**Before vs After HTML stripping + larger dataset:**

| | Baseline (500 cases, HTML noise) | Final (50k cases, clean text) |
|---|---|---|
| Avg cosine score | 0.4576 | 0.5899 (+29%) |
| Keyword hit rate | 19.5% | 73.5% (+277%) |
| Avg latency | 212ms | 96ms (-55%) |

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
  "retrieval_ms": 96,
  "generation_ms": 65000,
  "total_ms": 65096
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
git clone https://github.com/Hrushitha12/legal-rag-api.git
cd legal-rag-api
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install streamlit datasets tqdm mlflow ragas
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
# Quick test (500 cases, ~5 min)
python ingest.py --limit 500

# Development subset used in this project (50k cases, ~4 hours)
python ingest.py --limit 50000

# Full dataset (8.3M cases, overnight)
python ingest.py
```

### 5. Start the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 6. Start the Streamlit frontend

```bash
# In a new terminal
API_BASE_URL=http://127.0.0.1:8000 streamlit run streamlit_app.py
```

Open **`http://localhost:8501`**

---

## Run with Docker

```bash
docker build -t legal-rag-api .
docker run -p 8000:8000 \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  legal-rag-api
```

---

## Evaluation

```bash
# Retrieval only (fast, ~2 min)
python evaluate.py --no-generate

# Full evaluation with LLM generation and RAGAS scoring
python evaluate.py

# View results in MLflow UI
mlflow ui    # open http://localhost:5000
```

RAGAS metrics measured: faithfulness, answer relevancy, context recall (requires Ollama running as judge LLM).

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

This project extends published research on legal precedent retrieval ([arXiv:2406.01609](https://arxiv.org/abs/2406.01609), presented at MIWAI 2024).

**Original system → This system:**

| Component | Original | New |
|---|---|---|
| Embeddings | Universal Sentence Encoder (2019, 512-dim) | all-mpnet-base-v2 (2021, 768-dim) |
| Retrieval | KMeans + SVM + Euclidean distance | LangChain + Qdrant ANN + Cohere Rerank |
| Storage | 35MB flat CSV | Qdrant vector DB (HNSW index) |
| Generation | None — ranked list only | Ollama LLM → legal memo with citations |
| Interface | Streamlit UI only | FastAPI REST API + Streamlit frontend |
| Evaluation | None | MLflow tracking + RAGAS framework |

---

## Repository structure

```
legal-rag-api/
├── api.py              # FastAPI application — all endpoints
├── retriever.py        # LangChain + Qdrant search + Cohere reranking
├── generator.py        # Ollama LLM prompt builder and caller
├── ingest.py           # Dataset loading, HTML stripping, chunking, embedding
├── evaluate.py         # Evaluation harness with MLflow + RAGAS
├── streamlit_app.py    # Streamlit frontend
├── Dockerfile          # Docker build
├── .dockerignore
├── requirements.txt
└── PROJECT_DOCUMENTATION.md  # Full engineering decisions doc
```

---

## License

MIT
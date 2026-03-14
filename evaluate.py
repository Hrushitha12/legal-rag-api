"""
evaluate.py
-----------
Evaluation harness for the Legal RAG pipeline with MLflow + RAGAS scoring.

Measures:
  Retrieval : cosine score, keyword hit rate, latency, coverage
  Generation: RAGAS faithfulness, answer relevance, context recall
  Structure : citation presence, length, truncation checks

Run:
    python evaluate.py                # full eval with LLM + RAGAS
    python evaluate.py --no-generate  # retrieval only (fast, no LLM)
    python evaluate.py --no-ragas     # generation without RAGAS scoring

View MLflow results:
    mlflow ui   →   http://localhost:5000
"""

import argparse
import json
import time
import os
from datetime import datetime

import mlflow
from dotenv import load_dotenv
load_dotenv()

from retriever import get_retriever, EMBEDDING_MODEL, COLLECTION_NAME, TOP_K_CASES
from generator import generate_memo, OLLAMA_MODEL


# ── Test queries ──────────────────────────────────────────────────────────────

TEST_QUERIES = [
    {
        "id"               : "Q1",
        "query"            : "First Amendment freedom of speech government censorship",
        "expected_keywords": ["speech", "amendment", "first", "constitution"],
    },
    {
        "id"               : "Q2",
        "query"            : "Fourth Amendment unlawful search and seizure warrant",
        "expected_keywords": ["search", "seizure", "warrant", "fourth"],
    },
    {
        "id"               : "Q3",
        "query"            : "equal protection racial discrimination employment civil rights",
        "expected_keywords": ["discrimination", "equal", "civil", "rights", "race"],
    },
    {
        "id"               : "Q4",
        "query"            : "due process right to a fair trial criminal defendant",
        "expected_keywords": ["due process", "trial", "defendant", "criminal"],
    },
    {
        "id"               : "Q5",
        "query"            : "habeas corpus unlawful detention prisoner rights",
        "expected_keywords": ["habeas", "detention", "prisoner", "custody"],
    },
]

# RAGAS uses only 5 queries — LLM-as-judge is slow and costly on free keys.
# The retrieval-only evaluation still runs all 10.
RAGAS_QUERIES = TEST_QUERIES[:5]


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def keyword_hit(cases, expected_keywords):
    combined = " ".join([(c.case_name + " " + c.best_chunk).lower() for c in cases])
    hits   = [kw for kw in expected_keywords if kw.lower() in combined]
    misses = [kw for kw in expected_keywords if kw.lower() not in combined]
    return {
        "hit_count": len(hits),
        "total"    : len(expected_keywords),
        "hit_rate" : round(len(hits) / len(expected_keywords), 2),
        "hits"     : hits,
        "misses"   : misses,
    }


def check_memo_structure(memo: str) -> dict:
    if memo.startswith("ERROR"):
        return {"passed": 0, "total_checks": 4, "score": "0/4",
                "details": {"error": memo[:100]}}
    checks = {
        "has_numbered_citation"   : any(f"[{i}]" in memo for i in range(1, 6)),
        "has_cited_cases_section" : "cited" in memo.lower(),
        "min_length_80_words"     : len(memo.split()) >= 80,
        "not_truncated"           : not memo.rstrip().endswith(("...", "…")),
    }
    passed = sum(checks.values())
    return {"passed": passed, "total_checks": 4,
            "score": f"{passed}/4", "details": checks}


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def run_ragas(query_results: list) -> dict:
    """
    Score the RAG pipeline using RAGAS metrics:
      - Faithfulness     : are all claims in the memo grounded in retrieved context?
      - Answer relevance : does the memo actually answer the query?
      - Context recall   : did the memo use the retrieved context well?

    Uses Ollama as the judge LLM (no OpenAI key needed).
    """
    print("\nRunning RAGAS evaluation...")
    print("(Uses Ollama as judge LLM — takes 2-5 min)")

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.llms import Ollama
        from langchain_huggingface import HuggingFaceEmbeddings
        from datasets import Dataset
    except ImportError as e:
        print(f"RAGAS import error: {e}")
        print("Run: pip install ragas langchain-community")
        return {}

    # Build dataset — RAGAS expects lists of questions, answers, contexts
    questions  = []
    answers    = []
    contexts   = []
    # ground_truth is required by context_recall — we use the query itself
    # as a proxy (standard practice when no gold labels exist)
    ground_truths = []

    for r in query_results:
        if not r.get("memo") or r["memo"].startswith("ERROR"):
            continue
        questions.append(r["query"])
        answers.append(r["memo"])
        contexts.append(r["context_chunks"])   # list of chunk strings
        ground_truths.append(r["query"])       # proxy ground truth

    if not questions:
        print("No valid memo outputs to evaluate with RAGAS.")
        return {}

    dataset = Dataset.from_dict({
        "question"     : questions,
        "answer"       : answers,
        "contexts"     : contexts,
        "ground_truth" : ground_truths,
    })

    # RAGAS uses a SEPARATE judge model from the generation model.
    # Generation uses OLLAMA_MODEL (llama3.2:1b — fast on CPU).
    # Judging uses RAGAS_JUDGE_MODEL (llama3.1:8b — better reasoning).
    # This way memos generate in ~60s and RAGAS gets a capable judge.
    ollama_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    judge_model  = os.getenv("RAGAS_JUDGE_MODEL", "llama3.1:8b")
    print(f"Judge LLM  : {judge_model}")
    print(f"(Generation used: {os.getenv('OLLAMA_MODEL', 'llama3.2:1b')})")

    try:
        judge_llm = LangchainLLMWrapper(
            Ollama(model=judge_model, base_url=ollama_url, temperature=0,
                   timeout=300)
        )
        judge_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name    = EMBEDDING_MODEL,
                model_kwargs  = {"device": "cpu"},
                encode_kwargs = {"normalize_embeddings": True},
            )
        )
    except Exception as e:
        print(f"Could not initialise Ollama for RAGAS: {e}")
        print("Make sure Ollama is running: ollama serve")
        return {}

    # Set judge LLM and embeddings on each metric
    metrics = [faithfulness, answer_relevancy, context_recall]
    for m in metrics:
        m.llm        = judge_llm
        m.embeddings = judge_emb

    try:
        scores = evaluate(dataset, metrics=metrics)
        result = {
            "faithfulness"    : round(float(scores["faithfulness"]), 4),
            "answer_relevancy": round(float(scores["answer_relevancy"]), 4),
            "context_recall"  : round(float(scores["context_recall"]), 4),
        }
        print(f"\nRAGAS Results:")
        print(f"  Faithfulness     : {result['faithfulness']} (target > 0.8)")
        print(f"  Answer relevancy : {result['answer_relevancy']} (target > 0.8)")
        print(f"  Context recall   : {result['context_recall']} (target > 0.7)")
        return result

    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        print("This can happen if Ollama times out during judge LLM calls.")
        print("Try running with --no-ragas and add RAGAS separately later.")
        return {}


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(generate: bool = True, run_ragas_eval: bool = True):

    mlflow.set_experiment("legal-rag-evaluation")
    run_name = (
        f"eval-{'gen' if generate else 'ret'}"
        f"{'-ragas' if run_ragas_eval and generate else ''}"
        f"-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    with mlflow.start_run(run_name=run_name):

        print("=" * 60)
        print("LEGAL RAG EVALUATION HARNESS")
        print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"MLflow    : {run_name}")
        print(f"Generate  : {generate} | RAGAS: {run_ragas_eval and generate}")
        print("=" * 60)

        retriever = get_retriever()

        mlflow.log_params({
            "embedding_model"    : EMBEDDING_MODEL,
            "collection_name"    : COLLECTION_NAME,
            "top_k_cases"        : TOP_K_CASES,
            "generation_enabled" : generate,
            "ragas_enabled"      : run_ragas_eval and generate,
            "ollama_model"       : OLLAMA_MODEL if generate else "N/A",
            "num_test_queries"   : len(TEST_QUERIES),
        })

        try:
            from qdrant_client import QdrantClient
            qc = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
            vectors_count = qc.get_collection(COLLECTION_NAME).points_count
            mlflow.log_param("vectors_in_collection", vectors_count)
        except Exception:
            pass

        results          = []
        retrieval_times  = []
        scores_all       = []
        keyword_rates    = []
        memo_scores      = []
        queries_with_results = 0

        for tq in TEST_QUERIES:
            qid   = tq["id"]
            query = tq["query"]
            print(f"\n{qid}: {query[:60]}...")

            t0     = time.time()
            cases  = retriever.retrieve(query, top_k_cases=5)
            ret_ms = int((time.time() - t0) * 1000)
            retrieval_times.append(ret_ms)

            if not cases:
                print(f"  [!] No results")
                results.append({"id": qid, "query": query, "results_count": 0})
                continue

            queries_with_results += 1
            scores = [c.score for c in cases]
            scores_all.extend(scores)

            kw     = keyword_hit(cases, tq["expected_keywords"])
            keyword_rates.append(kw["hit_rate"])
            courts = list(set(c.court for c in cases if c.court))
            dates  = [c.date_filed[:4] for c in cases if c.date_filed]

            print(f"  Retrieval: {len(cases)} cases | top: {max(scores):.4f} | avg: {sum(scores)/len(scores):.4f}")
            print(f"  Keywords : {kw['hit_count']}/{kw['total']} ({kw['hit_rate']*100:.0f}%) misses: {kw['misses']}")
            print(f"  Courts   : {', '.join(courts[:3])}")
            print(f"  Latency  : {ret_ms}ms")

            mlflow.log_metrics({
                f"{qid}_top_score"       : round(max(scores), 4),
                f"{qid}_avg_score"       : round(sum(scores) / len(scores), 4),
                f"{qid}_keyword_hit_rate": kw["hit_rate"],
                f"{qid}_retrieval_ms"    : ret_ms,
            })

            memo_text      = None
            memo_structure = None
            gen_ms         = None
            context_chunks = [c.best_chunk for c in cases]

            if generate:
                t1     = time.time()
                result = generate_memo(query, cases[:3])
                gen_ms = int((time.time() - t1) * 1000)
                memo_text      = result.memo
                memo_structure = check_memo_structure(result.memo)
                memo_scores.append(memo_structure["passed"])
                print(f"  Memo     : {gen_ms}ms | structure: {memo_structure['score']}")
                mlflow.log_metric(f"{qid}_generation_ms", gen_ms)
                mlflow.log_metric(f"{qid}_memo_structure", memo_structure["passed"])

            results.append({
                "id"              : qid,
                "query"           : query,
                "results_count"   : len(cases),
                "top_score"       : round(max(scores), 4),
                "avg_score"       : round(sum(scores) / len(scores), 4),
                "keyword_hit_rate": kw["hit_rate"],
                "keyword_misses"  : kw["misses"],
                "courts"          : courts,
                "date_range"      : f"{min(dates)} – {max(dates)}" if dates else "",
                "retrieval_ms"    : ret_ms,
                "generation_ms"   : gen_ms,
                "memo_structure"  : memo_structure,
                "top_case"        : cases[0].case_name,
                "memo"            : memo_text,
                "context_chunks"  : context_chunks,
            })

        # ── Summary ───────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("RETRIEVAL SUMMARY")
        print("=" * 60)

        coverage    = queries_with_results / len(TEST_QUERIES)
        avg_ret_ms  = sum(retrieval_times) / len(retrieval_times)
        avg_score   = sum(scores_all) / len(scores_all) if scores_all else 0
        avg_kw_rate = sum(keyword_rates) / len(keyword_rates) if keyword_rates else 0

        print(f"Coverage             : {queries_with_results}/{len(TEST_QUERIES)} ({coverage*100:.0f}%)")
        print(f"Avg cosine score     : {avg_score:.4f}  (target > 0.60)")
        print(f"Avg keyword hit rate : {avg_kw_rate*100:.1f}%  (target > 60%)")
        print(f"Avg retrieval latency: {avg_ret_ms:.0f}ms")

        if generate and memo_scores:
            avg_memo = sum(memo_scores) / len(memo_scores)
            print(f"Avg memo structure   : {avg_memo:.1f}/4 checks passed")

        summary_metrics = {
            "coverage"            : round(coverage, 2),
            "avg_retrieval_score" : round(avg_score, 4),
            "avg_keyword_hit_rate": round(avg_kw_rate, 4),
            "avg_retrieval_ms"    : round(avg_ret_ms, 1),
        }
        if generate and memo_scores:
            summary_metrics["avg_memo_structure"] = round(
                sum(memo_scores) / len(memo_scores), 2)

        # ── RAGAS ─────────────────────────────────────────────────────────────
        ragas_scores = {}
        if generate and run_ragas_eval:
            ragas_results = [r for r in results if r.get("memo")]
            ragas_scores  = run_ragas(ragas_results)
            if ragas_scores:
                print("\nRAGAS SUMMARY")
                print("=" * 60)
                for k, v in ragas_scores.items():
                    print(f"  {k:<22}: {v}")
                summary_metrics.update(ragas_scores)
                mlflow.log_metrics(ragas_scores)

        mlflow.log_metrics(summary_metrics)

        # Save full report
        summary = {
            "timestamp"           : datetime.now().isoformat(),
            "mlflow_run"          : run_name,
            "total_queries"       : len(TEST_QUERIES),
            "coverage"            : round(coverage, 2),
            "avg_retrieval_score" : round(avg_score, 4),
            "avg_keyword_hit_rate": round(avg_kw_rate, 4),
            "avg_retrieval_ms"    : round(avg_ret_ms, 1),
            "ragas_scores"        : ragas_scores,
            "generation_enabled"  : generate,
            "query_results"       : [
                {k: v for k, v in r.items() if k != "context_chunks"}
                for r in results
            ],
        }

        with open("evaluation_report.json", "w") as f:
            json.dump(summary, f, indent=2)

        mlflow.log_artifact("evaluation_report.json")

        print(f"\nReport saved : evaluation_report.json")
        print(f"MLflow UI    : run 'mlflow ui' → http://localhost:5000")

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip LLM generation, retrieval only")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Skip RAGAS scoring, run generation only")
    args = parser.parse_args()
    run_evaluation(
        generate      = not args.no_generate,
        run_ragas_eval= not args.no_ragas,
    )
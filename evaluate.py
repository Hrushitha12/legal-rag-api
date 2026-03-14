"""
evaluate.py
-----------
Evaluation harness for the Legal RAG pipeline with MLflow experiment tracking.

Every run is logged to MLflow with:
  - Metrics  : coverage, avg score, keyword hit rate, latency, memo quality
  - Params   : model name, top_k, collection name, vector count
  - Artifacts: full evaluation_report.json

Run:
    python evaluate.py                # full evaluation with LLM generation
    python evaluate.py --no-generate  # retrieval only (faster)

View results in MLflow UI:
    mlflow ui
    open http://localhost:5000
"""

import argparse
import json
import time
from datetime import datetime

import mlflow
import mlflow.artifacts

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
        "query"            : "interstate commerce clause federal regulation state law",
        "expected_keywords": ["commerce", "federal", "state", "regulation"],
    },
    {
        "id"               : "Q6",
        "query"            : "habeas corpus unlawful detention prisoner rights",
        "expected_keywords": ["habeas", "detention", "prisoner", "custody"],
    },
    {
        "id"               : "Q7",
        "query"            : "contract breach damages remedies commercial dispute",
        "expected_keywords": ["contract", "breach", "damages", "commercial"],
    },
    {
        "id"               : "Q8",
        "query"            : "negligence tort liability personal injury standard of care",
        "expected_keywords": ["negligence", "injury", "tort", "liability"],
    },
    {
        "id"               : "Q9",
        "query"            : "Second Amendment right to bear arms gun control regulation",
        "expected_keywords": ["second", "amendment", "arms", "gun", "firearm"],
    },
    {
        "id"               : "Q10",
        "query"            : "immigration deportation asylum refugee status federal law",
        "expected_keywords": ["immigration", "deportation", "asylum", "alien"],
    },
]


# ── Evaluation helpers ────────────────────────────────────────────────────────

def keyword_hit(cases, expected_keywords):
    combined = " ".join([
        (c.case_name + " " + c.best_chunk).lower()
        for c in cases
    ])
    hits   = [kw for kw in expected_keywords if kw.lower() in combined]
    misses = [kw for kw in expected_keywords if kw.lower() not in combined]
    return {
        "hit_count" : len(hits),
        "total"     : len(expected_keywords),
        "hit_rate"  : round(len(hits) / len(expected_keywords), 2),
        "hits"      : hits,
        "misses"    : misses,
    }


def check_memo_quality(memo: str) -> dict:
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
    return {
        "passed"      : passed,
        "total_checks": len(checks),
        "score"       : f"{passed}/{len(checks)}",
        "details"     : checks,
    }


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(generate: bool = True):

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_experiment("legal-rag-evaluation")
    run_name = f"eval-{'gen' if generate else 'retrieval'}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name):

        print("=" * 60)
        print("LEGAL RAG EVALUATION HARNESS")
        print(f"Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"MLflow run : {run_name}")
        print(f"Generate   : {generate}")
        print("=" * 60)

        retriever = get_retriever()

        # Log run parameters to MLflow
        mlflow.log_params({
            "embedding_model"   : EMBEDDING_MODEL,
            "collection_name"   : COLLECTION_NAME,
            "top_k_cases"       : TOP_K_CASES,
            "generation_enabled": generate,
            "ollama_model"      : OLLAMA_MODEL if generate else "N/A",
            "num_test_queries"  : len(TEST_QUERIES),
        })

        # Get vector count from Qdrant and log it
        try:
            from qdrant_client import QdrantClient
            import os
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

            # Retrieval
            t0    = time.time()
            cases = retriever.retrieve(query, top_k_cases=5)
            ret_ms = int((time.time() - t0) * 1000)
            retrieval_times.append(ret_ms)

            if not cases:
                print(f"  [!] No results returned")
                results.append({"id": qid, "query": query, "results_count": 0})
                continue

            queries_with_results += 1
            scores = [c.score for c in cases]
            scores_all.extend(scores)

            kw = keyword_hit(cases, tq["expected_keywords"])
            keyword_rates.append(kw["hit_rate"])

            courts = list(set(c.court for c in cases if c.court))
            dates  = [c.date_filed[:4] for c in cases if c.date_filed]

            print(f"  Results  : {len(cases)} | top: {max(scores):.4f} | avg: {sum(scores)/len(scores):.4f}")
            print(f"  Keywords : {kw['hit_count']}/{kw['total']} ({kw['hit_rate']*100:.0f}%) — misses: {kw['misses']}")
            print(f"  Courts   : {', '.join(courts[:3])}")
            print(f"  Latency  : {ret_ms}ms")

            # Log per-query metrics to MLflow
            mlflow.log_metrics({
                f"{qid}_top_score"       : round(max(scores), 4),
                f"{qid}_avg_score"       : round(sum(scores)/len(scores), 4),
                f"{qid}_keyword_hit_rate": kw["hit_rate"],
                f"{qid}_retrieval_ms"    : ret_ms,
            })

            # Generation
            memo_quality = None
            gen_ms       = None

            if generate:
                t1     = time.time()
                result = generate_memo(query, cases[:3])
                gen_ms = int((time.time() - t1) * 1000)
                memo_quality = check_memo_quality(result.memo)
                memo_scores.append(memo_quality["passed"])
                print(f"  Memo     : {gen_ms}ms | quality: {memo_quality['score']}")
                mlflow.log_metric(f"{qid}_generation_ms", gen_ms)
                mlflow.log_metric(f"{qid}_memo_quality",  memo_quality["passed"])

            results.append({
                "id"              : qid,
                "query"           : query,
                "results_count"   : len(cases),
                "top_score"       : round(max(scores), 4),
                "avg_score"       : round(sum(scores)/len(scores), 4),
                "keyword_hit_rate": kw["hit_rate"],
                "keyword_misses"  : kw["misses"],
                "courts"          : courts,
                "retrieval_ms"    : ret_ms,
                "generation_ms"   : gen_ms,
                "memo_quality"    : memo_quality,
                "top_case"        : cases[0].case_name,
            })

        # ── Summary ───────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        coverage    = queries_with_results / len(TEST_QUERIES)
        avg_ret_ms  = sum(retrieval_times) / len(retrieval_times)
        avg_score   = sum(scores_all) / len(scores_all) if scores_all else 0
        avg_kw_rate = sum(keyword_rates) / len(keyword_rates) if keyword_rates else 0

        print(f"Queries run          : {len(TEST_QUERIES)}")
        print(f"Coverage             : {queries_with_results}/{len(TEST_QUERIES)} ({coverage*100:.0f}%)")
        print(f"Avg retrieval score  : {avg_score:.4f}")
        print(f"Avg keyword hit rate : {avg_kw_rate*100:.1f}%")
        print(f"Avg retrieval latency: {avg_ret_ms:.0f}ms")

        if generate and memo_scores:
            avg_memo = sum(memo_scores) / len(memo_scores)
            print(f"Avg memo quality     : {avg_memo:.1f}/4 checks passed")

        # Log summary metrics to MLflow
        summary_metrics = {
            "coverage"            : round(coverage, 2),
            "avg_retrieval_score" : round(avg_score, 4),
            "avg_keyword_hit_rate": round(avg_kw_rate, 4),
            "avg_retrieval_ms"    : round(avg_ret_ms, 1),
        }
        if generate and memo_scores:
            summary_metrics["avg_memo_quality"] = round(
                sum(memo_scores) / len(memo_scores), 2
            )
        mlflow.log_metrics(summary_metrics)

        # Save and log the full report as an MLflow artifact
        summary = {
            "timestamp"            : datetime.now().isoformat(),
            "mlflow_run"           : run_name,
            "total_queries"        : len(TEST_QUERIES),
            "coverage"             : round(coverage, 2),
            "avg_retrieval_score"  : round(avg_score, 4),
            "avg_keyword_hit_rate" : round(avg_kw_rate, 4),
            "avg_retrieval_ms"     : round(avg_ret_ms, 1),
            "generation_enabled"   : generate,
            "query_results"        : results,
        }

        with open("evaluation_report.json", "w") as f:
            json.dump(summary, f, indent=2)

        mlflow.log_artifact("evaluation_report.json")
        print(f"\nFull report    : evaluation_report.json")
        print(f"MLflow run     : {run_name}")
        print(f"View in UI     : run 'mlflow ui' then open http://localhost:5000")

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip LLM generation, test retrieval only")
    args = parser.parse_args()
    run_evaluation(generate=not args.no_generate)
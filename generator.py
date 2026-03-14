"""
generator.py
------------
Takes a user query + retrieved cases from retriever.py and calls
a local Ollama LLM to generate a structured legal memo with citations.

Ollama must be running locally:
    ollama serve          (if not already running as a service)
    ollama pull llama3.2  (if not already pulled)

Usage (standalone test):
    python generator.py
"""

import os
import re
import requests
import json
from dataclasses import dataclass
from retriever import RetrievedCase
from dotenv import load_dotenv

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
MAX_CHUNK_CHARS = 600   # how much of each case's text to include in the prompt
TIMEOUT_SECONDS = 120


# ── Output model ──────────────────────────────────────────────────────────────

@dataclass
class GeneratedMemo:
    query        : str
    memo         : str           # the full LLM-generated legal memo
    cited_cases  : list          # list of RetrievedCase objects actually cited
    model_used   : str


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_html_tags(text: str) -> str:
    """Remove any HTML tags from opinion text before sending to LLM."""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def build_prompt(query: str, cases: list) -> str:
    """
    Construct the prompt sent to the LLM.
    Cases are formatted as numbered context blocks so the LLM
    can cite them by number in its response.
    """
    context_blocks = []
    for i, case in enumerate(cases, 1):
        chunk = strip_html_tags(case.best_chunk)[:MAX_CHUNK_CHARS]
        block = (
            f"[{i}] Case: {case.case_name}\n"
            f"    Court: {case.court} | Date: {case.date_filed}\n"
            f"    Author: {case.author_name} | Type: {case.category}\n"
            f"    Relevant excerpt:\n    \"{chunk}\"\n"
            f"    URL: {case.absolute_url}"
        )
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are a legal research assistant. A lawyer has submitted the following query:

QUERY: {query}

Below are the most relevant precedent cases retrieved from the legal database:

{context_str}

Based on these cases, write a concise legal research memo that:
1. Summarises the key legal principles relevant to the query
2. Explains how each cited case is relevant, referencing them by name and number e.g. [1]
3. Identifies the strongest precedent and explains why
4. Notes any circuit splits or conflicting rulings if present
5. Ends with a "Cited Cases" section listing each case with its URL

Keep the memo professional, structured, and under 400 words.
Do not invent cases or facts not present in the provided excerpts."""

    return prompt


def call_ollama(prompt: str) -> str:
    """Send prompt to locally running Ollama and return the response text."""
    url     = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model" : OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,   # low temp = more factual, less creative
            "num_predict": 600,   # max tokens in response
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return (
            "ERROR: Could not connect to Ollama. "
            "Make sure Ollama is running: open a new terminal and run 'ollama serve'"
        )
    except requests.exceptions.Timeout:
        return "ERROR: Ollama request timed out. Try a smaller model or increase TIMEOUT_SECONDS."
    except Exception as e:
        return f"ERROR: Unexpected error calling Ollama: {str(e)}"


# ── Main generator function ───────────────────────────────────────────────────

def generate_memo(query: str, cases: list) -> GeneratedMemo:
    """
    Given a query and a list of RetrievedCase objects,
    generate a legal memo using the local LLM.
    """
    if not cases:
        return GeneratedMemo(
            query       = query,
            memo        = "No relevant cases were found for this query.",
            cited_cases = [],
            model_used  = OLLAMA_MODEL,
        )

    prompt = build_prompt(query, cases)
    memo   = call_ollama(prompt)

    return GeneratedMemo(
        query       = query,
        memo        = memo,
        cited_cases = cases,
        model_used  = OLLAMA_MODEL,
    )


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from retriever import get_retriever

    retriever = get_retriever()

    test_query = "Fourth Amendment unlawful search and seizure warrant requirement"

    print(f"Query: {test_query}")
    print("Retrieving cases...")
    cases = retriever.retrieve(test_query, top_k_cases=3)

    print(f"Retrieved {len(cases)} cases. Generating memo...\n")
    result = generate_memo(test_query, cases)

    print("=" * 60)
    print("LEGAL RESEARCH MEMO")
    print("=" * 60)
    print(result.memo)
    print("\n" + "=" * 60)
    print(f"Model used: {result.model_used}")
    print(f"Cases retrieved: {len(result.cited_cases)}")
    for i, c in enumerate(result.cited_cases, 1):
        print(f"  [{i}] {c.case_name} (score: {c.score})")
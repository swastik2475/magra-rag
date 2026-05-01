"""
eval/run_eval.py  –  Run the eval set and report results.

Usage:
    python eval/run_eval.py

Output:
    eval/results.json   – per-question results
    Prints summary table to stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add parent dir to path so we can import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from retriever import get_index, retrieve, load_chunk_lookup
from qa import answer


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_answer(result: dict, expected_contains: list[str], q_type: str) -> dict:
    """
    Score a single answer.

    Pass criteria:
      - refusal questions : answer must contain a refusal phrase
      - factual questions : answer must contain ALL expected strings (case-insensitive)
      - numeric questions : answer must contain all expected numbers
    """
    ans_lower = result["answer"].lower()

    if q_type == "refusal":
        passed = result["refused"] or any(
            p in ans_lower for p in ["don't know", "cannot", "not contain", "no information"]
        )
        reason = "correct refusal" if passed else "SHOULD HAVE REFUSED but gave an answer"
    else:
        # Each item in expected_contains can be a string (must match) or
        # a list of strings (ANY one must match — alternate formats OK)
        missing = []
        for item in expected_contains:
            if isinstance(item, list):
                if not any(alt.lower() in ans_lower for alt in item):
                    missing.append(item)
            else:
                if item.lower() not in ans_lower:
                    missing.append(item)
        passed = len(missing) == 0
        reason = "all expected values found" if passed else f"missing: {missing}"

    return {"passed": passed, "reason": reason}


# ── Main eval loop ─────────────────────────────────────────────────────────────

def run_eval(
    questions_path: str = "eval/questions.jsonl",
    results_path:   str = "eval/results.json",
    top_k: int = 8,
) -> None:
    # Load FAISS index + chunk lookup
    index        = get_index(persist_dir=os.getenv("FAISS_DIR", "./faiss_db"))
    chunk_lookup = load_chunk_lookup("./data/corpus_chunks.json")

    # Load questions
    questions = []
    with open(questions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"\nRunning eval on {len(questions)} questions …\n")

    results = []
    passed_count = 0

    for q in questions:
        qid   = q["id"]
        qtype = q["type"]
        qtext = q["question"]
        exp   = q.get("expected_contains", [])

        print(f"  [{qid}] {qtext[:65]} …", end=" ", flush=True)

        # Retrieve + answer
        chunks = retrieve(qtext, index, top_k=top_k, chunk_lookup=chunk_lookup)
        result = answer(qtext, chunks)
        score  = score_answer(result, exp, qtype)

        status = "✓ PASS" if score["passed"] else "✗ FAIL"
        print(status)

        if score["passed"]:
            passed_count += 1

        results.append({
            "id":          qid,
            "type":        qtype,
            "question":    qtext,
            "answer":      result["answer"],
            "citations":   result["citations"],
            "chunks_used": result["chunks_used"],
            "refused":     result["refused"],
            "passed":      score["passed"],
            "reason":      score["reason"],
            "expected":    exp,
        })

        time.sleep(0.3)   # avoid rate limit

    # Save results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    total = len(questions)
    rate  = passed_count / total * 100
    print(f"\n{'='*60}")
    print(f"PASS RATE: {passed_count}/{total}  ({rate:.0f}%)")
    print(f"{'='*60}")

    # Per-type breakdown
    types = {}
    for r in results:
        t = r["type"]
        types.setdefault(t, {"pass": 0, "total": 0})
        types[t]["total"] += 1
        if r["passed"]:
            types[t]["pass"] += 1

    print("\nBy question type:")
    for t, counts in types.items():
        tp = counts["pass"]
        tt = counts["total"]
        print(f"  {t:<30} {tp}/{tt}")

    # Show failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures:
            print(f"  [{r['id']}] {r['question'][:60]}")
            print(f"          Reason: {r['reason']}")
            print(f"          Answer snippet: {r['answer'][:120]} …")

    print(f"\nFull results saved → {results_path}")


if __name__ == "__main__":
    run_eval()



from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


SYSTEM_PROMPT = """You are a construction contract analyst.
Answer ONLY from the provided source excerpts below.

STRICT RULES:
1. Cite EVERY fact using the citation tag from each excerpt.
   e.g.  [Contract p.3 – contract.pdf]  or  [Schedule: Activity PROC-11]
2. If the excerpts do NOT contain enough info, say EXACTLY:
   "I don't know – the corpus does not contain enough information to answer this reliably."
3. Never invent numbers, dates, clause numbers, or activity IDs.
4. For numeric questions: show arithmetic step by step.
5. For clause questions: quote the clause (under 15 words) then explain.
6. Be concise and professional."""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        citation = c.get("citation", f"[source {i}]")
        parts.append(f"--- EXCERPT {i} {citation} ---\n{c['text']}\n")
    return "\n".join(parts)


def answer(
    question: str,
    chunks: list[dict],
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
) -> dict:
    """Answer a question grounded in retrieved chunks using Groq (free)."""
    if not chunks:
        return {
            "answer":      "I don't know – no relevant context was retrieved.",
            "citations":   [],
            "chunks_used": 0,
            "refused":     True,
        }

    context  = _build_context(chunks)
    user_msg = (
        f"SOURCE EXCERPTS:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the excerpts. Cite every fact."
    )

    response = _get_client().chat.completions.create(
        model       = model,
        temperature = temperature,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )

    answer_text = response.choices[0].message.content.strip()
    refused     = "i don't know" in answer_text.lower()
    citations   = [c["citation"] for c in chunks if c.get("citation", "") in answer_text]

    return {
        "answer":      answer_text,
        "citations":   citations,
        "chunks_used": len(chunks),
        "refused":     refused,
    }


def print_answer(question: str, result: dict) -> None:
    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("-" * 70)
    print(result["answer"])
    print("-" * 70)
    if result["citations"]:
        print("Citations:")
        for c in set(result["citations"]):
            print(f"  • {c}")
    print(f"[{result['chunks_used']} chunks | refused={result['refused']}]")
    print("=" * 70)

# WRITEUP.md

## Architecture

I went with a three-stage pipeline: **Ingest → Index → Answer**.

The idea was simple — get the text out cleanly first, then worry about retrieval. I spent most of my time on ingestion because I knew messy PDF extraction would kill retrieval quality downstream no matter how good the embeddings were.

**Ingest** (`ingest.py`): pymupdf handles per-page text extraction with a custom reading-order fix I had to write (more on that below). pdfplumber pulls tables out separately — I didn't want contract pricing tables getting mixed into prose chunks. pandas reads the schedule CSV row by row, one chunk per activity so Activity IDs stay addressable. SOV comes from a PDF, so pdfplumber table extraction handles that too.

**Index** (`retriever.py`): I used `all-MiniLM-L6-v2` from sentence-transformers for embeddings — runs locally, no API key, fast enough on CPU. Vectors go into a FAISS flat index with cosine similarity. I also added a keyword boost layer on top because I noticed early on that semantic search alone would sometimes miss exact Activity ID matches like `PROC-11` — they're rare tokens and embeddings don't always handle them well.

**Answer** (`qa.py`): Groq's `llama-3.3-70b-versatile` at temperature=0. The system prompt is strict — cite every fact inline, refuse if the corpus doesn't support an answer. I tested a few phrasings here and the "refuse explicitly" instruction needed to be direct or the model would hedge instead of refusing cleanly.

```
contract.pdf ──► extract_pdf() ──► chunk_pdf_pages() ──┐
schedule.csv ──► parse_schedule_csv()                  ├──► FAISS ──► retrieve() ──► Groq LLM ──► Answer
sov.pdf      ──► parse_sov_pdf()      ─────────────────┘
```

---

## OCR / Extraction Decision

This was the part I spent the most time debugging.

Page 30 of `contract.pdf` (Article 14 – Miscellaneous Provisions) has a two-column layout. pymupdf's default `get_text("blocks")` sorts by Y-coordinate, which means it reads across both columns simultaneously — so you'd get something like "§ 14.2.1 The Owner and Construction Manager shall..." immediately followed by a mid-sentence fragment from the right column. The legal text was completely scrambled.

My first attempt was pymupdf's `sort=True` flag inside `get_text("dict")` — it has built-in reading-order heuristics. It still failed on this page because the gap between columns was narrow enough that the heuristic couldn't separate them reliably.

What actually worked: I wrote a two-column detector that counts how many text blocks fall in the left half vs. right half of the page. If both halves have ≥ 35% of the total blocks, the page gets flagged. For flagged pages, I re-sort manually — left column blocks top→bottom, then right column blocks top→bottom. Simple, but it fixed the reading order correctly.

---

## Dependency / Environment Issues I Hit

This section is worth documenting because it affected stack choices.

**ChromaDB → FAISS switch:** I originally planned to use ChromaDB as the vector store. When I ran `pip install chromadb==1.0.0` on Windows, it tried to build `chroma-hnswlib` from source and failed with:

```
error: Microsoft Visual C++ 14.0 or greater is required.
Get it with "Microsoft C++ Build Tools"
```

Installing C++ Build Tools is a heavy dependency I didn't want to force on whoever runs this. I switched to `faiss-cpu` which ships pre-built Windows wheels — `pip install faiss-cpu` just works, no compiler needed. The switch also simplified the codebase: FAISS persists as two plain files (`faiss_index.bin` + `faiss_meta.json`) instead of a SQLite database with a server process.

**Groq version conflict:** After the FAISS switch, the first question I ran hit this error:

```
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```

Root cause: `groq==0.9.0` was pinned in requirements but my system had a newer `httpx` installed that removed the `proxies` argument. Fixed with `pip install groq --upgrade`. Updated requirements.txt to use a minimum version bound instead of a hard pin for groq to avoid this for whoever runs it next.

**Eval scoring bug:** Q02 (substantial completion date) and Q04 (notice to proceed date) were failing even though the answers were factually correct. The system was returning `06/30/2027` — which is exactly what the contract says — but the eval expected `June 30, 2027`. The scorer was doing AND logic: both formats had to appear in the answer. Fixed by changing `expected_contains` to OR-groups for date fields, and updating the scorer to handle list-of-lists as alternate formats. Both questions went FAIL → PASS without touching any retrieval or generation code.

---

## Retrieval Choices and Tradeoffs

| Choice | Why I made it | What I gave up |
|--------|--------------|----------------|
| `all-MiniLM-L6-v2` | Free, local, no rate limits, solid on English legal text | Slightly weaker than large commercial models on niche construction terms |
| FAISS `IndexFlatIP` | Pre-built Windows wheels, no compiler needed, exact search is fine at <10k chunks | Nothing meaningful at this scale — ANN approximation would be overkill |
| 350-word chunks | Fits roughly one contract clause; keeps retrieval signal tight | Loses some cross-clause context |
| 50-word overlap | Prevents clause boundaries from getting split across chunks | Small amount of duplication in the index |
| Tables as separate chunks | Pricing tables have numeric ground truth — mixing them with prose tanks retrieval on both | Need to handle two chunk types in retrieval |
| Keyword boost | Semantic search misses exact Activity ID matches too often | Slight latency increase, rare false positives |

**Where the retriever struggles:** Cross-document questions are the hardest case. If the linking term (e.g. "66-inch Force Main") shows up in both the contract and SOV, top-8 retrieval might not surface chunks from both sources in the same call. I'd fix this with a reranker pass over a larger candidate set.

---

## Eval Design and Results

I wrote 15 questions covering all five shapes from the brief. I'll be upfront — I wrote both the questions and the system, so the eval is self-referential and likely optimistic. A blind eval from someone who hasn't seen the documents would give a more honest number.

| Type | Count | Pass |
|------|-------|------|
| Single-source factual | 6 | 6/6 |
| Cross-source factual | 3 | 3/3 |
| Clause-grounded reasoning | 2 | 2/2 |
| Refusal (negative) | 2 | 2/2 |
| Numeric reconciliation | 2 | 2/2 |

**Overall: 15/15**

**How I define passing:** For factual questions, the answer must contain all expected values (case-insensitive). Date questions accept alternate formats — `06/30/2027` or `June 30, 2027` both count. Refusal questions pass if the answer contains a refusal phrase.

**Metric I care about beyond accuracy:** Citation coverage — what fraction of factual sentences in the answer include an inline citation. I'd target ≥ 90%. In construction contract work, a correct answer with no citation is still a liability — you can't audit it, and a PM won't trust it in a dispute.

---

## Tuning Iteration

Early runs had chunk size at 500 words. I noticed Q11 (monthly contingency limit) kept failing — the answer cited the wrong clause number.

When I looked at what was being retrieved, a single 500-word chunk was spanning both § 3.2.4 (Contingency) and § 3.2.5 (GMP acceptance). The LLM was reading both clauses in the same context and getting confused about which section number applied to which rule.

Dropping `max_words` to 350 separated them cleanly. Q11 went from FAIL to PASS, and nothing else broke. Small change, clear cause and effect.

---

## What I'd Do Next

1. **Reranker** — a cross-encoder like `ms-marco-MiniLM` to re-score top-20 down to top-8. I think this would directly fix the cross-document retrieval weakness.
2. **Structured schedule queries** — right now schedule questions go through embeddings like everything else. A pandas query layer for "which activities have float < 10 days?" would be more reliable and faster.
3. **Blind eval** — get someone unfamiliar with the documents to write 20 questions. My self-written eval is useful for regression but not for honest accuracy measurement.
4. **Citation verification** — post-process answers to check that every cited page/activity actually exists in the retrieved chunks. Catch phantom citations before they reach the user.
# Magra RAG – Construction Contract Q&A

| Component | Tool | Cost |
|-----------|------|------|
| LLM | Groq `llama-3.3-70b-versatile` | FREE (14,400 req/day) |
| Embeddings | `all-MiniLM-L6-v2` local | FREE (runs on CPU) |
| Vector DB | FAISS (local) | FREE |

---

## Setup

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Get FREE Groq API key
1. Go to https://console.groq.com
2. Sign up (Google/GitHub — no credit card)
3. Click **API Keys → Create new key**

### 3. Configure
```bash
cp .env.example .env
# Edit .env → set GROQ_API_KEY=gsk_xxxx
```

### 4. Build index (one time, ~2-3 mins)
```bash
python main.py --setup
```
Downloads embedding model (~90MB) on first run. Saves index to `./faiss_db/`.

### 5. Ask a question
```bash
python main.py --ask "What is the total contract value?"
python main.py --ask "When is substantial completion?"
python main.py --ask "What is the lead time for PROC-11?"
python main.py --ask "What is the liquidated damages rate per day?"
python main.py --ask "Does the SOV total tie out to line items?"

# Show retrieved chunks
python main.py --ask "What is the total contract value?" --verbose

# Force rebuild index
python main.py --setup --force-rebuild
```

## Run Evaluation
```bash
python eval/run_eval.py
```
Results saved to `eval/results.json`. Current pass rate: **15/15 (100%)**.

## Run Tests
```bash
python -m pytest tests/
```

## Project Structure
```
├── main.py              # CLI entry point
├── ingest.py            # PDF + CSV ingestion, OCR fix
├── retriever.py         # FAISS index + retrieval
├── qa.py                # Groq LLM answer generation
├── requirements.txt
├── .env.example
├── WRITEUP.md
├── data/
│   ├── contract.pdf
│   ├── schedule-construction.csv
│   └── schedule-of-values.pdf
├── eval/
│   ├── questions.jsonl  # 15-question eval set
│   ├── run_eval.py      # eval runner
│   └── results.json     # latest results (15/15 pass)
└── tests/
    └── test_retriever.py
```

## Index Files
FAISS saves two files in `./faiss_db/` (auto-created on `--setup`):
- `faiss_index.bin` — vector index
- `faiss_meta.json` — chunk metadata

Delete this folder to rebuild from scratch.

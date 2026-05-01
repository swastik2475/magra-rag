"""
ingest.py  –  Corpus ingestion for Magra RAG system.

Handles:
  - Contract PDF  : multi-column layout detection + reading-order fix
  - SOV PDF       : table-heavy single-column, pdfplumber for tables
  - Schedule CSV  : row-per-activity, addressable by Activity ID

OCR Decision (documented for WRITEUP.md):
  Page 30 of the contract (Article 14 - Miscellaneous) uses a two-column
  layout where pymupdf's default block sort reads across columns left→right
  instead of column-by-column, scrambling the legal text.
  Fix: detect pages where text blocks cluster in two horizontal bands
  (left-col x < 50% page width, right-col x > 50%) and re-sort blocks
  column-first (left top→bottom, then right top→bottom).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import fitz          # pymupdf
import pandas as pd
import pdfplumber
from tqdm import tqdm


#Constants

HEADER_FOOTER_RATIO = 0.06   # top/bottom 6% of page height stripped
MIN_BLOCK_CHARS     = 15     # ignore tiny fragments (page numbers, etc.)
TWO_COL_THRESHOLD   = 0.35   # if ≥35% blocks are in each half → two-column


#Two-column detection 

def _is_two_column(page: fitz.Page) -> bool:
    """Return True if the page has a two-column text layout."""
    blocks = page.get_text("blocks")
    if not blocks:
        return False
    pw = page.rect.width
    mid = pw / 2
    left  = [b for b in blocks if b[6] == 0 and b[2] < mid * 1.15 and len(b[4].strip()) > MIN_BLOCK_CHARS]
    right = [b for b in blocks if b[6] == 0 and b[0] > mid * 0.85 and len(b[4].strip()) > MIN_BLOCK_CHARS]
    total = len([b for b in blocks if b[6] == 0 and len(b[4].strip()) > MIN_BLOCK_CHARS])
    if total == 0:
        return False
    return len(left) / total >= TWO_COL_THRESHOLD and len(right) / total >= TWO_COL_THRESHOLD


def _reorder_two_column(blocks: list, page_width: float) -> list:
    """Sort blocks: left column top→bottom, then right column top→bottom."""
    mid   = page_width / 2
    left  = sorted([b for b in blocks if b[0] < mid], key=lambda b: b[1])
    right = sorted([b for b in blocks if b[0] >= mid], key=lambda b: b[1])
    return left + right


def _strip_header_footer(blocks: list, page_height: float) -> list:
    """Remove blocks in the top/bottom margin zones."""
    margin = page_height * HEADER_FOOTER_RATIO
    return [b for b in blocks if b[1] > margin and b[3] < page_height - margin]


# ── Table extraction (pdfplumber) ──────────────────────────────────────────────

def _extract_tables(pdf_path: str, page_idx: int) -> list[dict]:
    """Extract tables from a single page using pdfplumber."""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_idx >= len(pdf.pages):
                return tables
            for tbl in pdf.pages[page_idx].extract_tables():
                if not tbl:
                    continue
                rows = []
                for row in tbl:
                    cleaned = [str(c).strip() if c else "" for c in row]
                    rows.append(" | ".join(cleaned))
                tables.append({
                    "type": "table",
                    "content": "\n".join(rows),
                    "row_count": len(tbl),
                })
    except Exception:
        pass
    return tables


# ── PDF extractor ──────────────────────────────────────────────────────────────

def extract_pdf(pdf_path: str) -> list[dict]:
    """
    Extract per-page structured content from a PDF.

    Returns list of dicts:
        page_num    : 1-based int
        text        : cleaned body text (reading order corrected)
        tables      : list of table dicts from pdfplumber
        is_two_col  : bool (logged for writeup transparency)
        source      : filename
    """
    path = Path(pdf_path)
    doc  = fitz.open(str(path))
    results: list[dict] = []

    print(f"[ingest] Extracting {len(doc)} pages from {path.name} …")

    for idx in tqdm(range(len(doc)), desc=f"  {path.name}"):
        page   = doc[idx]
        ph     = page.rect.height
        pw     = page.rect.width
        two_col = _is_two_column(page)

        raw_blocks  = page.get_text("blocks")
        text_blocks = [b for b in raw_blocks if b[6] == 0 and len(b[4].strip()) >= MIN_BLOCK_CHARS]
        text_blocks = _strip_header_footer(text_blocks, ph)

        if two_col:
            text_blocks = _reorder_two_column(text_blocks, pw)
        else:
            text_blocks = sorted(text_blocks, key=lambda b: (round(b[1] / 12), b[0]))

        body = "\n\n".join(b[4].strip() for b in text_blocks)
        tables = _extract_tables(str(path), idx)

        results.append({
            "page_num":   idx + 1,
            "text":       body,
            "tables":     tables,
            "is_two_col": two_col,
            "source":     path.name,
        })

    doc.close()
    return results


# ── PDF chunker ────────────────────────────────────────────────────────────────

def chunk_pdf_pages(pages: list[dict], max_words: int = 350, overlap_words: int = 40) -> list[dict]:
    """
    Split PDF pages into overlapping paragraph-level chunks.
    Each chunk carries page_num so citations are accurate.
    """
    chunks: list[dict] = []

    for page in pages:
        pn     = page["page_num"]
        src    = page["source"]
        paras  = [p.strip() for p in re.split(r"\n{2,}", page["text"]) if p.strip()]

        current: list[str] = []
        cur_len = 0
        cidx    = 0

        for para in paras:
            wc = len(para.split())
            if cur_len + wc > max_words and current:
                chunks.append(_pdf_chunk(src, pn, cidx, current, page["is_two_col"]))
                cidx += 1
                # Keep last paragraph for overlap
                current = current[-1:] if overlap_words > 0 else []
                cur_len = len(current[0].split()) if current else 0
            current.append(para)
            cur_len += wc

        if current:
            chunks.append(_pdf_chunk(src, pn, cidx, current, page["is_two_col"]))

        # Table chunks (always separate so retrieval finds them)
        for ti, tbl in enumerate(page.get("tables", [])):
            chunks.append({
                "source":     src,
                "chunk_type": "pdf_table",
                "page_num":   pn,
                "chunk_idx":  f"t{ti}",
                "text":       f"[TABLE – page {pn}]\n{tbl['content']}",
                "metadata":   {"page_num": pn, "row_count": tbl["row_count"]},
            })

    return chunks


def _pdf_chunk(src: str, pn: int, cidx: int, paras: list[str], two_col: bool) -> dict:
    return {
        "source":     src,
        "chunk_type": "pdf_text",
        "page_num":   pn,
        "chunk_idx":  cidx,
        "text":       "\n\n".join(paras),
        "metadata":   {"page_num": pn, "is_two_col": two_col},
    }


# Schedule CSV parser 

def parse_schedule_csv(csv_path: str) -> list[dict]:
    """Parse construction schedule CSV → one chunk per activity row."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    chunks: list[dict] = []
    for i, row in df.iterrows():
        row_dict = {k: v for k, v in row.items() if pd.notna(v) and str(v).strip()}
        act_id   = str(row_dict.get("Activity ID", "")).strip()
        if not act_id:
            act_id = f"row_{i}"
        text = _schedule_to_text(row_dict)
        chunks.append({
            "source":      Path(csv_path).name,
            "chunk_type":  "schedule_activity",
            "activity_id": act_id,
            "text":        text,
            "metadata":    {k: str(v) for k, v in row_dict.items()},
        })

    print(f"[ingest] Schedule CSV → {len(chunks)} activities")
    return chunks


def _schedule_to_text(r: dict) -> str:
    parts = []
    if r.get("Activity ID"):  parts.append(f"Activity ID: {r['Activity ID']}.")
    if r.get("Activity Name"): parts.append(f"Name: {r['Activity Name']}.")
    if r.get("Duration"):      parts.append(f"Duration: {r['Duration']}.")
    if r.get("Start"):         parts.append(f"Start: {r['Start']}.")
    if r.get("Finish"):        parts.append(f"Finish: {r['Finish']}.")
    if r.get("Total Float"):   parts.append(f"Total Float: {r['Total Float']}.")
    if r.get("Constraint"):    parts.append(f"Constraint: {r['Constraint']}.")
    return " ".join(parts)


# ── SOV PDF parser ─────────────────────────────────────────────────────────────

def parse_sov_pdf(pdf_path: str) -> list[dict]:
    """
    Parse Schedule of Values PDF using pdfplumber table extraction.
    Falls back to text extraction if no tables found.
    """
    chunks: list[dict] = []
    path = Path(pdf_path)

    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tbl in tables:
                if not tbl:
                    continue
                # First row = header if it contains text keywords
                header = [str(c).strip() if c else "" for c in (tbl[0] or [])]
                data_rows = tbl[1:] if any(h for h in header) else tbl

                for row in data_rows:
                    if not row:
                        continue
                    cleaned = [str(c).strip() if c else "" for c in row]
                    if not any(cleaned):
                        continue

                    # Build a natural-language text from the row
                    if len(cleaned) >= 2:
                        desc  = cleaned[0] if cleaned[0] else "Unknown"
                        value = cleaned[1] if len(cleaned) > 1 else ""
                        text  = f"SOV line item: {desc}. Value: {value}."
                    else:
                        text = " | ".join(cleaned)

                    row_dict = {h: v for h, v in zip(header, cleaned) if h}
                    chunks.append({
                        "source":     path.name,
                        "chunk_type": "sov_line_item",
                        "line_item":  cleaned[0] if cleaned else "",
                        "text":       text,
                        "metadata":   row_dict,
                    })

    # Fallback: if no table rows found, extract raw text
    if not chunks:
        pages = extract_pdf(str(pdf_path))
        for page in pages:
            if page["text"].strip():
                chunks.append({
                    "source":     path.name,
                    "chunk_type": "sov_text",
                    "line_item":  "",
                    "text":       page["text"],
                    "metadata":   {"page_num": page["page_num"]},
                })

    print(f"[ingest] SOV PDF → {len(chunks)} line items")
    return chunks


# ── Top-level ingest ───────────────────────────────────────────────────────────

def ingest_corpus(
    contract_pdf: str,
    schedule_csv: str,
    sov_pdf: str,
    output_json: str = "./data/corpus_chunks.json",
) -> list[dict]:
    """
    Ingest all three corpus files → unified list of chunk dicts.
    Writes JSON cache to output_json for fast re-use.
    """
    all_chunks: list[dict] = []

    # 1. Contract PDF
    pages      = extract_pdf(contract_pdf)
    pdf_chunks = chunk_pdf_pages(pages)
    print(f"[ingest] Contract PDF → {len(pdf_chunks)} chunks")
    all_chunks.extend(pdf_chunks)

    # 2. Schedule CSV
    sched_chunks = parse_schedule_csv(schedule_csv)
    all_chunks.extend(sched_chunks)

    # 3. SOV PDF
    sov_chunks = parse_sov_pdf(sov_pdf)
    all_chunks.extend(sov_chunks)

    # Assign stable IDs
    for i, c in enumerate(all_chunks):
        c["id"] = f"chunk_{i:05d}"

    # Cache
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(all_chunks, f, indent=2, default=str)
    print(f"[ingest] Total: {len(all_chunks)} chunks → {output_json}")
    return all_chunks


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    chunks = ingest_corpus(
        os.getenv("CONTRACT_PDF",  "./data/contract.pdf"),
        os.getenv("SCHEDULE_CSV",  "./data/schedule-construction.csv"),
        os.getenv("SOV_PDF",       "./data/schedule-of-values.pdf"),
    )
    print(f"\nDone. {len(chunks)} chunks ready.")

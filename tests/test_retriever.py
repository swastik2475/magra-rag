"""
tests/test_retriever.py  –  Basic sanity checks for retriever helpers.
Run with: python -m pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever import _safe_meta, _format_citation


def test_safe_meta_defaults():
    """_safe_meta should return str/int/float values for all keys."""
    result = _safe_meta({})
    assert result["source"] == ""
    assert result["page_num"] == 0
    assert isinstance(result["chunk_idx"], str)


def test_safe_meta_full():
    chunk = {
        "source": "contract.pdf",
        "chunk_type": "pdf_text",
        "page_num": 5,
        "chunk_idx": 2,
        "activity_id": "PROC-11",
        "line_item": "66 inch Force Main",
    }
    result = _safe_meta(chunk)
    assert result["source"] == "contract.pdf"
    assert result["page_num"] == 5
    assert result["activity_id"] == "PROC-11"


def test_format_citation_pdf():
    meta = {"chunk_type": "pdf_text", "page_num": 12, "source": "contract.pdf"}
    assert "p.12" in _format_citation(meta)
    assert "contract.pdf" in _format_citation(meta)


def test_format_citation_schedule():
    meta = {"chunk_type": "schedule_activity", "activity_id": "PROC-11", "source": "schedule.csv"}
    citation = _format_citation(meta)
    assert "PROC-11" in citation
    assert "Schedule" in citation


def test_format_citation_sov():
    meta = {"chunk_type": "sov_line_item", "line_item": "66 inch SS Force Main", "source": "sov.pdf"}
    citation = _format_citation(meta)
    assert "SOV" in citation

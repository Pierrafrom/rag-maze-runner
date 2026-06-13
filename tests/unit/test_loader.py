"""Tests unitaires pour src/loader.py.

Couvre : _page_name_from_url, filter_documents, split_documents.
Aucun appel réseau requis.
"""

import pytest
from langchain_core.documents import Document

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.loader import _page_name_from_url, filter_documents, split_documents

# ---------------------------------------------------------------------------
# _page_name_from_url
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("url,expected", [
    ("https://mazerunner.fandom.com/fr/wiki/Thomas", "Thomas"),
    ("https://mazerunner.fandom.com/fr/wiki/Newt", "Newt"),
    ("https://mazerunner.fandom.com/fr/wiki/Terre_Br%C3%BBl%C3%A9e", "Terre_Brûlée"),
    ("https://mazerunner.fandom.com/fr/wiki/L%27%C3%89preuve_(s%C3%A9rie)", "L'Épreuve_(série)"),
    ("https://mazerunner.fandom.com/fr/wiki/Immunis%C3%A9", "Immunisé"),
])
def test_page_name_from_url(url: str, expected: str) -> None:
    """Les noms de pages doivent être décodés correctement depuis les URLs."""
    assert _page_name_from_url(url) == expected


@pytest.mark.unit
def test_page_name_from_url_simple() -> None:
    assert _page_name_from_url("https://mazerunner.fandom.com/fr/wiki/Chuck") == "Chuck"


# ---------------------------------------------------------------------------
# filter_documents
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_filter_documents_removes_short(sample_documents: list[Document]) -> None:
    """Les documents trop courts (< min_chars) doivent être écartés."""
    result = filter_documents(sample_documents, min_chars=600)
    titles = [d.metadata["title"] for d in result]
    assert "Braise" not in titles


@pytest.mark.unit
def test_filter_documents_keeps_long(sample_documents: list[Document]) -> None:
    result = filter_documents(sample_documents, min_chars=600)
    assert any(d.metadata["title"] == "Thomas" for d in result)
    assert any(d.metadata["title"] == "Newt" for d in result)


@pytest.mark.unit
def test_filter_documents_empty_list() -> None:
    assert filter_documents([], min_chars=500) == []


@pytest.mark.unit
@pytest.mark.parametrize("min_chars,expected_count", [
    (0, 3),    # tout passe
    (600, 2),  # Braise (5 chars) filtré
    (1000, 1), # seul Newt (2000 chars) passe
    (9999, 0), # tout filtré
])
def test_filter_documents_parametrized(
    sample_documents: list[Document],
    min_chars: int,
    expected_count: int,
) -> None:
    assert len(filter_documents(sample_documents, min_chars=min_chars)) == expected_count


# ---------------------------------------------------------------------------
# split_documents
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_split_documents_produces_chunks(sample_document: Document) -> None:
    """Un document doit être découpé en au moins un chunk."""
    chunks = split_documents([sample_document])
    assert len(chunks) >= 1


@pytest.mark.unit
def test_split_documents_chunk_size_respected(sample_document: Document) -> None:
    """Aucun chunk ne doit dépasser chunk_size (+ overlap margin)."""
    chunks = split_documents([sample_document])
    for chunk in chunks:
        assert len(chunk.page_content) <= CHUNK_SIZE + CHUNK_OVERLAP


@pytest.mark.unit
def test_split_documents_preserves_metadata(sample_document: Document) -> None:
    """Les métadonnées du document source doivent être propagées à chaque chunk."""
    chunks = split_documents([sample_document])
    for chunk in chunks:
        assert chunk.metadata["source"] == sample_document.metadata["source"]
        assert chunk.metadata["title"] == sample_document.metadata["title"]


@pytest.mark.unit
def test_split_documents_empty_list() -> None:
    assert split_documents([]) == []


@pytest.mark.unit
def test_split_documents_large_doc() -> None:
    """Un très long document doit être découpé en plusieurs chunks."""
    big_doc = Document(
        page_content="Thomas est un Coureur. " * 200,
        metadata={"source": "url", "title": "Thomas"},
    )
    chunks = split_documents([big_doc])
    assert len(chunks) > 1

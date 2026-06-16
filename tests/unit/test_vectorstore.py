"""Tests unitaires pour src/vectorstore.py.

Couvre : _split_parent_child, advanced_index_exists.
Aucun appel réseau requis — les fonctions testées sont purement locales.
"""

import os
import tempfile

import pytest
from langchain_core.documents import Document

from src.vectorstore import _split_parent_child, advanced_index_exists

# ---------------------------------------------------------------------------
# _split_parent_child
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_split_parent_child_produces_parents_and_children(
    sample_document: Document,
) -> None:
    """Un document doit être découpé en au moins un parent et un enfant."""
    parents, children = _split_parent_child([sample_document])
    assert len(parents) >= 1
    assert len(children) >= 1


@pytest.mark.unit
def test_split_parent_child_children_have_doc_id(
    sample_document: Document,
) -> None:
    """Chaque enfant doit porter le doc_id de son parent dans ses métadonnées."""
    parents, children = _split_parent_child([sample_document])
    parent_ids = {pid for pid, _ in parents}
    for child in children:
        assert "doc_id" in child.metadata
        assert child.metadata["doc_id"] in parent_ids


@pytest.mark.unit
def test_split_parent_child_parent_ids_are_unique(
    sample_documents: list[Document],
) -> None:
    """Chaque parent doit avoir un ID unique (UUID)."""
    long_docs = [d for d in sample_documents if len(d.page_content) >= 600]
    parents, _ = _split_parent_child(long_docs)
    ids = [pid for pid, _ in parents]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
def test_split_parent_child_preserves_source_metadata(
    sample_document: Document,
) -> None:
    """La source du document d'origine doit être propagée aux parents."""
    parents, _ = _split_parent_child([sample_document])
    for _, parent in parents:
        assert parent.metadata.get("source") == sample_document.metadata["source"]


@pytest.mark.unit
def test_split_parent_child_children_smaller_than_parents() -> None:
    """Les enfants doivent être plus petits que les parents (document assez long)."""
    # Document suffisamment long pour que parent (~1500) et enfant (~250) diffèrent.
    long_doc = Document(
        page_content="Thomas est un Coureur du Labyrinthe. " * 80,  # ~2880 car.
        metadata={"source": "url", "title": "Thomas"},
    )
    parents, children = _split_parent_child([long_doc])
    avg_parent = sum(len(p.page_content) for _, p in parents) / len(parents)
    avg_child = sum(len(c.page_content) for c in children) / len(children)
    assert avg_child < avg_parent


@pytest.mark.unit
def test_split_parent_child_empty_list() -> None:
    """Une liste vide de documents doit renvoyer deux listes vides."""
    parents, children = _split_parent_child([])
    assert parents == []
    assert children == []


@pytest.mark.unit
def test_split_parent_child_large_document() -> None:
    """Un document long doit produire plusieurs parents et enfants."""
    big_doc = Document(
        page_content="Thomas parcourt le Labyrinthe chaque jour. " * 100,
        metadata={"source": "url", "title": "Thomas"},
    )
    parents, children = _split_parent_child([big_doc])
    assert len(parents) > 1
    assert len(children) > len(parents)


# ---------------------------------------------------------------------------
# advanced_index_exists
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_advanced_index_exists_false_when_dirs_missing() -> None:
    """Doit retourner False si les répertoires n'existent pas."""
    assert (
        advanced_index_exists("/tmp/__nonexistent_child__", "/tmp/__nonexistent_parent__") is False
    )


@pytest.mark.unit
def test_advanced_index_exists_false_when_dirs_empty() -> None:
    """Doit retourner False si les répertoires existent mais sont vides."""
    with tempfile.TemporaryDirectory() as child_dir, tempfile.TemporaryDirectory() as parent_dir:
        assert advanced_index_exists(child_dir, parent_dir) is False


@pytest.mark.unit
def test_advanced_index_exists_true_when_both_populated() -> None:
    """Doit retourner True si les deux répertoires contiennent des fichiers."""
    with tempfile.TemporaryDirectory() as child_dir, tempfile.TemporaryDirectory() as parent_dir:
        # Créer un fichier factice dans chaque répertoire
        open(os.path.join(child_dir, "dummy.bin"), "w").close()
        open(os.path.join(parent_dir, "dummy.bin"), "w").close()
        assert advanced_index_exists(child_dir, parent_dir) is True


@pytest.mark.unit
def test_advanced_index_exists_false_when_only_child_populated() -> None:
    """Doit retourner False si seul le répertoire enfant est peuplé."""
    with tempfile.TemporaryDirectory() as child_dir, tempfile.TemporaryDirectory() as parent_dir:
        open(os.path.join(child_dir, "dummy.bin"), "w").close()
        assert advanced_index_exists(child_dir, parent_dir) is False

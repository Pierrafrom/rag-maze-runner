"""Fixtures partagées pour tous les tests du RAG Maze Runner.

Les fixtures de ce fichier mocquent les dépendances externes (LLM, embeddings,
index Chroma) pour que les tests unitaires n'aient besoin ni de clé API ni
d'un index construit sur disque.
"""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Documents factices
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_document() -> Document:
    """Un document wiki factice pour Thomas."""
    return Document(
        page_content=(
            "Thomas est le protagoniste principal de l'univers du Labyrinthe. "
            "Il est envoyé dans le Bloc avec les autres Blocards et devient "
            "rapidement Coureur. Il découvre avec Minho une sortie du Labyrinthe."
        ),
        metadata={"source": "https://mazerunner.fandom.com/fr/wiki/Thomas", "title": "Thomas"},
    )


@pytest.fixture()
def sample_documents() -> list[Document]:
    """Plusieurs documents factices représentant différentes pages wiki."""
    return [
        Document(
            page_content="x" * 700,
            metadata={"source": "https://mazerunner.fandom.com/fr/wiki/Thomas", "title": "Thomas"},
        ),
        Document(
            page_content="y" * 2000,
            metadata={"source": "https://mazerunner.fandom.com/fr/wiki/Newt", "title": "Newt"},
        ),
        Document(
            page_content="court",  # trop court, doit être filtré
            metadata={"source": "https://mazerunner.fandom.com/fr/wiki/Braise", "title": "Braise"},
        ),
    ]


@pytest.fixture()
def ranked_list_a() -> list[tuple[str, Document]]:
    """Classement simulé pour une requête (doc_a en tête)."""
    return [
        ("id_a", Document(page_content="A", metadata={"title": "Thomas", "source": "url_a"})),
        ("id_b", Document(page_content="B", metadata={"title": "Newt", "source": "url_b"})),
    ]


@pytest.fixture()
def ranked_list_b() -> list[tuple[str, Document]]:
    """Classement simulé pour une autre requête (doc_a toujours présent)."""
    return [
        ("id_a", Document(page_content="A", metadata={"title": "Thomas", "source": "url_a"})),
        ("id_c", Document(page_content="C", metadata={"title": "Minho", "source": "url_c"})),
    ]


# ---------------------------------------------------------------------------
# Mock LLM (évite les appels API Gemini/Groq)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm() -> MagicMock:
    """LLM factice qui renvoie une réponse prédéfinie."""
    llm = MagicMock()
    llm.invoke.return_value = "Réponse factice du LLM."
    return llm


@pytest.fixture()
def mock_multiquery_chain() -> MagicMock:
    """Chaîne multi-query factice : renvoie 3 reformulations séparées par newline."""
    chain = MagicMock()
    chain.invoke.return_value = (
        "Quel est le rôle de Thomas dans le Bloc ?\n"
        "Qui est Thomas dans l'univers du Labyrinthe ?\n"
        "Quelle est la fonction de Thomas parmi les Blocards ?"
    )
    return chain


@pytest.fixture()
def mock_crag_chain_pertinent() -> MagicMock:
    """Grader CRAG qui renvoie toujours PERTINENT."""
    chain = MagicMock()
    chain.invoke.return_value = "PERTINENT"
    return chain


@pytest.fixture()
def mock_crag_chain_irrelevant() -> MagicMock:
    """Grader CRAG qui renvoie toujours HORS-SUJET."""
    chain = MagicMock()
    chain.invoke.return_value = "HORS-SUJET"
    return chain


@pytest.fixture()
def mock_selfrag_chain_ok() -> MagicMock:
    """Auto-évaluateur Self-RAG qui valide toujours."""
    chain = MagicMock()
    chain.invoke.return_value = "OK"
    return chain


@pytest.fixture()
def mock_selfrag_chain_ko() -> MagicMock:
    """Auto-évaluateur Self-RAG qui rejette toujours."""
    chain = MagicMock()
    chain.invoke.return_value = "A_CORRIGER : information inventée détectée"
    return chain


@pytest.fixture()
def mock_generation_chain() -> MagicMock:
    """Chaîne de génération factice."""
    chain = MagicMock()
    chain.invoke.return_value = "Thomas est le protagoniste du Labyrinthe."
    return chain


@pytest.fixture()
def mock_correction_chain() -> MagicMock:
    """Chaîne de correction factice."""
    chain = MagicMock()
    chain.invoke.return_value = "Réponse corrigée : Thomas est un Coureur."
    return chain


# ---------------------------------------------------------------------------
# Mock embeddings (évite les appels API Gemini pour les embeddings)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_embeddings() -> MagicMock:
    """Embeddings factices qui renvoient des vecteurs de dimension 768."""
    emb = MagicMock()
    emb.embed_documents.return_value = [[0.1] * 768] * 10
    emb.embed_query.return_value = [0.1] * 768
    return emb


# ---------------------------------------------------------------------------
# Patch global get_embeddings (utilisable dans tout test unitaire)
# ---------------------------------------------------------------------------


@pytest.fixture()
def patch_embeddings(mock_embeddings: MagicMock) -> Generator[MagicMock, None, None]:
    """Patch src.vectorstore.get_embeddings pour éviter les appels API."""
    with patch("src.vectorstore.get_embeddings", return_value=mock_embeddings) as m:
        yield m


@pytest.fixture()
def patch_get_llm(mock_llm: MagicMock) -> Generator[MagicMock, None, None]:
    """Patch src.generator.get_llm pour éviter les appels API."""
    with patch("src.generator.get_llm", return_value=mock_llm) as m:
        yield m

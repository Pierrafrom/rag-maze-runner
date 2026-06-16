"""Tests unitaires de ``RagPipeline`` (orchestration), entièrement mockés.

Aucun index Chroma ni clé API n'est requis : les stockages, les chaînes LLM et
les fonctions de récupération sont remplacés par des doubles de test. On vérifie
le câblage des étapes (CRAG, Self-RAG, repli) et le contenu de ``RagAnswer``.
"""

from collections.abc import Iterator
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.config import CRAG_AMBIGUOUS, CRAG_IRRELEVANT, CRAG_RELEVANT, FALLBACK_ANSWER
from src.prompts import CRAG_INSUFFICIENT_NOTICE
from src.rag import RagPipeline


def _docs() -> list[Document]:
    return [
        Document(
            page_content="Thomas est un Coureur du Bloc.",
            metadata={"title": "Thomas", "source": "url_t", "relevance_score": 0.9},
        ),
        Document(
            page_content="Le Labyrinthe entoure le Bloc.",
            metadata={"title": "Labyrinthe", "source": "url_l"},
        ),
    ]


@pytest.fixture()
def pipeline() -> Iterator[RagPipeline]:
    """Construit un ``RagPipeline`` aux dépendances externes neutralisées.

    Les chaînes LLM sont des ``MagicMock`` reconfigurables par chaque test ;
    la récupération renvoie deux documents factices.
    """
    docs = _docs()
    with ExitStack() as stack:
        stack.enter_context(patch("src.rag.advanced_index_exists", return_value=True))
        stack.enter_context(patch("src.rag.get_child_vectorstore", return_value=MagicMock()))
        stack.enter_context(patch("src.rag.get_parent_docstore", return_value=MagicMock()))
        for builder in (
            "build_multiquery_chain",
            "build_generation_chain",
            "build_crag_grader_chain",
            "build_selfrag_chain",
            "build_correction_chain",
        ):
            stack.enter_context(patch(f"src.rag.{builder}", return_value=MagicMock()))
        stack.enter_context(
            patch("src.rag.generate_query_variants", return_value=["question", "variante"])
        )
        stack.enter_context(patch("src.rag.fusion_retrieve", return_value=[(d, 1.0) for d in docs]))
        stack.enter_context(patch("src.rag.rerank_documents", return_value=docs))

        pipe = RagPipeline(verbose=False)
        # Valeurs par défaut « heureuses » ; chaque test peut les surcharger.
        pipe.generation_chain.invoke.return_value = "Thomas est un Coureur."  # type: ignore[attr-defined]
        pipe.crag_chain.invoke.return_value = CRAG_RELEVANT  # type: ignore[attr-defined]
        pipe.selfrag_chain.invoke.return_value = "OK"  # type: ignore[attr-defined]
        yield pipe


@pytest.mark.unit
def test_answer_pertinent_grounded(pipeline: RagPipeline) -> None:
    result = pipeline.answer("Qui est Thomas ?")
    assert result["grounded"] is True
    assert result["crag_status"] == CRAG_RELEVANT
    assert result["answer"] == "Thomas est un Coureur."
    assert result["self_rag"] == "OK"


@pytest.mark.unit
def test_answer_exposes_contexts(pipeline: RagPipeline) -> None:
    # contexts = page_content des documents rerankés (prérequis RAGAS).
    result = pipeline.answer("Qui est Thomas ?")
    assert result["contexts"] == [d.page_content for d in _docs()]


@pytest.mark.unit
def test_answer_sources_carry_scores(pipeline: RagPipeline) -> None:
    result = pipeline.answer("Qui est Thomas ?")
    assert result["sources"][0]["title"] == "Thomas"
    assert result["sources"][0]["score"] == 0.9


@pytest.mark.unit
def test_crag_hors_sujet_triggers_fallback_without_generation(pipeline: RagPipeline) -> None:
    pipeline.crag_chain.invoke.return_value = CRAG_IRRELEVANT  # type: ignore[attr-defined]
    result = pipeline.answer("Quel est le PIB de la France ?")
    assert result["answer"] == FALLBACK_ANSWER
    assert result["grounded"] is False
    assert result["crag_status"] == CRAG_IRRELEVANT
    assert result["sources"] == []
    assert result["contexts"] == []
    # La génération ne doit jamais être appelée sur un repli HORS-SUJET.
    pipeline.generation_chain.invoke.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.unit
def test_crag_ambigu_injects_notice_in_context(pipeline: RagPipeline) -> None:
    pipeline.crag_chain.invoke.return_value = CRAG_AMBIGUOUS  # type: ignore[attr-defined]
    result = pipeline.answer("Qui est Thomas ?")
    assert result["crag_status"] == CRAG_AMBIGUOUS
    assert result["grounded"] is True
    sent_context = pipeline.generation_chain.invoke.call_args.args[0]["context"]  # type: ignore[attr-defined]
    assert sent_context.startswith(CRAG_INSUFFICIENT_NOTICE)


@pytest.mark.unit
def test_self_rag_correction_path(pipeline: RagPipeline) -> None:
    pipeline.selfrag_chain.invoke.return_value = "A_CORRIGER : fait non ancré"  # type: ignore[attr-defined]
    pipeline.correction_chain.invoke.return_value = "Réponse corrigée."  # type: ignore[attr-defined]
    result = pipeline.answer("Qui est Thomas ?")
    assert result["self_rag"] == "A_CORRIGER"
    assert result["answer"] == "Réponse corrigée."
    pipeline.correction_chain.invoke.assert_called_once()  # type: ignore[attr-defined]


@pytest.mark.unit
def test_empty_retrieval_falls_back(pipeline: RagPipeline) -> None:
    with patch("src.rag.fusion_retrieve", return_value=[]):
        result = pipeline.answer("Question sans documents")
    assert result["grounded"] is False
    assert result["crag_status"] == CRAG_IRRELEVANT


@pytest.mark.unit
def test_crag_disabled_skips_grader(pipeline: RagPipeline) -> None:
    pipeline.use_crag = False
    result = pipeline.answer("Qui est Thomas ?")
    assert result["crag_status"] == CRAG_RELEVANT
    pipeline.crag_chain.invoke.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.unit
def test_grade_crag_parses_unreadable_verdict_as_ambiguous(pipeline: RagPipeline) -> None:
    pipeline.crag_chain.invoke.return_value = "verdict illisible"  # type: ignore[attr-defined]
    assert pipeline._grade_crag("q", "ctx") == CRAG_AMBIGUOUS

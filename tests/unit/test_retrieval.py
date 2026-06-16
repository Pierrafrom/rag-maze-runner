"""Tests unitaires pour src/retrieval.py.

Couvre : reciprocal_rank_fusion, generate_query_variants.
Aucun appel réseau ni index Chroma requis.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.retrieval import generate_query_variants, reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rrf_prefers_doc_present_in_both_lists(
    ranked_list_a: list[tuple[str, Document]],
    ranked_list_b: list[tuple[str, Document]],
) -> None:
    """Un document présent dans les deux listes doit avoir le score RRF le plus élevé."""
    result = reciprocal_rank_fusion([ranked_list_a, ranked_list_b], k=60, top_n=3)
    assert result[0][0].metadata["title"] == "Thomas"


@pytest.mark.unit
def test_rrf_scores_are_positive(
    ranked_list_a: list[tuple[str, Document]],
) -> None:
    """Tous les scores RRF doivent être strictement positifs."""
    result = reciprocal_rank_fusion([ranked_list_a], k=60, top_n=2)
    assert all(score > 0 for _, score in result)


@pytest.mark.unit
def test_rrf_scores_are_sorted_descending(
    ranked_list_a: list[tuple[str, Document]],
    ranked_list_b: list[tuple[str, Document]],
) -> None:
    """Les résultats doivent être triés par score décroissant."""
    result = reciprocal_rank_fusion([ranked_list_a, ranked_list_b], k=60, top_n=3)
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_rrf_respects_top_n(
    ranked_list_a: list[tuple[str, Document]],
    ranked_list_b: list[tuple[str, Document]],
) -> None:
    """top_n doit limiter le nombre de résultats retournés."""
    result = reciprocal_rank_fusion([ranked_list_a, ranked_list_b], k=60, top_n=1)
    assert len(result) == 1


@pytest.mark.unit
def test_rrf_empty_lists() -> None:
    """RRF sur des listes vides doit retourner une liste vide."""
    result = reciprocal_rank_fusion([], k=60, top_n=5)
    assert result == []


@pytest.mark.unit
def test_rrf_single_list_single_doc() -> None:
    """Cas minimal : une liste avec un seul document."""
    doc = Document(page_content="seul", metadata={"title": "Seul", "source": "url"})
    result = reciprocal_rank_fusion([[("id_x", doc)]], k=60, top_n=5)
    assert len(result) == 1
    assert result[0][0].metadata["title"] == "Seul"


@pytest.mark.unit
def test_rrf_deduplicates_same_id(ranked_list_a: list[tuple[str, Document]]) -> None:
    """Un même document apparaissant deux fois dans la même liste ne doit compter qu'une fois."""
    result = reciprocal_rank_fusion([ranked_list_a, ranked_list_a], k=60, top_n=5)
    ids = [doc.metadata["title"] for doc, _ in result]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
@pytest.mark.parametrize("k", [1, 10, 60, 100])
def test_rrf_k_affects_score_magnitude(k: int) -> None:
    """Un k plus élevé produit des scores plus faibles (amortissement plus fort)."""
    doc = Document(page_content="x", metadata={"title": "X", "source": "u"})
    result_low_k = reciprocal_rank_fusion([[("id", doc)]], k=1, top_n=1)
    result_high_k = reciprocal_rank_fusion([[("id", doc)]], k=k, top_n=1)
    if k > 1:
        assert result_low_k[0][1] > result_high_k[0][1]


# ---------------------------------------------------------------------------
# generate_query_variants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_query_variants_includes_original(
    mock_multiquery_chain: MagicMock,
) -> None:
    """La question d'origine doit toujours être en tête de la liste."""
    question = "Qui est Thomas ?"
    variants = generate_query_variants(question, mock_multiquery_chain, num_queries=3)
    assert variants[0] == question


@pytest.mark.unit
def test_generate_query_variants_returns_list_of_strings(
    mock_multiquery_chain: MagicMock,
) -> None:
    variants = generate_query_variants("test ?", mock_multiquery_chain, num_queries=3)
    assert isinstance(variants, list)
    assert all(isinstance(v, str) for v in variants)


@pytest.mark.unit
def test_generate_query_variants_deduplicates(mock_multiquery_chain: MagicMock) -> None:
    """Les variantes identiques à la question d'origine ne doivent pas être dupliquées."""
    question = "Qui est Thomas ?"
    mock_multiquery_chain.invoke.return_value = f"{question}\n{question}\nautreQuestion"
    variants = generate_query_variants(question, mock_multiquery_chain, num_queries=3)
    assert variants.count(question) == 1


@pytest.mark.unit
def test_generate_query_variants_strips_bullets(mock_multiquery_chain: MagicMock) -> None:
    """Les puces, tirets et espaces doivent être retirés des variantes."""
    mock_multiquery_chain.invoke.return_value = "- Variante 1\n• Variante 2\n  Variante 3"
    variants = generate_query_variants("Q ?", mock_multiquery_chain, num_queries=3)
    for v in variants[1:]:  # on ignore la question d'origine
        assert not v.startswith(("-", "•", " "))


@pytest.mark.unit
def test_generate_query_variants_empty_lines_ignored(mock_multiquery_chain: MagicMock) -> None:
    """Les lignes vides dans la réponse du LLM ne doivent pas produire de variantes vides."""
    mock_multiquery_chain.invoke.return_value = "Variante 1\n\n\nVariante 2\n"
    variants = generate_query_variants("Q ?", mock_multiquery_chain, num_queries=3)
    assert all(v.strip() for v in variants)

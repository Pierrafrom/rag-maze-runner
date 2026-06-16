"""Tests d'intégrité du jeu d'évaluation (aucun réseau, aucun index)."""

import pytest

from src.config import CRAG_AMBIGUOUS, CRAG_IRRELEVANT, CRAG_RELEVANT
from tests.evaluation.dataset import load_dataset

VALID_STATUSES = {CRAG_RELEVANT, CRAG_AMBIGUOUS, CRAG_IRRELEVANT}


@pytest.mark.unit
def test_dataset_has_at_least_20_questions() -> None:
    assert len(load_dataset()) >= 20


@pytest.mark.unit
def test_dataset_ids_are_unique() -> None:
    ids = [q["id"] for q in load_dataset()]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
def test_dataset_has_both_in_and_out_of_domain() -> None:
    questions = load_dataset()
    assert any(q["should_answer"] for q in questions)
    assert any(not q["should_answer"] for q in questions)


@pytest.mark.unit
def test_crag_statuses_are_valid() -> None:
    for q in load_dataset():
        assert q["expected_crag_status"] in VALID_STATUSES


@pytest.mark.unit
def test_in_domain_questions_have_reference_and_keywords() -> None:
    for q in load_dataset():
        if q["should_answer"]:
            assert q["reference"], f"{q['id']} : référence manquante"
            assert q["expected_keywords"], f"{q['id']} : mots-clés manquants"


@pytest.mark.unit
def test_out_of_domain_questions_expect_fallback() -> None:
    for q in load_dataset():
        if not q["should_answer"]:
            assert q["expected_crag_status"] == CRAG_IRRELEVANT
            assert q["reference"] is None

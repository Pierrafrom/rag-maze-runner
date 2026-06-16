"""Chargement et schéma du jeu d'évaluation de référence.

Source unique pour les deux runners (``run_eval``, ``run_ragas``) et les tests
d'intégrité, afin de ne pas dupliquer la logique de lecture du JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

# Fichier de référence versionné, à côté de ce module.
DATASET_PATH = Path(__file__).with_name("eval_dataset.json")


class EvalQuestion(TypedDict):
    """Une entrée du jeu d'évaluation.

    Attributes:
        id: Identifiant stable (ex. ``"q01"``).
        question: La question posée au pipeline.
        reference: Réponse de référence (``None`` pour les hors-sujet).
        expected_keywords: Mots-clés attendus dans la réponse (in-domain).
        expected_crag_status: Statut CRAG attendu (PERTINENT / AMBIGU / HORS-SUJET).
        should_answer: ``True`` si le pipeline doit produire une réponse ancrée.
    """

    id: str
    question: str
    reference: str | None
    expected_keywords: list[str]
    expected_crag_status: str
    should_answer: bool


def load_dataset(path: Path = DATASET_PATH) -> list[EvalQuestion]:
    """Charge et valide le jeu d'évaluation depuis le JSON.

    Args:
        path: Chemin du fichier JSON (défaut : ``eval_dataset.json`` local).

    Returns:
        La liste des questions d'évaluation.

    Raises:
        FileNotFoundError: Si le fichier est introuvable.
        ValueError: Si une entrée ne respecte pas le schéma attendu.
    """
    if not path.exists():
        raise FileNotFoundError(f"Jeu d'évaluation introuvable : {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = raw["questions"]
    required = {
        "id",
        "question",
        "reference",
        "expected_keywords",
        "expected_crag_status",
        "should_answer",
    }
    for entry in questions:
        missing = required - set(entry)
        if missing:
            raise ValueError(f"Entrée {entry.get('id', '?')} : champs manquants {missing}")
    return [
        EvalQuestion(
            id=entry["id"],
            question=entry["question"],
            reference=entry["reference"],
            expected_keywords=entry["expected_keywords"],
            expected_crag_status=entry["expected_crag_status"],
            should_answer=entry["should_answer"],
        )
        for entry in questions
    ]

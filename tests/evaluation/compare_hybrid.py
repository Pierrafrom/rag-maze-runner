"""Comparaison « dense seul » vs « hybride (BM25) » sur le jeu d'évaluation.

Exécute le pipeline DEUX fois sur les mêmes questions, à provider/modèle
identiques — une fois sans BM25 (``use_hybrid=False``), une fois avec — puis
affiche un tableau avant/après des métriques comportementales et l'écart.

Pensé pour la soutenance : produit un chiffre reproductible et versionné
(``results/compare_hybrid_latest.json``). À lancer en local (``--provider
ollama``) pour éviter tout quota sur le LLM.

⚠️ Rappel : en ``--provider ollama`` le LLM est local, mais le *retrieval*
utilise les embeddings de l'index actif (Gemini par défaut → ``GOOGLE_API_KEY``
requise pour les embeddings de requête). Pour un comparatif 100 % offline,
construire d'abord l'index local (``EMBEDDING_PROVIDER=ollama uv run python
ingest.py --force``).

Exemple :
    uv run python -m tests.evaluation.compare_hybrid --provider ollama --model mistral --limit 6
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

from src.logging_config import setup_logging
from src.rag import RagPipeline
from tests.evaluation.dataset import EvalQuestion, load_dataset
from tests.evaluation.run_eval import RESULTS_DIR, Aggregate, aggregate, evaluate_case

logger = logging.getLogger(__name__)

# (libellé, clé dans Aggregate, « plus haut = mieux » ?)
_ROWS: list[tuple[str, str, bool]] = [
    ("Exactitude décision réponse", "answer_decision_accuracy", True),
    ("Taux de repli correct", "correct_fallback_rate", True),
    ("Accord CRAG", "crag_agreement", True),
    ("Couverture mots-clés (moy.)", "mean_keyword_coverage", True),
    ("Latence moyenne (s)", "mean_latency_s", False),
]


def _run(
    *, use_hybrid: bool, questions: list[EvalQuestion], provider: str | None, model: str | None
) -> Aggregate:
    """Instancie un pipeline (hybride ou non) et agrège les résultats."""
    pipeline = RagPipeline(verbose=False, use_hybrid=use_hybrid, provider=provider, model=model)
    return aggregate([evaluate_case(pipeline, q) for q in questions])


def _print_table(dense: Aggregate, hybrid: Aggregate) -> None:
    """Affiche le tableau comparatif dense vs hybride avec l'écart."""
    print(f"\n{'Métrique':<32}{'Dense':>10}{'Hybride':>10}{'Δ':>10}")
    print("-" * 62)
    for label, key, higher_better in _ROWS:
        d, h = float(dense[key]), float(hybrid[key])  # type: ignore[literal-required]
        delta = h - d
        if key == "mean_latency_s":
            cells = f"{d:>9.1f}s{h:>9.1f}s{delta:>+9.1f}s"
        else:
            cells = f"{d:>9.1%}{h:>9.1%}{delta:>+9.1%}"
        arrow = ""
        if abs(delta) > 1e-9:
            improved = (delta > 0) == higher_better
            arrow = "  ✅" if improved else "  ⚠️"
        print(f"{label:<32}{cells}{arrow}")


def main() -> None:
    """Point d'entrée CLI du comparatif dense vs hybride."""
    parser = argparse.ArgumentParser(description="Comparatif dense vs hybride (BM25).")
    parser.add_argument("--limit", type=int, default=None, help="ne traiter que les N premières.")
    parser.add_argument(
        "--provider", default=None, choices=["gemini", "groq", "ollama"], help="fournisseur LLM."
    )
    parser.add_argument("--model", default=None, help="modèle LLM (ex. 'mistral').")
    parser.add_argument("--verbose", action="store_true", help="logs INFO.")
    args = parser.parse_args()

    setup_logging(logging.INFO if args.verbose else logging.WARNING)

    questions = load_dataset()
    if args.limit is not None:
        questions = questions[: args.limit]

    print(f"Comparatif sur {len(questions)} question(s) — provider={args.provider or 'config'}")
    print("→ run DENSE (sans BM25)…")
    dense = _run(use_hybrid=False, questions=questions, provider=args.provider, model=args.model)
    print("→ run HYBRIDE (avec BM25)…")
    hybrid = _run(use_hybrid=True, questions=questions, provider=args.provider, model=args.model)

    _print_table(dense, hybrid)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    payload = {
        "timestamp": stamp,
        "n_questions": len(questions),
        "provider": args.provider,
        "model": args.model,
        "dense": dense,
        "hybrid": hybrid,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    (RESULTS_DIR / f"compare_hybrid_{stamp}.json").write_text(serialized, encoding="utf-8")
    (RESULTS_DIR / "compare_hybrid_latest.json").write_text(serialized, encoding="utf-8")
    print(f"\nRésultats sauvegardés dans {RESULTS_DIR / 'compare_hybrid_latest.json'}")


if __name__ == "__main__":
    main()

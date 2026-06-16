"""Évaluation comportementale du pipeline RAG avancé (sans juge LLM).

Exécute ``RagPipeline`` sur le jeu de référence ``eval_dataset.json`` et mesure,
sans appel à un juge externe (donc bon marché et déterministe côté métriques) :

* **Taux de repli correct** — les questions hors-sujet déclenchent bien le repli,
  les questions du domaine produisent bien une réponse ancrée ;
* **Accord CRAG** — le statut CRAG observé correspond au statut attendu ;
* **Couverture des mots-clés** — proportion des mots-clés attendus présents dans
  la réponse (questions du domaine uniquement) ;
* **Latence** — temps de réponse par question (total + moyenne).

Chaque appel à ``RagPipeline.answer`` consomme néanmoins le LLM de génération
(Gemini/Groq) : prévoir le quota. Utiliser ``--limit`` pour un essai rapide.

Les résultats sont sauvegardés (versionnés) dans
``tests/evaluation/results/eval_<horodatage>.json`` + un fichier
``eval_latest.json`` stable.

Exemples :
    uv run python -m tests.evaluation.run_eval
    uv run python -m tests.evaluation.run_eval --limit 4
    uv run python -m tests.evaluation.run_eval --no-multiquery --no-rerank
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from src.config import CRAG_IRRELEVANT
from src.logging_config import setup_logging
from src.rag import RagAnswer, RagPipeline
from tests.evaluation.dataset import EvalQuestion, load_dataset

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).with_name("results")


class CaseResult(TypedDict):
    """Résultat détaillé pour une question évaluée."""

    id: str
    question: str
    answer: str
    grounded: bool
    crag_status: str
    expected_crag_status: str
    crag_match: bool
    should_answer: bool
    answer_match: bool
    keyword_coverage: float
    missing_keywords: list[str]
    latency_s: float


def _keyword_coverage(answer: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Calcule la fraction de mots-clés présents dans la réponse.

    Args:
        answer: Réponse produite par le pipeline.
        keywords: Mots-clés attendus.

    Returns:
        Un couple ``(couverture, mots_clés_manquants)``. La couverture vaut
        ``1.0`` quand aucun mot-clé n'est attendu (rien à vérifier).
    """
    if not keywords:
        return 1.0, []
    lowered = answer.lower()
    missing = [kw for kw in keywords if kw.lower() not in lowered]
    coverage = (len(keywords) - len(missing)) / len(keywords)
    return coverage, missing


def evaluate_case(pipeline: RagPipeline, question: EvalQuestion) -> CaseResult:
    """Évalue une question et renvoie le résultat détaillé.

    Args:
        pipeline: Le pipeline RAG instancié.
        question: L'entrée du jeu de référence à évaluer.

    Returns:
        Le ``CaseResult`` correspondant.
    """
    start = time.perf_counter()
    result: RagAnswer = pipeline.answer(question["question"])
    latency = time.perf_counter() - start

    grounded = result["grounded"]
    crag_status = result["crag_status"]
    coverage, missing = _keyword_coverage(result["answer"], question["expected_keywords"])

    return CaseResult(
        id=question["id"],
        question=question["question"],
        answer=result["answer"],
        grounded=grounded,
        crag_status=crag_status,
        expected_crag_status=question["expected_crag_status"],
        crag_match=crag_status == question["expected_crag_status"],
        should_answer=question["should_answer"],
        # Repli attendu (should_answer=False) ⇔ réponse non ancrée.
        answer_match=grounded == question["should_answer"],
        keyword_coverage=coverage,
        missing_keywords=missing,
        latency_s=latency,
    )


class Aggregate(TypedDict):
    """Métriques agrégées sur l'ensemble du jeu."""

    n_cases: int
    n_in_domain: int
    n_out_of_domain: int
    answer_decision_accuracy: float
    correct_fallback_rate: float
    crag_agreement: float
    mean_keyword_coverage: float
    mean_latency_s: float


def aggregate(results: list[CaseResult]) -> Aggregate:
    """Agrège les résultats par cas en métriques globales.

    Args:
        results: La liste des résultats détaillés.

    Returns:
        Les métriques agrégées.
    """
    n = len(results)
    in_domain = [r for r in results if r["should_answer"]]
    out_domain = [r for r in results if not r["should_answer"]]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    correct_fallback = [r for r in out_domain if not r["grounded"]]

    return Aggregate(
        n_cases=n,
        n_in_domain=len(in_domain),
        n_out_of_domain=len(out_domain),
        answer_decision_accuracy=_mean([1.0 if r["answer_match"] else 0.0 for r in results]),
        correct_fallback_rate=(len(correct_fallback) / len(out_domain)) if out_domain else 0.0,
        crag_agreement=_mean([1.0 if r["crag_match"] else 0.0 for r in results]),
        mean_keyword_coverage=_mean([r["keyword_coverage"] for r in in_domain]),
        mean_latency_s=_mean([r["latency_s"] for r in results]),
    )


def _print_report(results: list[CaseResult], summary: Aggregate) -> None:
    """Affiche un rapport lisible en console."""
    print("\n=== Détail par question ===")
    for r in results:
        ok_answer = "✅" if r["answer_match"] else "❌"
        ok_crag = "✅" if r["crag_match"] else "❌"
        print(
            f"{r['id']} {ok_answer} décision  {ok_crag} CRAG "
            f"({r['crag_status']}/{r['expected_crag_status']})  "
            f"mots-clés={r['keyword_coverage']:.0%}  {r['latency_s']:.1f}s  "
            f"| {r['question']}"
        )
        if r["missing_keywords"]:
            print(f"      mots-clés manquants : {', '.join(r['missing_keywords'])}")

    print("\n=== Synthèse ===")
    print(
        f"Questions évaluées          : {summary['n_cases']} "
        f"({summary['n_in_domain']} domaine / {summary['n_out_of_domain']} hors-sujet)"
    )
    print(f"Exactitude décision réponse : {summary['answer_decision_accuracy']:.1%}")
    print(f"Taux de repli correct       : {summary['correct_fallback_rate']:.1%}")
    print(f"Accord CRAG                 : {summary['crag_agreement']:.1%}")
    print(f"Couverture mots-clés (moy.) : {summary['mean_keyword_coverage']:.1%}")
    print(f"Latence moyenne             : {summary['mean_latency_s']:.1f}s")


def _save_results(results: list[CaseResult], summary: Aggregate) -> Path:
    """Sauvegarde les résultats horodatés + un alias ``eval_latest.json``.

    Returns:
        Le chemin du fichier horodaté.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    payload = {"timestamp": stamp, "summary": summary, "cases": results}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)

    dated = RESULTS_DIR / f"eval_{stamp}.json"
    dated.write_text(serialized, encoding="utf-8")
    (RESULTS_DIR / "eval_latest.json").write_text(serialized, encoding="utf-8")
    return dated


def main() -> None:
    """Point d'entrée CLI de l'évaluation comportementale."""
    parser = argparse.ArgumentParser(description="Évaluation comportementale du RAG Maze Runner.")
    parser.add_argument("--limit", type=int, default=None, help="ne traiter que les N premières.")
    parser.add_argument("--no-multiquery", action="store_true", help="désactive le Multi-Query.")
    parser.add_argument("--no-rerank", action="store_true", help="désactive le Re-Ranking.")
    parser.add_argument("--no-hybrid", action="store_true", help="désactive le BM25 (dense seul).")
    parser.add_argument("--no-crag", action="store_true", help="désactive le CRAG.")
    parser.add_argument("--no-self-rag", action="store_true", help="désactive le Self-RAG.")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["gemini", "groq", "ollama"],
        help="fournisseur LLM (défaut : LLM_PROVIDER). 'ollama' = local, sans quota.",
    )
    parser.add_argument("--model", default=None, help="modèle LLM (ex. 'mistral' pour Ollama).")
    parser.add_argument("--verbose", action="store_true", help="logs INFO du pipeline.")
    args = parser.parse_args()

    setup_logging(logging.INFO if args.verbose else logging.WARNING)

    questions = load_dataset()
    if args.limit is not None:
        questions = questions[: args.limit]

    pipeline = RagPipeline(
        verbose=args.verbose,
        use_multiquery=not args.no_multiquery,
        use_rerank=not args.no_rerank,
        use_crag=not args.no_crag,
        use_self_rag=not args.no_self_rag,
        use_hybrid=not args.no_hybrid,
        provider=args.provider,
        model=args.model,
    )

    print(f"Évaluation de {len(questions)} question(s)…")
    if args.no_crag:
        print(
            f"⚠️  CRAG désactivé : le repli ne dépend plus que de l'absence "
            f"de documents (statut figé à {CRAG_IRRELEVANT} en repli)."
        )

    results: list[CaseResult] = []
    for q in questions:
        case = evaluate_case(pipeline, q)
        results.append(case)
        print(f"  {case['id']} traité ({case['latency_s']:.1f}s)")

    summary = aggregate(results)
    _print_report(results, summary)
    saved = _save_results(results, summary)
    print(f"\nRésultats sauvegardés dans {saved}")


if __name__ == "__main__":
    main()

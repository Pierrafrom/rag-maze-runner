"""Évaluation RAGAS du pipeline RAG avancé (4 métriques de l'énoncé).

Porte la preuve de concept RAGAS (initialement écrite dans un notebook sur la
branche ``main``, qui évaluait une chaîne *naïve* avec un juge Ollama) vers le
*vrai* ``RagPipeline`` à 6 étapes du projet.

Différences clés avec la PoC d'origine :

* On évalue ``RagPipeline.answer`` (Multi-Query → RAG-Fusion → Re-Ranking → CRAG
  → Génération → Self-RAG), pas une chaîne naïve ;
* Les contextes évalués sont ceux réellement fournis au générateur
  (``RagAnswer["contexts"]``), pas un retriever séparé ;
* Le **juge** et les **embeddings** réutilisent les providers du projet
  (``get_llm`` / ``get_embeddings``, soit Gemini/Groq) — reproductible dans
  notre environnement, sans dépendance Ollama supplémentaire.

Métriques (les 4 demandées par le sujet) :
    * Faithfulness                       — fidélité (réponse ancrée au contexte) ;
    * ResponseRelevancy                  — pertinence de la réponse ;
    * LLMContextPrecisionWithReference   — précision du contexte récupéré ;
    * LLMContextRecall                   — rappel du contexte.

RAGAS est une dépendance *optionnelle* (groupe ``eval``) :
    uv sync --group eval

Attention au quota : chaque question déclenche le pipeline complet *puis*
plusieurs appels du juge. Utiliser ``--limit`` pour un essai.

Exemples :
    uv sync --group eval
    uv run python -m tests.evaluation.run_ragas --limit 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.logging_config import setup_logging
from src.rag import RagPipeline
from src.vectorstore import get_embeddings
from tests.evaluation.dataset import load_dataset

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).with_name("results")


def _install_vertexai_shim() -> None:
    """Neutralise un import cassé de RAGAS (ChatVertexAI) qu'on n'utilise pas.

    Certaines versions de RAGAS importent ``ChatVertexAI`` depuis un module de
    ``langchain_community`` désormais supprimé. Comme on n'utilise pas Vertex AI,
    on enregistre un module factice pour débloquer l'import (jamais instancié).
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    fake = types.ModuleType(module_name)
    fake.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
    sys.modules[module_name] = fake


def _build_samples(pipeline: RagPipeline, limit: int | None) -> list[dict[str, Any]]:
    """Construit les échantillons RAGAS à partir des sorties du pipeline.

    Seules les questions du domaine (avec ``reference``) sont évaluées : les
    métriques RAGAS supposent une réponse et un contexte non vides.

    Args:
        pipeline: Le pipeline RAG instancié.
        limit: Nombre maximum de questions du domaine à évaluer.

    Returns:
        Une liste de dicts au format attendu par ``EvaluationDataset.from_list``.
    """
    answerable = [q for q in load_dataset() if q["should_answer"] and q["reference"]]
    if limit is not None:
        answerable = answerable[:limit]

    samples: list[dict[str, Any]] = []
    for q in answerable:
        result = pipeline.answer(q["question"])
        contexts = result["contexts"] or [s["title"] for s in result["sources"]]
        samples.append(
            {
                "user_input": q["question"],
                "retrieved_contexts": contexts,
                "response": result["answer"],
                "reference": q["reference"],
            }
        )
        logger.info("[RAGAS] échantillon prêt : %s", q["id"])
    return samples


def _save(scores: dict[str, float], n_samples: int) -> Path:
    """Sauvegarde les scores RAGAS (horodaté + alias ``ragas_latest.json``)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    payload = {"timestamp": stamp, "n_samples": n_samples, "scores": scores}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    dated = RESULTS_DIR / f"ragas_{stamp}.json"
    dated.write_text(serialized, encoding="utf-8")
    (RESULTS_DIR / "ragas_latest.json").write_text(serialized, encoding="utf-8")
    return dated


def _judge_embeddings(provider: str) -> Any:
    """Renvoie les embeddings du juge RAGAS.

    En mode local (``provider == "ollama"``), on utilise des embeddings Ollama
    (``OLLAMA_EMBED_MODEL``) afin de rester totalement hors quota ; sinon on
    réutilise les embeddings du projet (Gemini). Ces embeddings sont
    indépendants de ceux de l'index (ils ne servent qu'à la métrique de
    pertinence de la réponse).
    """
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        from src.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

        logger.info("[RAGAS] embeddings juge : Ollama %s (local)", OLLAMA_EMBED_MODEL)
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    logger.info("[RAGAS] embeddings juge : Gemini")
    return get_embeddings()


def main() -> None:
    """Point d'entrée CLI de l'évaluation RAGAS."""
    parser = argparse.ArgumentParser(description="Évaluation RAGAS du RAG Maze Runner.")
    parser.add_argument("--limit", type=int, default=None, help="ne traiter que les N premières.")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["gemini", "groq", "ollama"],
        help="fournisseur du pipeline ET du juge (défaut : LLM_PROVIDER). "
        "'ollama' = juge + embeddings locaux, sans quota.",
    )
    parser.add_argument("--model", default=None, help="modèle LLM (ex. 'llama3.1:8b' pour Ollama).")
    parser.add_argument("--verbose", action="store_true", help="logs INFO du pipeline.")
    args = parser.parse_args()

    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    _install_vertexai_shim()

    try:
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:  # pragma: no cover - dépend de l'install
        raise SystemExit("RAGAS n'est pas installé. Exécutez :  uv sync --group eval") from exc

    # Le juge réutilise le provider du projet (Gemini/Groq/Ollama). En local
    # (--provider ollama), l'évaluation tourne entièrement hors quota.
    from src.config import LLM_PROVIDER
    from src.generator import get_llm

    effective_provider = args.provider or LLM_PROVIDER
    evaluator_llm = LangchainLLMWrapper(
        get_llm(temperature=0.0, provider=args.provider, model=args.model)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(_judge_embeddings(effective_provider))
    print(f"Juge RAGAS : provider={effective_provider}")

    pipeline = RagPipeline(verbose=args.verbose, provider=args.provider, model=args.model)
    samples = _build_samples(pipeline, args.limit)
    print(f"Évaluation RAGAS sur {len(samples)} question(s) du domaine…")

    dataset = EvaluationDataset.from_list(samples)
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    # Providers distants mono-instance → on limite la concurrence pour le quota.
    run_config = RunConfig(timeout=600, max_workers=1)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    scores = {k: float(v) for k, v in dict(result).items()}
    print("\n=== Scores RAGAS ===")
    for name, value in scores.items():
        print(f"  {name:40s} : {value:.4f}")

    saved = _save(scores, len(samples))
    print(f"\nRésultats sauvegardés dans {saved}")


if __name__ == "__main__":
    main()

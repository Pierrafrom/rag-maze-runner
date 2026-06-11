"""Point d'entrée CLI du RAG Maze Runner (pipeline avancé).

Pose une question à l'index parent-enfant (lecture seule) en passant par toutes
les étapes (Multi-Query → RAG-Fusion → Re-Ranking → CRAG → Génération →
Self-RAG) et affiche la réponse, le statut CRAG et les sources du wiki.

Usage :
    uv run python main.py "Quel est le but de WICKED ?"

Prérequis : GOOGLE_API_KEY définie (ou .env) et index construit via ingest.py.
"""

import sys

from src.config import GOOGLE_API_KEY, GROQ_API_KEY, LLM_PROVIDER
from src.logging_config import setup_logging
from src.rag import RagPipeline


def main(question: str) -> dict:
    setup_logging()

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            sys.exit("LLM_PROVIDER=groq mais GROQ_API_KEY n'est pas définie dans .env")
    elif not GOOGLE_API_KEY:
        sys.exit("La variable d'environnement GOOGLE_API_KEY n'est pas définie.")

    pipeline = RagPipeline()
    result = pipeline.answer(question)

    print("\n" + "=" * 72)
    print(f"Question     : {question}")
    print(f"Statut CRAG  : {result['crag_status']}")
    print(f"Self-RAG     : {result['self_rag']}")
    print(f"\nRéponse      : {result['answer']}")

    if result["grounded"] and result["sources"]:
        print("\nSources (wiki Fandom FR) :")
        for src in result["sources"]:
            score = f"score={src['score']}" if src["score"] is not None else ""
            print(f"  - {src['title']} {score} — {src['source']}")
    else:
        print("\n(Aucune source suffisamment pertinente : réponse de repli.)")

    return result


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Quel est le but de WICKED ?"
    main(query)

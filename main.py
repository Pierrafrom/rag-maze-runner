"""Point d'entrée CLI du RAG Maze Runner.

Pose une question à la base déjà construite (connexion en lecture seule) et
affiche la réponse ainsi que les sources du wiki utilisées.

Usage :
    python main.py "Qui est Thomas ?"

Prérequis : GOOGLE_API_KEY définie (ou .env) et base construite via ingest.py.
"""

import sys

from src.config import GOOGLE_API_KEY
from src.rag import RagPipeline


def main(question: str) -> dict:
    if not GOOGLE_API_KEY:
        sys.exit("La variable d'environnement GOOGLE_API_KEY n'est pas définie.")

    pipeline = RagPipeline()
    result = pipeline.answer(question)

    print(f"\nQuestion : {question}")
    print(f"\nRéponse  : {result['answer']}")

    if result["grounded"]:
        print("\nSources (wiki Fandom FR) :")
        for src in result["sources"]:
            print(f"  - {src['title']} (cosinus={src['score']}) — {src['source']}")
    else:
        print("\n(Aucune source suffisamment pertinente : réponse de repli.)")

    return result


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Qui est Thomas ?"
    main(query)

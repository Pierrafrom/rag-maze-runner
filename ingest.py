"""Script d'ingestion : (re)construit la base vectorielle Chroma.

Reproduit l'étape P1 du notebook de façon reproductible :
    chargement wiki → filtrage → chunking → embeddings → Chroma.

Usage :
    python ingest.py

Prérequis : variable d'environnement GOOGLE_API_KEY (ou fichier .env).
À ne lancer que pour construire/reconstruire la base. La couche de
question-réponse (main.py / Streamlit) s'y connecte ensuite en lecture seule.
"""

import sys

from src.config import CHROMA_PERSIST_DIR, GOOGLE_API_KEY
from src.loader import prepare_chunks
from src.vectorstore import build_vectorstore, vectorstore_exists


def main() -> None:
    if not GOOGLE_API_KEY:
        sys.exit("La variable d'environnement GOOGLE_API_KEY n'est pas définie.")

    if vectorstore_exists(CHROMA_PERSIST_DIR):
        print(
            f"Une base existe déjà dans '{CHROMA_PERSIST_DIR}'.\n"
            "Supprimez ce dossier pour forcer une reconstruction complète."
        )
        return

    print("Chargement et préparation des documents du wiki Maze Runner...\n")
    chunks = prepare_chunks()

    print("\nIndexation dans Chroma (par batchs)...\n")
    build_vectorstore(chunks)

    print("\nBase vectorielle construite avec succès.")


if __name__ == "__main__":
    main()

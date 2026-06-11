"""Script d'ingestion : construit l'index parent-enfant (pipeline avancé).

Reproduit l'étape d'indexation de façon reproductible :
    chargement wiki → documents nettoyés → découpage parent/enfant →
    embeddings des enfants (Chroma) + parents (docstore persistant).

Usage :
    uv run python ingest.py

Prérequis : variable d'environnement GOOGLE_API_KEY (ou fichier .env).
À ne lancer que pour construire/reconstruire l'index. La couche de
question-réponse (main.py / Streamlit) s'y connecte ensuite en lecture seule.
"""

import sys

from src.config import EMBEDDING_PROVIDER, GOOGLE_API_KEY
from src.loader import load_clean_documents
from src.logging_config import setup_logging
from src.vectorstore import advanced_index_exists, build_parent_document_index


def main() -> None:
    setup_logging()

    if EMBEDDING_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        sys.exit("La variable d'environnement GOOGLE_API_KEY n'est pas définie.")

    if advanced_index_exists():
        print(
            "L'index parent-enfant existe déjà (chroma_children/ + "
            "parent_docstore/).\nSupprimez ces dossiers pour forcer une "
            "reconstruction complète."
        )
        return

    print("Chargement et nettoyage des documents du wiki Maze Runner...\n")
    documents = load_clean_documents()

    print("\nIndexation parent-enfant dans Chroma (par lots, avec pauses)...\n")
    build_parent_document_index(documents)

    print("\nIndex parent-enfant construit avec succès.")


if __name__ == "__main__":
    main()

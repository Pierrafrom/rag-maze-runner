"""Script d'ingestion : construit l'index parent-enfant (pipeline avancé).

Reproduit l'étape d'indexation de façon reproductible :
    chargement wiki → documents nettoyés → découpage parent/enfant →
    embeddings des enfants (Chroma) + parents (docstore persistant).

Usage :
    uv run python ingest.py           # construit l'index (skip si déjà présent)
    uv run python ingest.py --force   # écrase et reconstruit même si présent

Prérequis : variable d'environnement GOOGLE_API_KEY (ou fichier .env).
À ne lancer que pour construire/reconstruire l'index. La couche de
question-réponse (main.py / Streamlit) s'y connecte ensuite en lecture seule.
"""

import argparse
import shutil
import sys

from src.config import CHILD_CHROMA_DIR, EMBEDDING_PROVIDER, GOOGLE_API_KEY, PARENT_DOCSTORE_DIR
from src.loader import load_clean_documents
from src.logging_config import setup_logging
from src.vectorstore import advanced_index_exists, build_parent_document_index


def _delete_index() -> None:
    """Supprime chroma_children/ et parent_docstore/ pour forcer une reconstruction."""
    for path in (CHILD_CHROMA_DIR, PARENT_DOCSTORE_DIR):
        if shutil.os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  Supprimé : {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit l'index parent-enfant du RAG.")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Écrase l'index existant et reconstruit depuis zéro.",
    )
    args = parser.parse_args()

    setup_logging()

    if EMBEDDING_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        sys.exit("La variable d'environnement GOOGLE_API_KEY n'est pas définie.")

    if advanced_index_exists():
        if not args.force:
            print(
                "L'index parent-enfant existe déjà (chroma_children/ + "
                "parent_docstore/).\nUtilisez --force pour forcer une "
                "reconstruction complète."
            )
            return
        print("--force : suppression de l'index existant...")
        _delete_index()

    print("Chargement et nettoyage des documents du wiki Maze Runner...\n")
    documents = load_clean_documents()

    print("\nIndexation parent-enfant dans Chroma (par lots, avec pauses)...\n")
    build_parent_document_index(documents)

    print("\nIndex parent-enfant construit avec succès.")


if __name__ == "__main__":
    main()

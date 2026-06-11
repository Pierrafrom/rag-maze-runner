"""Accès à la base vectorielle Chroma (embeddings Gemini).

Deux usages distincts :

* ``build_vectorstore`` — (re)construction de la base par batchs. Reproduit
  l'étape d'indexation P1 du notebook (utilisé par ``ingest.py``).
* ``load_vectorstore`` — connexion **en lecture seule** à la base déjà
  alimentée par P1, pour la couche de question-réponse. N'écrit rien.
"""

import os
import time

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBED_BATCH_SIZE,
    EMBED_PAUSE,
    EMBEDDING_MODEL,
)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Instancie le modèle d'embedding Gemini.

    Le modèle doit être identique à celui utilisé pour construire la base,
    faute de quoi les vecteurs requête/documents ne sont pas comparables.
    """
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def vectorstore_exists(persist_directory: str = CHROMA_PERSIST_DIR) -> bool:
    """Indique si une base Chroma persistée existe déjà sur disque."""
    return os.path.isdir(persist_directory) and bool(os.listdir(persist_directory))


def build_vectorstore(
    chunks: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    batch_size: int = EMBED_BATCH_SIZE,
    pause: int = EMBED_PAUSE,
    verbose: bool = True,
) -> Chroma:
    """Construit la base Chroma en indexant les chunks par batchs.

    L'indexation est fractionnée (batchs + pauses) pour respecter le quota de
    l'API d'embeddings Gemini, comme dans le notebook P1.

    Args:
        chunks: documents découpés à indexer.
        persist_directory: dossier de persistance Chroma.
        collection_name: nom de la collection Chroma.
        batch_size: nombre de chunks par batch.
        pause: pause (s) entre deux batchs.
        verbose: journalise la progression.

    Returns:
        L'instance ``Chroma`` alimentée.
    """
    if not chunks:
        raise ValueError("Aucun chunk à indexer : la liste fournie est vide.")

    embeddings = get_embeddings()
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    if verbose:
        print(f"{len(chunks)} chunks → {total_batches} batch(s) de {batch_size}\n")

    # Premier batch : crée la collection.
    vectorstore = Chroma.from_documents(
        documents=chunks[:batch_size],
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    if verbose:
        print(f"Batch 1/{total_batches} OK")

    # Batchs suivants : on patiente puis on ajoute.
    for i in range(batch_size, len(chunks), batch_size):
        batch_num = i // batch_size + 1
        if verbose:
            print(f"Pause {pause}s avant batch {batch_num}/{total_batches}...")
        time.sleep(pause)
        vectorstore.add_documents(chunks[i : i + batch_size])
        if verbose:
            print(f"Batch {batch_num}/{total_batches} OK")

    if verbose:
        print(f"\nTerminé : {vectorstore._collection.count()} vecteurs stockés")
    return vectorstore


def load_vectorstore(
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """Ouvre la base Chroma existante en lecture seule.

    Ne reconstruit pas et ne réalimente pas la base : il s'agit d'un simple
    handle vers la collection déjà persistée par P1.

    Raises:
        FileNotFoundError: si aucune base n'est trouvée (lancer ``ingest.py``).
    """
    if not vectorstore_exists(persist_directory):
        raise FileNotFoundError(
            f"Base Chroma introuvable dans '{persist_directory}'. "
            "Construisez-la d'abord avec : python ingest.py"
        )
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
    )

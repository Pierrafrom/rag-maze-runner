"""Accès aux bases vectorielles et au docstore parent.

Deux familles de fonctions :

* **Pipeline simple (legacy)** — `build_vectorstore` / `load_vectorstore` :
  base Chroma de chunks de 1000 car. (chemin « RAG naïf »).
* **Pipeline avancé (multi-représentation)** — `ParentDocumentRetriever` :
  petits enfants (~250 car.) embarqués dans Chroma `chroma_children/`, gros
  parents (~1500 car.) stockés dans un docstore persistant `parent_docstore/`.
  Au retrieval, un enfant sélectionné fait remonter son parent.
"""

import logging
import os
import time
import uuid
from typing import Literal, cast

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHILD_CHROMA_DIR,
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_COLLECTION_NAME,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBED_BATCH_SIZE,
    EMBED_PAUSE,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GOOGLE_API_KEYS,
    HF_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_DOCSTORE_DIR,
)

logger = logging.getLogger(__name__)


class RotatingGeminiEmbeddings(Embeddings):
    """Wrapper multi-clés Gemini : bascule vers la clé suivante sur quota 429.

    Instancier avec une liste de clés API différentes (comptes Google distincts).
    La rotation est transparente pour le reste du pipeline.
    """

    def __init__(self, api_keys: list[str], model: str) -> None:
        self._clients = [
            GoogleGenerativeAIEmbeddings(model=model, google_api_key=key) for key in api_keys
        ]
        self._current = 0
        logger.info("[Embeddings] %d clé(s) API Gemini chargée(s)", len(api_keys))

    def _rotate(self) -> None:
        next_idx = (self._current + 1) % len(self._clients)
        logger.warning(
            "[Embeddings] Clé %d épuisée (429) — rotation vers la clé %d/%d",
            self._current + 1,
            next_idx + 1,
            len(self._clients),
        )
        self._current = next_idx

    def _call_with_rotation(
        self, method: Literal["embed_documents", "embed_query"], *args: list[str] | str
    ) -> list[list[float]] | list[float]:
        for _ in range(len(self._clients)):
            try:
                return cast(
                    "list[list[float]] | list[float]",
                    getattr(self._clients[self._current], method)(*args),
                )
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                    self._rotate()
                else:
                    raise
        raise RuntimeError(
            f"[Embeddings] Toutes les clés API ({len(self._clients)}) sont "
            "épuisées (quota 429). Attendez la réinitialisation journalière ou "
            "ajoutez d'autres clés (GOOGLE_API_KEY_2, …) dans le .env."
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self._call_with_rotation("embed_documents", texts)
        return cast("list[list[float]]", result)

    def embed_query(self, text: str) -> list[float]:
        result = self._call_with_rotation("embed_query", text)
        return cast("list[float]", result)


def get_embeddings() -> Embeddings:
    """Instancie le modèle d'embedding selon EMBEDDING_PROVIDER.

    * ``"gemini"`` (défaut) : utilise l'API Gemini avec rotation automatique
      des clés si plusieurs GOOGLE_API_KEY_* sont définis dans le .env.
    * ``"huggingface"`` : modèle local sentence-transformers, sans quota.
      Nécessite : ``uv add langchain-huggingface sentence-transformers``.

    ⚠️ Provider et modèle doivent être identiques entre indexation et requête.
    """
    if EMBEDDING_PROVIDER == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain-huggingface n'est pas installé. "
                "Exécutez : uv add langchain-huggingface sentence-transformers"
            ) from exc
        logger.info("[Embeddings] HuggingFace — %s", HF_EMBEDDING_MODEL)
        # Import optionnel non typé (dépendance non installée par défaut) : la
        # classe respecte bien l'interface Embeddings à l'exécution.
        return cast("Embeddings", HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL))

    if EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings  # noqa: PLC0415

        logger.info("[Embeddings] Ollama (local) — %s", OLLAMA_EMBED_MODEL)
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    # Gemini (par défaut)
    if len(GOOGLE_API_KEYS) > 1:
        logger.info(
            "[Embeddings] Gemini avec rotation (%d clés) — %s",
            len(GOOGLE_API_KEYS),
            EMBEDDING_MODEL,
        )
        return RotatingGeminiEmbeddings(GOOGLE_API_KEYS, EMBEDDING_MODEL)

    logger.info("[Embeddings] Gemini (clé unique) — %s", EMBEDDING_MODEL)
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def vectorstore_exists(persist_directory: str = CHROMA_PERSIST_DIR) -> bool:
    """Indique si une base Chroma persistée existe déjà sur disque."""
    return os.path.isdir(persist_directory) and bool(os.listdir(persist_directory))


# ===========================================================================
# Pipeline simple (legacy)
# ===========================================================================
def build_vectorstore(
    chunks: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    batch_size: int = EMBED_BATCH_SIZE,
    pause: int = EMBED_PAUSE,
    verbose: bool = True,
) -> Chroma:
    """Construit la base Chroma simple en indexant les chunks par batchs."""
    if not chunks:
        raise ValueError("Aucun chunk à indexer : la liste fournie est vide.")

    embeddings = get_embeddings()
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    logger.info(
        "[Vectorstore] %d chunks → %d batch(s) de %d",
        len(chunks),
        total_batches,
        batch_size,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks[:batch_size],
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    logger.info("[Vectorstore] Batch 1/%d OK", total_batches)

    for i in range(batch_size, len(chunks), batch_size):
        batch_num = i // batch_size + 1
        logger.info("[Vectorstore] Pause %ds avant batch %d/%d...", pause, batch_num, total_batches)
        time.sleep(pause)
        vectorstore.add_documents(chunks[i : i + batch_size])
        logger.info("[Vectorstore] Batch %d/%d OK", batch_num, total_batches)

    logger.info(
        "[Vectorstore] Terminé : %d vecteurs stockés",
        vectorstore._collection.count(),
    )
    return vectorstore


def load_vectorstore(
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """Ouvre la base Chroma simple existante en lecture seule."""
    if not vectorstore_exists(persist_directory):
        raise FileNotFoundError(
            f"Base Chroma introuvable dans '{persist_directory}'. "
            "Construisez-la d'abord avec : uv run python ingest.py"
        )
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
    )


# ===========================================================================
# Pipeline avancé : indexation multi-représentation (parent / enfant)
# ===========================================================================
def get_child_vectorstore(
    persist_directory: str = CHILD_CHROMA_DIR,
    collection_name: str = CHILD_COLLECTION_NAME,
) -> Chroma:
    """Base Chroma des petits chunks « enfants » (avec embeddings Gemini)."""
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
    )


def get_parent_docstore(directory: str = PARENT_DOCSTORE_DIR) -> BaseStore[str, Document]:
    """Docstore persistant des « parents » (LocalFileStore + (dé)sérialisation).

    `create_kv_docstore` enveloppe le stockage d'octets pour y ranger/relire
    directement des objets `Document`.
    """
    return create_kv_docstore(LocalFileStore(directory))


def _child_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )


def _parent_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
    )


def get_parent_document_retriever(
    child_vectorstore: Chroma | None = None,
    docstore: BaseStore[str, Document] | None = None,
) -> ParentDocumentRetriever:
    """Construit le `ParentDocumentRetriever` (enfants Chroma + parents docstore).

    Réutilisable aussi bien pour l'indexation (`add_documents`) que pour le
    retrieval (les deux stockages sont persistants).
    """
    child_vectorstore = child_vectorstore or get_child_vectorstore()
    docstore = docstore if docstore is not None else get_parent_docstore()
    return ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=docstore,
        child_splitter=_child_splitter(),
        parent_splitter=_parent_splitter(),
    )


def advanced_index_exists(
    child_dir: str = CHILD_CHROMA_DIR, docstore_dir: str = PARENT_DOCSTORE_DIR
) -> bool:
    """Vrai si l'index parent-enfant (base enfant + docstore) existe déjà."""
    child_ok = os.path.isdir(child_dir) and bool(os.listdir(child_dir))
    parent_ok = os.path.isdir(docstore_dir) and bool(os.listdir(docstore_dir))
    return child_ok and parent_ok


def _split_parent_child(
    documents: list[Document], id_key: str = "doc_id"
) -> tuple[list[tuple[str, Document]], list[Document]]:
    """Découpe les documents en parents (avec id) et en enfants (liés au parent).

    Reproduit la logique interne du `ParentDocumentRetriever` mais sans appel
    réseau, afin de maîtriser ensuite le débit d'embeddings.

    Returns:
        (parents [(id, doc)], enfants [doc avec metadata[id_key] = id parent]).
    """
    parent_splitter = _parent_splitter()
    child_splitter = _child_splitter()

    parent_pairs: list[tuple[str, Document]] = []
    children: list[Document] = []

    for document in documents:
        for parent in parent_splitter.split_documents([document]):
            parent_id = str(uuid.uuid4())
            parent_pairs.append((parent_id, parent))
            for child in child_splitter.split_documents([parent]):
                child.metadata[id_key] = parent_id
                children.append(child)

    return parent_pairs, children


def _add_children_with_retry(
    child_vectorstore: Chroma,
    children: list[Document],
    max_retries: int = 5,
    wait_seconds: int = 65,
    verbose: bool = True,
) -> None:
    """Ajoute un lot d'enfants en réessayant en cas de quota dépassé (429)."""
    for attempt in range(1, max_retries + 1):
        try:
            child_vectorstore.add_documents(children)
            return
        except Exception as exc:
            message = str(exc)
            is_quota = "RESOURCE_EXHAUSTED" in message or "429" in message
            if not is_quota or attempt == max_retries:
                raise
            logger.warning(
                "[Vectorstore] Quota d'embeddings atteint — nouvelle "
                "tentative dans %ds (essai %d/%d)",
                wait_seconds,
                attempt,
                max_retries,
            )
            time.sleep(wait_seconds)


def build_parent_document_index(
    documents: list[Document],
    batch_size: int = EMBED_BATCH_SIZE,
    pause: int = EMBED_PAUSE,
    verbose: bool = True,
) -> ParentDocumentRetriever:
    """Indexe les documents complets en structure parent-enfant (quota-safe).

    Étapes :
        1. Découpage parent/enfant **local** (aucun appel API).
        2. Stockage des parents dans le docstore (aucun appel API).
        3. Embedding des enfants dans Chroma **par lots de ``batch_size`` avec
           pauses de ``pause`` s** + retry sur 429, pour rester sous la limite
           de l'API Gemini (free tier : 100 requêtes d'embedding / minute).

    Args:
        documents: documents wiki complets et nettoyés (non découpés).
        batch_size: nombre d'enfants embarqués par lot (≈ requêtes/minute).
        pause: pause (s) entre deux lots d'enfants.
    """
    if not documents:
        raise ValueError("Aucun document à indexer : la liste fournie est vide.")

    child_vectorstore = get_child_vectorstore()
    docstore = get_parent_docstore()

    # 1-2. Découpage local + stockage des parents (sans réseau).
    parent_pairs, children = _split_parent_child(documents)
    docstore.mset(parent_pairs)
    logger.info(
        "[Ingestion] %d parents stockés, %d enfants à embarquer",
        len(parent_pairs),
        len(children),
    )

    # 3. Embedding des enfants par lots throttlés.
    total = len(children)
    total_batches = (total + batch_size - 1) // batch_size
    for b, i in enumerate(range(0, total, batch_size), start=1):
        if b > 1:
            logger.info("[Ingestion] Pause %ds avant lot %d/%d...", pause, b, total_batches)
            time.sleep(pause)
        _add_children_with_retry(child_vectorstore, children[i : i + batch_size])
        logger.info(
            "[Ingestion] Lot %d/%d OK (%d/%d enfants)",
            b,
            total_batches,
            min(i + batch_size, total),
            total,
        )

    logger.info(
        "[Ingestion] Terminé : %d enfants indexés, %d parents.",
        child_vectorstore._collection.count(),
        len(parent_pairs),
    )
    return get_parent_document_retriever(child_vectorstore, docstore)

"""Récupération avancée : Multi-Query → RAG-Fusion (RRF) → Re-Ranking.

Ce module porte la logique de récupération du pipeline avancé, indépendamment
de l'orchestration (qui vit dans ``src/rag.py``) :

* **Multi-Query** — génération de reformulations de la question (logique
  inspirée du ``MultiQueryRetriever`` du notebook prototype, mais en français
  et avec un nombre fixe de variantes).
* **RAG-Fusion** — recherche vectorielle pour chaque requête sur la base
  « enfants », remontée au parent, puis fusion des classements par
  **Reciprocal Rank Fusion (RRF)**.
* **Re-Ranking** — ré-ordonnancement sémantique du top fusionné avec FlashRank
  (modèle local multilingue) pour ne garder que les meilleurs documents.
"""

import logging
import re
from functools import lru_cache

from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.stores import BaseStore

from src.config import (
    BM25_K,
    CHILD_SEARCH_K,
    FUSION_TOP_N,
    NUM_QUERIES,
    RERANK_TOP_N,
    RERANKER_MODEL,
    RRF_K,
    USE_HYBRID,
)
from src.generator import StrChain

logger = logging.getLogger(__name__)

# Cache des index BM25 par base d'enfants (id) : construit une fois, réutilisé
# pour toutes les questions d'une même session (évite de re-tokeniser le corpus).
_bm25_cache: dict[int, BM25Retriever] = {}


def _tokenize_fr(text: str) -> list[str]:
    """Tokenizer lexical simple pour le français : minuscules + mots alphanum.

    Insensible à la casse et à la ponctuation, ce qui aide BM25 à apparier les
    noms propres du corpus (« WICKED », « Griffeur », « Braise »…).
    """
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# 1. Multi-Query
# ---------------------------------------------------------------------------
def generate_query_variants(
    question: str, multiquery_chain: StrChain, num_queries: int = NUM_QUERIES
) -> list[str]:
    """Génère ``num_queries`` reformulations et renvoie [original, *variantes].

    Args:
        question: question d'origine.
        multiquery_chain: chaîne LangChain (prompt → llm → parser) renvoyant le
            texte brut des reformulations (une par ligne).
        num_queries: nombre de reformulations souhaitées.

    Returns:
        Liste de requêtes : la question d'origine suivie des reformulations
        (dédupliquées, sans lignes vides).
    """
    logger.info("[Multi-Query] Génération de %d variante(s) pour : %s", num_queries, question)
    raw = multiquery_chain.invoke({"question": question, "num_queries": num_queries})
    variants = [line.strip(" -•\t") for line in raw.splitlines() if line.strip()]

    queries = [question]
    for v in variants:
        if v and v.lower() not in (q.lower() for q in queries):
            queries.append(v)
        if len(queries) >= num_queries + 1:
            break
    logger.debug("[Multi-Query] Variantes : %s", queries[1:])
    return queries


# ---------------------------------------------------------------------------
# 2. Recherche par requête + RAG-Fusion (RRF)
# ---------------------------------------------------------------------------
def search_parents_for_query(
    query: str,
    child_vectorstore: Chroma,
    docstore: BaseStore[str, Document],
    k: int = CHILD_SEARCH_K,
) -> list[tuple[str, Document]]:
    """Recherche les enfants les plus proches et remonte aux parents (dédupliqués).

    Returns:
        Liste ordonnée (du plus pertinent au moins pertinent) de couples
        ``(parent_id, parent_document)``.
    """
    child_hits = child_vectorstore.similarity_search(query, k=k)

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for child in child_hits:
        parent_id = child.metadata.get("doc_id")
        if parent_id is None or parent_id in seen:
            continue
        seen.add(parent_id)
        ordered_ids.append(parent_id)

    parents = docstore.mget(ordered_ids)
    return [
        (pid, pdoc) for pid, pdoc in zip(ordered_ids, parents, strict=False) if pdoc is not None
    ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, Document]]],
    k: int = RRF_K,
    top_n: int = FUSION_TOP_N,
) -> list[tuple[Document, float]]:
    """Fusionne plusieurs classements de parents via Reciprocal Rank Fusion.

    Score RRF d'un document = somme sur toutes les requêtes de 1 / (k + rang).

    Args:
        ranked_lists: une liste classée ``(parent_id, parent_doc)`` par requête.
        k: constante d'amortissement RRF (typiquement 60).
        top_n: nombre de documents fusionnés conservés.

    Returns:
        Liste ``(document, score_rrf)`` triée par score décroissant, tronquée.
    """
    scores: dict[str, float] = {}
    doc_by_id: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, (parent_id, parent_doc) in enumerate(ranked):
            scores[parent_id] = scores.get(parent_id, 0.0) + 1.0 / (k + rank + 1)
            doc_by_id.setdefault(parent_id, parent_doc)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [(doc_by_id[pid], score) for pid, score in ordered]


def _load_all_children(child_vectorstore: Chroma) -> list[Document]:
    """Charge tous les chunks « enfants » de Chroma (texte + métadonnées).

    Les enfants portent ``metadata['doc_id']`` (l'identifiant de leur parent),
    ce qui permet la même remontée enfant→parent que la recherche dense.
    """
    raw = child_vectorstore._collection.get(include=["documents", "metadatas"])
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(documents, metadatas, strict=False)
        if text
    ]


def bm25_rank_parents(
    query: str,
    child_vectorstore: Chroma,
    docstore: BaseStore[str, Document],
    k: int = BM25_K,
) -> list[tuple[str, Document]]:
    """Classe les parents par pertinence lexicale BM25, via les enfants.

    BM25 est appliqué sur les petits chunks « enfants » (plus précis que les
    parents pour les noms propres), puis les enfants sont remontés à leurs
    parents (dédupliqués), comme dans la recherche dense. L'index BM25 est
    construit une fois par base d'enfants puis mis en cache.

    Args:
        query: la requête lexicale (typiquement la question d'origine).
        child_vectorstore: base Chroma des enfants.
        docstore: docstore des parents (remontée enfant→parent).
        k: nombre de parents conservés.

    Returns:
        Liste ordonnée ``(parent_id, parent_document)`` (du plus pertinent au
        moins pertinent), vide si la base d'enfants est vide.
    """
    retriever = _bm25_cache.get(id(child_vectorstore))
    if retriever is None:
        children = _load_all_children(child_vectorstore)
        if not children:
            return []
        retriever = BM25Retriever.from_documents(children, preprocess_func=_tokenize_fr)
        _bm25_cache[id(child_vectorstore)] = retriever

    # On récupère plus d'enfants que k pour pouvoir dédupliquer vers k parents.
    retriever.k = max(k * 4, 20)
    hits = retriever.invoke(query)

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for child in hits:
        parent_id = child.metadata.get("doc_id")
        if parent_id is None or parent_id in seen:
            continue
        seen.add(parent_id)
        ordered_ids.append(parent_id)
        if len(ordered_ids) >= k:
            break

    parents = docstore.mget(ordered_ids)
    return [
        (pid, pdoc) for pid, pdoc in zip(ordered_ids, parents, strict=False) if pdoc is not None
    ]


def fusion_retrieve(
    queries: list[str],
    child_vectorstore: Chroma,
    docstore: BaseStore[str, Document],
    k: int = CHILD_SEARCH_K,
    rrf_k: int = RRF_K,
    top_n: int = FUSION_TOP_N,
    use_hybrid: bool = USE_HYBRID,
    bm25_k: int = BM25_K,
) -> list[tuple[Document, float]]:
    """Exécute la recherche par requête, ajoute un classement lexical BM25, fusionne par RRF.

    Args:
        queries: question d'origine suivie des reformulations Multi-Query.
        child_vectorstore: base Chroma des enfants (recherche dense).
        docstore: docstore des parents (remontée + index lexical BM25).
        k: enfants récupérés par requête dense.
        rrf_k: constante d'amortissement de la RRF.
        top_n: parents conservés après fusion.
        use_hybrid: si vrai, ajoute une liste lexicale BM25 (sur la question
            d'origine) aux classements denses avant la RRF.
        bm25_k: nombre de parents récupérés par BM25.

    Returns:
        Liste ``(document, score_rrf)`` triée par score décroissant, tronquée.
    """
    logger.info("[RAG-Fusion] %d requête(s) → recherche vectorielle (k=%d)...", len(queries), k)
    ranked_lists = [search_parents_for_query(q, child_vectorstore, docstore, k=k) for q in queries]

    if use_hybrid and queries:
        lexical = bm25_rank_parents(queries[0], child_vectorstore, docstore, k=bm25_k)
        if lexical:
            ranked_lists.append(lexical)
            logger.info("[RAG-Fusion] + liste lexicale BM25 (%d parents)", len(lexical))

    result = reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_n=top_n)
    logger.info("[RAG-Fusion] %d parents après RRF (top_n=%d)", len(result), top_n)
    return result


# ---------------------------------------------------------------------------
# 3. Re-Ranking (FlashRank local)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_reranker(top_n: int = RERANK_TOP_N) -> FlashrankRerank:
    """Instancie (et met en cache) le reranker FlashRank multilingue."""
    # `client` est construit dynamiquement par un validateur pydantic de
    # FlashrankRerank si absent ; le champ est pourtant déclaré obligatoire
    # dans le modèle, d'où l'ignore ciblé.
    return FlashrankRerank(model=RERANKER_MODEL, top_n=top_n)  # type: ignore[call-arg]


def rerank_documents(
    question: str, documents: list[Document], top_n: int = RERANK_TOP_N
) -> list[Document]:
    """Ré-ordonne sémantiquement les documents et conserve le top ``top_n``.

    Args:
        question: question d'origine (référence du reranking).
        documents: documents candidats (ex. top 15 issus de la fusion).
        top_n: nombre de documents conservés.

    Returns:
        Documents reclassés (les plus pertinents d'abord), avec le score de
        reranking dans ``metadata['relevance_score']``.
    """
    if not documents:
        return []
    logger.info("[Re-Ranking] %d documents candidats → top %d (FlashRank)", len(documents), top_n)
    reranker = get_reranker(top_n)
    result = list(reranker.compress_documents(documents=documents, query=question))
    scores = [round(float(d.metadata.get("relevance_score", 0)), 4) for d in result]
    logger.debug("[Re-Ranking] Scores : %s", scores)
    return result

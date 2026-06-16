"""Pipeline RAG avancé avec gestion des hallucinations.

Orchestration de bout en bout (couche B), par-dessus l'index parent-enfant :

    1. Multi-Query  — reformulations de la question.
    2. RAG-Fusion   — recherche par requête + Reciprocal Rank Fusion (RRF).
    3. Re-Ranking   — FlashRank : top 15 fusionné → top 4 sémantique.
    4. CRAG         — un LLM juge la pertinence du contexte (PERTINENT /
                      AMBIGU / HORS-SUJET) ; repli propre si insuffisant.
    5. Génération   — réponse ancrée au contexte (Gemini).
    6. Self-RAG     — auto-évaluation (fidélité + pertinence) puis correction.

Chaque étape est tracée par des logs clairs visibles depuis ``main.py``.
Des interrupteurs (``USE_*`` dans ``config.py``) permettent d'activer/désactiver
chaque brique — pratique pour l'évaluation comparative.
"""

import logging
from typing import TypedDict

from langchain_core.documents import Document

from src.config import (
    CHILD_SEARCH_K,
    CRAG_AMBIGUOUS,
    CRAG_IRRELEVANT,
    CRAG_RELEVANT,
    FALLBACK_ANSWER,
    FUSION_TOP_N,
    MAX_CORRECTIONS,
    NUM_QUERIES,
    RERANK_TOP_N,
    RRF_K,
    USE_CRAG,
    USE_HYBRID,
    USE_MULTIQUERY,
    USE_RERANK,
    USE_SELF_RAG,
)
from src.generator import (
    build_correction_chain,
    build_crag_grader_chain,
    build_generation_chain,
    build_multiquery_chain,
    build_selfrag_chain,
    format_docs,
)
from src.prompts import CRAG_INSUFFICIENT_NOTICE
from src.retrieval import (
    fusion_retrieve,
    generate_query_variants,
    rerank_documents,
)
from src.vectorstore import (
    advanced_index_exists,
    get_child_vectorstore,
    get_parent_docstore,
)

logger = logging.getLogger(__name__)


class SourceInfo(TypedDict):
    """Une source citée dans la réponse (page wiki + score de pertinence)."""

    title: str
    source: str
    score: float | None


class RagAnswer(TypedDict):
    """Résultat structuré renvoyé par ``RagPipeline.answer``."""

    answer: str
    grounded: bool
    crag_status: str
    sources: list[SourceInfo]
    contexts: list[str]
    queries: list[str]
    self_rag: str


class RagPipeline:
    """Pipeline RAG avancé (Multi-Query, RAG-Fusion, Re-Ranking, CRAG, Self-RAG)."""

    def __init__(
        self,
        verbose: bool = True,
        use_multiquery: bool = USE_MULTIQUERY,
        use_rerank: bool = USE_RERANK,
        use_crag: bool = USE_CRAG,
        use_self_rag: bool = USE_SELF_RAG,
        use_hybrid: bool = USE_HYBRID,
        provider: str | None = None,
        model: str | None = None,
    ):
        """Initialise le pipeline et toutes ses chaînes LLM.

        Args:
            verbose: si ``True``, les étapes sont loggées en INFO (sinon DEBUG).
            use_multiquery: active la génération de reformulations Multi-Query.
            use_rerank: active le re-ranking FlashRank.
            use_crag: active le grader CRAG (anti hors-sujet).
            use_self_rag: active l'auto-évaluation Self-RAG + correction.
            use_hybrid: ajoute une liste lexicale BM25 à la fusion RRF.
            provider: fournisseur LLM ("gemini", "groq", "ollama") ; défaut config.
            model: nom du modèle à utiliser (override du modèle par défaut du
                provider — ex. ``"mistral"`` pour Ollama).

        Raises:
            FileNotFoundError: si l'index parent-enfant n'a pas été construit.
        """
        if not advanced_index_exists():
            raise FileNotFoundError(
                "Index parent-enfant introuvable (chroma_children/ + "
                "parent_docstore/). Construisez-le d'abord avec : "
                "uv run python ingest.py"
            )

        self.verbose = verbose
        self.use_multiquery = use_multiquery
        self.use_rerank = use_rerank
        self.use_crag = use_crag
        self.use_self_rag = use_self_rag
        self.use_hybrid = use_hybrid
        self.provider = provider
        self.model = model

        # Stockages (lecture seule pour la couche requête).
        self.child_vectorstore = get_child_vectorstore()
        self.docstore = get_parent_docstore()

        # Chaînes LLM (toutes sur le provider/modèle choisi).
        self.multiquery_chain = build_multiquery_chain(provider, model)
        self.generation_chain = build_generation_chain(provider, model)
        self.crag_chain = build_crag_grader_chain(provider, model)
        self.selfrag_chain = build_selfrag_chain(provider, model)
        self.correction_chain = build_correction_chain(provider, model)

    # -- utilitaires --------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)
        else:
            logger.debug(message)

    @staticmethod
    def _sources(docs: list[Document]) -> list[SourceInfo]:
        sources: list[SourceInfo] = []
        for doc in docs:
            score = doc.metadata.get("relevance_score")
            sources.append(
                {
                    "title": doc.metadata.get("title", "?"),
                    "source": doc.metadata.get("source", "?"),
                    "score": round(float(score), 4) if score is not None else None,
                }
            )
        return sources

    def _fallback(self, crag_status: str, queries: list[str]) -> RagAnswer:
        return {
            "answer": FALLBACK_ANSWER,
            "grounded": False,
            "crag_status": crag_status,
            "sources": [],
            "contexts": [],
            "queries": queries,
            "self_rag": "n/a",
        }

    # -- étapes -------------------------------------------------------------
    def _grade_crag(self, question: str, context: str) -> str:
        verdict = self.crag_chain.invoke({"question": question, "context": context}).strip().upper()
        for status in (CRAG_IRRELEVANT, CRAG_AMBIGUOUS, CRAG_RELEVANT):
            if status in verdict:
                return status
        # Par prudence, un verdict illisible est traité comme AMBIGU.
        return CRAG_AMBIGUOUS

    def _self_reflect(self, question: str, context: str, answer: str) -> tuple[str, str]:
        """Auto-évalue puis corrige éventuellement la réponse.

        Returns:
            (réponse_finale, statut_self_rag).
        """
        for attempt in range(MAX_CORRECTIONS + 1):
            verdict = self.selfrag_chain.invoke(
                {"question": question, "context": context, "answer": answer}
            ).strip()

            if verdict.upper().startswith("OK"):
                self._log("[Self-RAG] Validation: OK")
                return answer, "OK"

            critique = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
            if attempt < MAX_CORRECTIONS:
                self._log(
                    f"[Self-RAG] Validation: A_CORRIGER → correction "
                    f"({attempt + 1}/{MAX_CORRECTIONS}) : {critique[:80]}"
                )
                answer = self.correction_chain.invoke(
                    {
                        "question": question,
                        "context": context,
                        "answer": answer,
                        "critique": critique,
                    }
                )
            else:
                self._log("[Self-RAG] Validation: A_CORRIGER (limite de corrections atteinte)")
                return answer, "A_CORRIGER"
        return answer, "OK"

    # -- point d'entrée -----------------------------------------------------
    def answer(self, question: str) -> RagAnswer:
        """Répond à une question via le pipeline RAG avancé complet."""
        self._log(f"\n=== Question : {question} ===")

        # 1. Multi-Query ----------------------------------------------------
        if self.use_multiquery:
            queries = generate_query_variants(question, self.multiquery_chain, NUM_QUERIES)
            self._log(
                f"[Multi-Query] {len(queries) - 1} variante(s) générée(s) (+ question originale)"
            )
        else:
            queries = [question]

        # 2. RAG-Fusion (RRF) ----------------------------------------------
        fused = fusion_retrieve(
            queries,
            self.child_vectorstore,
            self.docstore,
            k=CHILD_SEARCH_K,
            rrf_k=RRF_K,
            top_n=FUSION_TOP_N,
            use_hybrid=self.use_hybrid,
        )
        self._log(f"[RAG-Fusion] {len(fused)} parents fusionnés via RRF")
        candidates = [doc for doc, _ in fused]

        if not candidates:
            self._log("[RAG-Fusion] Aucun document récupéré → repli")
            return self._fallback(CRAG_IRRELEVANT, queries)

        # 3. Re-Ranking -----------------------------------------------------
        if self.use_rerank:
            top_docs = rerank_documents(question, candidates, RERANK_TOP_N)
            self._log(f"[Re-Ranking] top {len(top_docs)}/{len(candidates)} retenus (FlashRank)")
        else:
            top_docs = candidates[:RERANK_TOP_N]

        # 4. CRAG -----------------------------------------------------------
        context = format_docs(top_docs)
        crag_status = CRAG_RELEVANT
        if self.use_crag:
            crag_status = self._grade_crag(question, context)
            self._log(f"[CRAG] Statut: {crag_status}")

            if crag_status == CRAG_IRRELEVANT:
                self._log("[CRAG] Hors-sujet → repli (génération non appelée)")
                return self._fallback(crag_status, queries)

            if crag_status == CRAG_AMBIGUOUS:
                # On prévient explicitement le LLM que le contexte peut manquer.
                context = CRAG_INSUFFICIENT_NOTICE + context

        # 5. Génération -----------------------------------------------------
        answer = self.generation_chain.invoke({"context": context, "question": question})
        self._log("[Génération] réponse produite")

        # 6. Self-RAG -------------------------------------------------------
        self_rag_status = "désactivé"
        if self.use_self_rag:
            answer, self_rag_status = self._self_reflect(question, context, answer)

        return {
            "answer": answer,
            "grounded": True,
            "crag_status": crag_status,
            "sources": self._sources(top_docs),
            "contexts": [doc.page_content for doc in top_docs],
            "queries": queries,
            "self_rag": self_rag_status,
        }


# Singleton paresseux pour les usages simples (CLI, scripts).
# Stocké dans un conteneur mutable pour éviter le `global`.
_pipeline_ref: list[RagPipeline] = []


def answer_question(question: str) -> RagAnswer:
    """Répond à une question via un ``RagPipeline`` partagé (instancié à la demande)."""
    if not _pipeline_ref:
        _pipeline_ref.append(RagPipeline())
    return _pipeline_ref[0].answer(question)

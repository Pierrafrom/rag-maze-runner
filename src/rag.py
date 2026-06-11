"""Pipeline RAG de bout en bout avec gestion des hallucinations.

Orchestration de la couche question-réponse par-dessus la base Chroma déjà
construite par P1 (connexion en lecture seule) :

    1. **Récupération** des k chunks les plus proches, avec calcul du score de
       similarité **cosinus** entre la requête et chaque chunk.
    2. **Filtre par seuil** (anti-hallucination n°1) : on écarte tout chunk
       dont le score cosinus est inférieur à ``SIMILARITY_THRESHOLD``.
    3. **Réponse de repli** (anti-hallucination n°2) : si plus aucun chunk ne
       passe le seuil, on renvoie ``FALLBACK_ANSWER`` sans appeler le LLM.
    4. **Génération** : sinon, on assemble le contexte et on interroge le LLM
       via la chaîne de génération.

Le score cosinus est recalculé manuellement à partir des vecteurs stockés
(plutôt que via le score de distance brut de Chroma), afin que le seuil de
0.5 ait une signification stable quelle que soit la métrique de la collection.
"""

from langchain_core.documents import Document

from src.config import (
    FALLBACK_ANSWER,
    RETRIEVER_K,
    SIMILARITY_THRESHOLD,
)
from src.generator import build_generation_chain, format_docs
from src.vectorstore import get_embeddings, load_vectorstore


def _cosine_similarity(vec_a, vec_b) -> float:
    """Similarité cosinus entre deux vecteurs (Python pur, sans dépendance)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class RagPipeline:
    """Pipeline de question-réponse RAG avec garde-fous anti-hallucination."""

    def __init__(
        self,
        k: int = RETRIEVER_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        """Initialise le pipeline.

        Args:
            k: nombre de chunks récupérés par requête.
            similarity_threshold: score cosinus minimal pour retenir un chunk.
        """
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.embeddings = get_embeddings()
        # Connexion en lecture seule à la base construite par P1.
        self.vectorstore = load_vectorstore()
        self.generation_chain = build_generation_chain()

    def retrieve(self, question: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Récupère les k chunks les plus proches avec leur score cosinus.

        Returns:
            Liste de couples ``(Document, score_cosinus)`` triée par score
            décroissant.
        """
        k = k or self.k
        query_vec = self.embeddings.embed_query(question)

        # On interroge directement la collection Chroma pour récupérer aussi
        # les vecteurs stockés (nécessaires au calcul du cosinus).
        result = self.vectorstore._collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "embeddings"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        embeddings = result.get("embeddings", [[]])[0]

        scored: list[tuple[Document, float]] = []
        for text, metadata, embedding in zip(documents, metadatas, embeddings):
            score = _cosine_similarity(query_vec, embedding)
            doc = Document(page_content=text, metadata=metadata or {})
            scored.append((doc, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def answer(self, question: str) -> dict:
        """Répond à une question avec gestion des hallucinations.

        Returns:
            Un dict :
            - ``answer`` : la réponse (ou la réponse de repli) ;
            - ``grounded`` : ``True`` si la réponse s'appuie sur le wiki ;
            - ``sources`` : liste de dicts ``{title, source, score}`` ;
            - ``scores`` : scores cosinus des chunks retenus.
        """
        scored = self.retrieve(question)
        kept = [
            (doc, score)
            for doc, score in scored
            if score >= self.similarity_threshold
        ]

        # Anti-hallucination : aucun contexte fiable → repli, sans appeler le LLM.
        if not kept:
            return {
                "answer": FALLBACK_ANSWER,
                "grounded": False,
                "sources": [],
                "scores": [],
            }

        context = format_docs([doc for doc, _ in kept])
        answer = self.generation_chain.invoke(
            {"context": context, "question": question}
        )

        sources = [
            {
                "title": doc.metadata.get("title", "?"),
                "source": doc.metadata.get("source", "?"),
                "score": round(score, 4),
            }
            for doc, score in kept
        ]
        return {
            "answer": answer,
            "grounded": True,
            "sources": sources,
            "scores": [round(score, 4) for _, score in kept],
        }


# Singleton paresseux pour les usages simples (CLI, scripts).
_pipeline: RagPipeline | None = None


def answer_question(question: str) -> dict:
    """Répond à une question via un ``RagPipeline`` partagé (instancié à la demande)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline.answer(question)

"""Application Streamlit — interface chat pour le RAG Maze Runner.

Couche présentation au-dessus de `RagPipeline` (src/rag.py). Le pipeline est
mis en cache via `@st.cache_resource` pour n'être reconstruit qu'une seule
fois par combinaison d'interrupteurs (connexion en lecture seule à l'index
parent-enfant — aucune écriture, aucun appel à `ingest.py`).

Lancement :
    uv run streamlit run streamlit_app.py

Prérequis : `.env` avec une clé API valide (GOOGLE_API_KEY ou GROQ_API_KEY
selon LLM_PROVIDER) et l'index parent-enfant déjà construit (`ingest.py`).
"""

import logging

import streamlit as st

from src.config import GOOGLE_API_KEY, GROQ_API_KEY, LLM_PROVIDER
from src.logging_config import setup_logging
from src.rag import RagAnswer, RagPipeline

logger = logging.getLogger(__name__)

setup_logging()

st.set_page_config(page_title="Maze Runner RAG", page_icon="🏃", layout="wide")

_CRAG_BADGE = {"PERTINENT": "🟢", "AMBIGU": "🟡", "HORS-SUJET": "🔴"}
_SELF_RAG_BADGE = {"OK": "🟢", "A_CORRIGER": "🟠"}


@st.cache_resource(show_spinner="Connexion à l'index parent-enfant...")
def load_pipeline(
    use_multiquery: bool, use_rerank: bool, use_crag: bool, use_self_rag: bool
) -> RagPipeline:
    """Instancie (et met en cache) le pipeline pour une combinaison d'interrupteurs donnée."""
    logger.info(
        "[Streamlit] Instanciation RagPipeline (multiquery=%s, rerank=%s, crag=%s, self_rag=%s)",
        use_multiquery,
        use_rerank,
        use_crag,
        use_self_rag,
    )
    return RagPipeline(
        verbose=False,
        use_multiquery=use_multiquery,
        use_rerank=use_rerank,
        use_crag=use_crag,
        use_self_rag=use_self_rag,
    )


def _missing_api_key() -> str | None:
    """Renvoie un message d'erreur si la clé API requise par LLM_PROVIDER est absente."""
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        return "LLM_PROVIDER=groq mais GROQ_API_KEY n'est pas définie dans .env"
    if LLM_PROVIDER != "groq" and not GOOGLE_API_KEY:
        return "GOOGLE_API_KEY n'est pas définie dans .env"
    return None


def _render_meta(result: RagAnswer, show_sources: bool, show_queries: bool) -> None:
    """Affiche les statuts CRAG/Self-RAG, les sources et les reformulations d'une réponse."""
    crag_badge = _CRAG_BADGE.get(result["crag_status"], "⚪")
    self_rag_badge = _SELF_RAG_BADGE.get(result["self_rag"], "⚪")
    st.caption(
        f"{crag_badge} CRAG : {result['crag_status']} · "
        f"{self_rag_badge} Self-RAG : {result['self_rag']}"
    )

    if show_queries and len(result["queries"]) > 1:
        with st.expander("🔄 Reformulations (Multi-Query)"):
            for query in result["queries"]:
                st.write(f"- {query}")

    if show_sources and result["sources"]:
        with st.expander(f"🔗 Sources ({len(result['sources'])})"):
            for src in result["sources"]:
                score = f" — score {src['score']}" if src["score"] is not None else ""
                st.markdown(f"- [{src['title']}]({src['source']}){score}")


def main() -> None:
    """Point d'entrée de l'application Streamlit."""
    st.title("🏃 Wiki Maze Runner — Assistant IA")
    st.caption(
        "Posez vos questions sur l'univers de *L'Épreuve* (Maze Runner) — "
        "réponses ancrées sur le wiki Fandom FR, avec gestion des hallucinations."
    )

    error = _missing_api_key()
    if error:
        st.error(f"⚠️ {error}")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Configuration du pipeline")
        st.caption(f"Fournisseur LLM actif : **{LLM_PROVIDER}**")
        use_multiquery = st.checkbox(
            "Multi-Query", value=True, help="3 reformulations de la question avant recherche"
        )
        use_rerank = st.checkbox(
            "Re-Ranking (FlashRank)", value=True, help="Reclassement sémantique top 15 → top 4"
        )
        use_crag = st.checkbox(
            "CRAG", value=True, help="Grader de pertinence du contexte (anti hors-sujet)"
        )
        use_self_rag = st.checkbox(
            "Self-RAG", value=True, help="Auto-évaluation de la réponse + correction"
        )
        st.divider()
        show_sources = st.checkbox("Afficher les sources", value=True)
        show_queries = st.checkbox("Afficher les reformulations", value=False)
        st.divider()
        if st.button("🗑️ Vider l'historique"):
            st.session_state.messages = []
            st.rerun()

    try:
        pipeline = load_pipeline(use_multiquery, use_rerank, use_crag, use_self_rag)
    except FileNotFoundError as exc:
        st.error(f"⚠️ {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            meta = message.get("meta")
            if message["role"] == "assistant" and meta is not None:
                _render_meta(meta, show_sources, show_queries)

    question = st.chat_input("Pose ta question sur le Labyrinthe...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question, "meta": None})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                result = pipeline.answer(question)
            st.markdown(result["answer"])
            _render_meta(result, show_sources, show_queries)

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "meta": result}
        )


if __name__ == "__main__":
    main()

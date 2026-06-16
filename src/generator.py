"""Couche LLM : génération + chaînes de contrôle (Multi-Query, CRAG, Self-RAG).

Ce module construit les chaînes LangChain (LCEL) qui appellent le LLM configuré
(Gemini ou Groq, selon ``LLM_PROVIDER`` dans ``config.py``) :

* génération de la réponse (température ``LLM_TEMPERATURE``) ;
* génération des reformulations multi-query ;
* évaluation CRAG de la pertinence du contexte ;
* auto-évaluation Self-RAG de la réponse + correction.

Les graders tournent à température 0 (déterminisme). L'orchestration du flux
vit dans ``src/rag.py``.
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import (
    GRADER_TEMPERATURE,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)
from src.prompts import (
    CORRECTION_PROMPT_TEMPLATE,
    CRAG_GRADER_PROMPT_TEMPLATE,
    MULTIQUERY_PROMPT_TEMPLATE,
    QA_PROMPT_TEMPLATE,
    SELFRAG_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Type d'une chaîne LCEL « variables → réponse texte » (prompt | llm | parser).
StrChain = Runnable[dict[str, Any], str]


def get_llm(model: str | None = None, temperature: float = LLM_TEMPERATURE) -> BaseChatModel:
    """Instancie le LLM actif selon LLM_PROVIDER (Gemini ou Groq).

    Args:
        model: nom du modèle (override de LLM_MODEL / GROQ_MODEL si fourni).
        temperature: température de génération.

    Returns:
        Une instance ``BaseChatModel`` compatible LangChain.

    Raises:
        ImportError: si langchain-groq n'est pas installé et LLM_PROVIDER="groq".
    """
    if LLM_PROVIDER == "groq":
        try:
            from langchain_groq import ChatGroq  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain-groq n'est pas installé. Exécutez : uv add langchain-groq"
            ) from exc
        groq_model = model or GROQ_MODEL
        logger.info("[LLM] Groq — %s (temp=%.1f)", groq_model, temperature)
        return ChatGroq(
            model_name=groq_model, temperature=temperature, api_key=GROQ_API_KEY or None
        )

    # Gemini (par défaut)
    gemini_model = model or LLM_MODEL
    logger.info("[LLM] Gemini — %s (temp=%.1f)", gemini_model, temperature)
    return ChatGoogleGenerativeAI(model=gemini_model, temperature=temperature)


def get_qa_prompt() -> PromptTemplate:
    """Prompt de question-réponse (variables : context, question)."""
    return PromptTemplate.from_template(QA_PROMPT_TEMPLATE)


def format_docs(docs: list[Document]) -> str:
    """Concatène le contenu des documents en un bloc de contexte unique."""
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Génération
# ---------------------------------------------------------------------------
def build_generation_chain() -> StrChain:
    """Chaîne de génération à contexte fourni : {context, question} → réponse."""
    return get_qa_prompt() | get_llm() | StrOutputParser()


def build_rag_chain(retriever: BaseRetriever) -> Runnable[str, str]:
    """Chaîne RAG LCEL classique (legacy, sans filtrage) : retriever → llm."""
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | get_qa_prompt()
        | get_llm()
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# Multi-Query (génération des reformulations)
# ---------------------------------------------------------------------------
def build_multiquery_chain() -> StrChain:
    """Chaîne renvoyant le texte brut des reformulations (une par ligne).

    Entrée : ``{"question": str, "num_queries": int}``.
    """
    prompt = PromptTemplate.from_template(MULTIQUERY_PROMPT_TEMPLATE)
    return prompt | get_llm() | StrOutputParser()


# ---------------------------------------------------------------------------
# CRAG (évaluation de la pertinence du contexte)
# ---------------------------------------------------------------------------
def build_crag_grader_chain() -> StrChain:
    """Chaîne d'évaluation CRAG : {question, context} → statut (un mot)."""
    prompt = PromptTemplate.from_template(CRAG_GRADER_PROMPT_TEMPLATE)
    return prompt | get_llm(temperature=GRADER_TEMPERATURE) | StrOutputParser()


# ---------------------------------------------------------------------------
# Self-RAG (auto-évaluation + correction)
# ---------------------------------------------------------------------------
def build_selfrag_chain() -> StrChain:
    """Chaîne d'auto-évaluation : {question, context, answer} → 'OK' | 'A_CORRIGER : …'."""
    prompt = PromptTemplate.from_template(SELFRAG_PROMPT_TEMPLATE)
    return prompt | get_llm(temperature=GRADER_TEMPERATURE) | StrOutputParser()


def build_correction_chain() -> StrChain:
    """Chaîne de correction : {question, context, answer, critique} → réponse corrigée."""
    prompt = PromptTemplate.from_template(CORRECTION_PROMPT_TEMPLATE)
    return prompt | get_llm() | StrOutputParser()

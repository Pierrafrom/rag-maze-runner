"""Couche de génération : LLM Gemini + prompt + chaîne LangChain (LCEL).

Ce module fournit les briques de génération réutilisées par ``src/rag.py`` :
le modèle de chat, le prompt de question-réponse et une chaîne LCEL simple.
La logique anti-hallucination (filtrage par score, repli) vit dans ``rag.py``.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_MODEL
from src.prompts import QA_PROMPT_TEMPLATE


def get_llm(model: str = LLM_MODEL) -> ChatGoogleGenerativeAI:
    """Instancie le modèle de chat Gemini."""
    return ChatGoogleGenerativeAI(model=model)


def get_qa_prompt() -> PromptTemplate:
    """Construit le ``PromptTemplate`` de question-réponse (variables : context, question)."""
    return PromptTemplate.from_template(QA_PROMPT_TEMPLATE)


def format_docs(docs) -> str:
    """Concatène le contenu des documents en un bloc de contexte unique."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_generation_chain():
    """Chaîne de génération à contexte déjà fourni.

    Entrée : un dict ``{"context": str, "question": str}``.
    Sortie : la réponse texte du LLM.

    Utilisée par ``RagPipeline`` une fois le contexte récupéré et validé.
    """
    prompt = get_qa_prompt()
    llm = get_llm()
    return prompt | llm | StrOutputParser()


def build_rag_chain(retriever):
    """Chaîne RAG LCEL classique : ``retriever → prompt → llm → parser``.

    Variante « bout en bout » sans filtrage par score (équivalente à celle du
    notebook). Pour la gestion des hallucinations, préférer ``RagPipeline``
    (cf. ``src/rag.py``).
    """
    prompt = get_qa_prompt()
    llm = get_llm()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

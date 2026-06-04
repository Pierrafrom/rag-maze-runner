from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_MODEL, LLM_PROMPT_TEMPLATE


def get_llm(model: str = LLM_MODEL) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model)


def get_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(LLM_PROMPT_TEMPLATE)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever):
    llm = get_llm()
    prompt = get_prompt()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def build_vectorstore(
    docs: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    embeddings = get_embeddings()
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


def load_vectorstore(persist_directory: str = CHROMA_PERSIST_DIR) -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

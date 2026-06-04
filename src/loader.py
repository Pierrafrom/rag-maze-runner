from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from src.config import SOURCE_URL, TEXT_START_DELIMITER, TEXT_END_DELIMITER


def load_web_document(url: str = SOURCE_URL) -> list[Document]:
    loader = WebBaseLoader(url)
    return loader.load()


def extract_relevant_text(
    docs: list[Document],
    start: str = TEXT_START_DELIMITER,
    end: str = TEXT_END_DELIMITER,
) -> list[Document]:
    raw_text = docs[0].page_content
    after_start = raw_text.split(start, 1)[1]
    relevant = after_start.split(end, 1)[0]
    return [Document(page_content=relevant, metadata={"source": docs[0].metadata.get("source", "web")})]


def load_and_prepare(url: str = SOURCE_URL) -> list[Document]:
    docs = load_web_document(url)
    return extract_relevant_text(docs)

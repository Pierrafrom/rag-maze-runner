import os
import sys

from src.config import CHROMA_PERSIST_DIR
from src.generator import build_rag_chain
from src.loader import load_and_prepare
from src.vectorstore import build_vectorstore, load_vectorstore


def main(question: str = "What is Gemini?") -> str:
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("GOOGLE_API_KEY environment variable is not set.")

    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Loading existing vector store from '{CHROMA_PERSIST_DIR}'...")
        vectorstore = load_vectorstore()
    else:
        print("Building vector store from web data...")
        docs = load_and_prepare()
        vectorstore = build_vectorstore(docs)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    rag_chain = build_rag_chain(retriever)

    print(f"\nQuestion: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is Gemini?"
    main(query)

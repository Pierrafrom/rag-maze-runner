import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.0-flash"

SOURCE_URL = "https://blog.google/technology/ai/google-gemini-ai/"

# Delimiters used to slice the relevant portion of the page
TEXT_START_DELIMITER = "code, audio, image and video."
TEXT_END_DELIMITER = "Cloud TPU v5p"

CHROMA_PERSIST_DIR = "./chroma_db"
RETRIEVER_K = 1

LLM_PROMPT_TEMPLATE = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {question} \nContext: {context} \nAnswer:"""

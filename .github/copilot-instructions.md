# Copilot instructions

## Build, test, lint commands
No build/test/lint scripts are defined. This repo is a single Jupyter notebook; run cells top-to-bottom in Jupyter or Colab.

## High-level architecture
The project is entirely contained in `Gemini_LangChain_QA_Chroma_WebLoad.ipynb` and implements a RAG pipeline:
1. **Setup + auth**: installs `langchain`, `langchain-google-genai`, `langchain-community`, `chromadb`, and `bs4`; reads `GOOGLE_API_KEY` from Colab secrets and sets it in `os.environ`.
2. **Retriever**: `WebBaseLoader` fetches the Gemini blog page, the text is sliced to a relevant section and wrapped as a `Document`, embeddings are generated with `GoogleGenerativeAIEmbeddings` (`models/gemini-embedding-001`), and stored in Chroma at `./chroma_db`. A retriever is created with `k=1`.
3. **Generator**: `ChatGoogleGenerativeAI` (gemini-3.5-flash) + `PromptTemplate` + LCEL chain (`retriever -> format_docs -> prompt -> llm -> StrOutputParser`), invoked via `rag_chain.invoke(...)`.

## Key conventions
- The notebook assumes Colab (`from google.colab import userdata`) and reads `GOOGLE_API_KEY` from a Colab Secret.
- Dependency pins in the `%pip install` cell are intentional to match older LangChain import paths (e.g., `PromptTemplate` from `langchain`, `WebBaseLoader` from `langchain.document_loaders`, `Chroma` from `langchain.vectorstores`).
- The Chroma persistence directory is `./chroma_db` and is used both when creating the vector store and when reloading it.
- The source content is narrowed via string `split()` markers on the Gemini blog page; if the source URL changes, update the split markers or use a different extraction strategy.

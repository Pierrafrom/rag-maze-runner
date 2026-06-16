"""Configuration centralisée du RAG Maze Runner.

Tous les paramètres réglables du projet sont regroupés ici afin de garder un
point d'entrée unique pour les modèles, les sources de données, le chunking,
l'indexation et la couche de récupération/génération (y compris le pipeline
avancé : multi-représentation, RAG-Fusion, re-ranking, CRAG, Self-RAG).
"""

import os

from dotenv import load_dotenv

# Charge les variables d'environnement depuis un fichier .env si présent.
load_dotenv()

# --- Authentification -------------------------------------------------------
# Plusieurs clés permettent la rotation automatique sur quota 429.
# Définir GOOGLE_API_KEY (obligatoire), puis optionnellement GOOGLE_API_KEY_2,
# GOOGLE_API_KEY_3, GOOGLE_API_KEY_4 dans le .env pour augmenter le quota.
GOOGLE_API_KEYS: list[str] = [
    k
    for k in [
        os.environ.get("GOOGLE_API_KEY", ""),
        os.environ.get("GOOGLE_API_KEY_2", ""),
        os.environ.get("GOOGLE_API_KEY_3", ""),
        os.environ.get("GOOGLE_API_KEY_4", ""),
    ]
    if k
]
GOOGLE_API_KEY = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else ""

# --- Modèles d'embedding ----------------------------------------------------
# IMPORTANT : provider ET modèle doivent rester IDENTIQUES entre l'indexation
# et l'interrogation — tout changement impose de reconstruire l'index.
#
# Fournisseur actif : "gemini" (défaut), "huggingface" ou "ollama" (local).
#   EMBEDDING_PROVIDER=gemini       → API Gemini, rotation auto sur 429 si
#                                     plusieurs GOOGLE_API_KEY_* définis.
#   EMBEDDING_PROVIDER=huggingface  → sentence-transformers local (sans quota).
#   EMBEDDING_PROVIDER=ollama       → embeddings servis par Ollama (sans quota,
#                                     idéal pour un déploiement 100 % offline).
# ⚠️ L'index enfant est stocké dans un dossier PROPRE à chaque provider
#    (``chroma_children_<provider>/``) : on peut donc garder en parallèle un
#    index Gemini (qualité cloud) et un index local, sans que l'un écrase l'autre.
EMBEDDING_PROVIDER: str = os.environ.get("EMBEDDING_PROVIDER", "gemini")

# Modèle HuggingFace utilisé quand EMBEDDING_PROVIDER="huggingface".
# Recommandations (French-friendly, par ordre qualité/poids) :
#   paraphrase-multilingual-mpnet-base-v2  — 768 dims, ~420 MB (défaut)
#   paraphrase-multilingual-MiniLM-L12-v2 — 384 dims, ~120 MB (léger)
#   BAAI/bge-m3                            — 1024 dims, ~570 MB (excellent)
#   dangvantuan/sentence-camembert-large   — 1024 dims, ~1.3 GB (FR natif)
HF_EMBEDDING_MODEL: str = os.environ.get(
    "HF_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)

# Modèle Gemini (utilisé quand EMBEDDING_PROVIDER="gemini").
EMBEDDING_MODEL = "models/gemini-embedding-001"

# --- Fournisseur LLM --------------------------------------------------------
# "gemini" (défaut) → API Google Gemini (ChatGoogleGenerativeAI).
# "groq"            → API Groq Cloud (ChatGroq), LLM open-source ultra-rapide.
#                     Prérequis : uv add langchain-groq + GROQ_API_KEY dans .env.
# "ollama"          → LLM local servi par Ollama (ChatOllama), SANS quota.
#                     Prérequis : Ollama lancé (`ollama serve`) + modèle tiré
#                     (`ollama pull mistral`). Idéal pour l'évaluation hors quota.
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "gemini")

# Clé et modèle Groq (ignorés si LLM_PROVIDER="gemini").
# Modèles Groq disponibles (gratuits) : llama3-8b-8192, llama3-70b-8192,
# mixtral-8x7b-32768, gemma2-9b-it … Voir : https://console.groq.com/docs/models
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

# --- Ollama (LLM local, ignoré si LLM_PROVIDER != "ollama") -----------------
# Modèles recommandés (bon français, exécutables sur un laptop) :
#   mistral     — Mistral 7B, ~4.4 Go RAM (Q4), très bon en FR (défaut)
#   gemma3:4b   — Gemma 3 4B (Google), ~3.3 Go RAM, excellent rapport qualité/poids
#   llama3.1:8b — Llama 3.1 8B (Meta), ~4.9 Go RAM, juge robuste pour l'éval
# Modèle d'embedding local Ollama (pour le juge RAGAS hors quota) :
#   nomic-embed-text — ~280 Mo, multilingue correct.
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL: str = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# Modèles locaux proposés dans le sélecteur Streamlit (séparés par des virgules).
LOCAL_MODELS: list[str] = [
    m.strip() for m in os.environ.get("LOCAL_MODELS", "mistral,gemma3:4b").split(",") if m.strip()
]

# Modèle Gemini (ignoré si LLM_PROVIDER="groq"/"ollama").
LLM_MODEL = "gemini-2.5-flash"
# Température de génération (le notebook prototype utilisait 0.3).
LLM_TEMPERATURE = 0.3
# Les graders (CRAG, Self-RAG) doivent être déterministes → température 0.
GRADER_TEMPERATURE = 0.0

# --- Sources : wiki Fandom FR du Labyrinthe (Maze Runner) -------------------
WIKI_BASE_URL = "https://mazerunner.fandom.com"
WIKI_API_URL = "https://mazerunner.fandom.com/fr/api.php"

# Liste figée des pages à indexer (personnages, lieux, créatures, œuvres).
_WIKI_PATHS = [
    # Personnages principaux
    "/fr/wiki/Thomas",
    "/fr/wiki/Teresa",
    "/fr/wiki/Newt",
    "/fr/wiki/Minho",
    "/fr/wiki/Alby",
    "/fr/wiki/Chuck",
    "/fr/wiki/Gally",
    "/fr/wiki/Brenda",
    "/fr/wiki/Jorge",
    "/fr/wiki/Aris",
    "/fr/wiki/Janson",
    "/fr/wiki/Ava_Paige",
    "/fr/wiki/Vince",
    "/fr/wiki/Mary_Cooper",
    "/fr/wiki/Lawrence",
    # Lieux
    "/fr/wiki/Labyrinthe",
    "/fr/wiki/Terre_Br%C3%BBl%C3%A9e",
    "/fr/wiki/Denver",
    "/fr/wiki/Asheville",
    "/fr/wiki/Bunker",
    "/fr/wiki/Quartier_g%C3%A9n%C3%A9ral_du_WICKED",
    "/fr/wiki/Trou_des_Griffeurs",
    "/fr/wiki/Salle_des_cartes",
    # Créatures et concepts
    "/fr/wiki/Griffeur",
    "/fr/wiki/Fondu",
    "/fr/wiki/Immunis%C3%A9",
    "/fr/wiki/Bo%C3%AEte",
    "/fr/wiki/Effacement",
    "/fr/wiki/Rem%C3%A8de",
    "/fr/wiki/S%C3%A9rum",
    "/fr/wiki/Transformation",
    "/fr/wiki/Braise",
    # Livres et films
    "/fr/wiki/Le_Labyrinthe_(livre)",
    "/fr/wiki/Le_Labyrinthe_(film)",
    "/fr/wiki/La_Terre_Br%C3%BBl%C3%A9e_(livre)",
    "/fr/wiki/La_Terre_Br%C3%BBl%C3%A9e_(film)",
    "/fr/wiki/Le_Rem%C3%A8de_Mortel_(livre)",
    "/fr/wiki/Le_Rem%C3%A8de_Mortel_(film)",
    "/fr/wiki/L%27%C3%89preuve_(s%C3%A9rie)",
]

# URLs complètes consommées par le loader.
WIKI_URLS = [WIKI_BASE_URL + path for path in _WIKI_PATHS]

# --- Nettoyage / filtrage ---------------------------------------------------
MIN_PAGE_CHARS = 500  # page quasi vide ignorée au chargement
MIN_DOC_CHARS = 600  # document écarté avant chunking
REQUEST_DELAY = 0.3  # délai (s) entre deux requêtes à l'API du wiki
USER_AGENT = "MazeRunnerRAG/1.0 (projet pédagogique LO17 UTC)"

# --- Chunking (pipeline simple / legacy) ------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", ".", " "]

# --- Indexation multi-représentation (Parent Document Retriever) ------------
# Petits "enfants" embarqués dans Chroma ; gros "parents" stockés à part.
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 250
CHILD_CHUNK_OVERLAP = 50

# --- Indexation (embeddings) ------------------------------------------------
# Indexation par batchs avec pause pour respecter le quota de l'API Gemini.
EMBED_BATCH_SIZE = 80
EMBED_PAUSE = 65  # secondes

# --- Bases vectorielles / docstore ------------------------------------------
# Base "simple" historique (chunks de 1000 car.) — chemin RAG naïf conservé.
CHROMA_PERSIST_DIR = "./chroma_maze_runner"
CHROMA_COLLECTION_NAME = "langchain"

# Index avancé parent-enfant.
# Le dossier des vecteurs « enfants » dépend du provider d'embedding (espaces
# vectoriels incompatibles entre providers) ; les « parents » (texte brut) sont
# indépendants du provider et donc partagés.
CHILD_CHROMA_DIR = f"./chroma_children_{EMBEDDING_PROVIDER}"
CHILD_COLLECTION_NAME = "maze_children"
PARENT_DOCSTORE_DIR = "./parent_docstore"

# --- Récupération simple (legacy) -------------------------------------------
RETRIEVER_K = 4
SIMILARITY_THRESHOLD = 0.5  # seuil cosinus du pipeline simple

# --- Pipeline avancé : Multi-Query / RAG-Fusion -----------------------------
NUM_QUERIES = 3  # nombre de reformulations générées
CHILD_SEARCH_K = 10  # enfants récupérés par requête (avant fusion)
RRF_K = 60  # constante de la Reciprocal Rank Fusion
FUSION_TOP_N = 15  # parents conservés après fusion (entrée du reranker)

# --- Recherche hybride (lexical BM25 + dense) -------------------------------
# Le corpus est dense en noms propres (Thomas, WICKED, Griffeur, Braise…) que la
# recherche dense sous-récupère parfois ; BM25 (lexical) compense, et son
# classement est fusionné aux classements denses via la RRF existante.
USE_HYBRID = True  # ajoute une liste lexicale BM25 à la fusion RRF
BM25_K = 10  # parents lexicaux récupérés par BM25 (avant fusion)

# --- Re-ranking (FlashRank, local) ------------------------------------------
RERANK_TOP_N = 4  # documents conservés après reranking
RERANKER_MODEL = "ms-marco-MultiBERT-L-12"  # modèle multilingue (FR ok)

# --- CRAG (Corrective RAG) --------------------------------------------------
CRAG_RELEVANT = "PERTINENT"
CRAG_AMBIGUOUS = "AMBIGU"
CRAG_IRRELEVANT = "HORS-SUJET"

# --- Self-RAG (auto-évaluation) ---------------------------------------------
MAX_CORRECTIONS = 1  # nombre de passes de correction si la réponse échoue

# --- Interrupteurs d'étapes (utiles pour l'évaluation comparative) ----------
USE_MULTIQUERY = True
USE_RERANK = True
USE_CRAG = True
USE_SELF_RAG = True

# --- Réponse de repli (anti-hallucination) ----------------------------------
FALLBACK_ANSWER = (
    "Je ne dispose pas d'informations suffisantes dans le wiki pour répondre à cette question."
)

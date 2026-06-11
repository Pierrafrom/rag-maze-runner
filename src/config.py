"""Configuration centralisée du RAG Maze Runner.

Tous les paramètres réglables du projet sont regroupés ici afin de garder un
point d'entrée unique pour les modèles, les sources de données, le chunking,
l'indexation et la couche de récupération/génération.

Inspiré de la centralisation des réglages du projet de référence
``Pasteuraize`` (settings.py), adapté ici à un contexte RAG question-réponse.
"""

import os

from dotenv import load_dotenv

# Charge les variables d'environnement depuis un fichier .env si présent.
load_dotenv()

# --- Authentification -------------------------------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# --- Modèles Gemini ---------------------------------------------------------
# IMPORTANT : le modèle d'embedding doit rester identique à celui utilisé par
# P1 pour construire la base, sinon les vecteurs ne sont pas comparables.
EMBEDDING_MODEL = "models/gemini-embedding-001"
# gemini-2.0-flash renvoyait un quota free tier à 0 sur ce projet ;
# gemini-2.5-flash est disponible. Ajustable selon le quota du compte.
LLM_MODEL = "gemini-2.5-flash"

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
# Pages plus courtes que ce seuil ignorées au chargement (page quasi vide).
MIN_PAGE_CHARS = 500
# Documents plus courts que ce seuil écartés avant chunking (faible valeur).
MIN_DOC_CHARS = 600
# Délai (s) entre deux requêtes à l'API du wiki (politesse / anti-throttle).
REQUEST_DELAY = 0.3
# En-tête User-Agent envoyé à l'API MediaWiki.
USER_AGENT = "MazeRunnerRAG/1.0 (projet pédagogique LO17 UTC)"

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", ".", " "]

# --- Indexation (embeddings) ------------------------------------------------
# Indexation par batchs avec pause pour respecter le quota de l'API Gemini.
EMBED_BATCH_SIZE = 80
EMBED_PAUSE = 65  # secondes

# --- Base vectorielle Chroma ------------------------------------------------
CHROMA_PERSIST_DIR = "./chroma_maze_runner"
# Nom de collection par défaut de LangChain/Chroma (utilisé par P1 dans le
# notebook, qui n'a pas passé de collection_name explicite).
CHROMA_COLLECTION_NAME = "langchain"

# --- Récupération -----------------------------------------------------------
RETRIEVER_K = 4
# Seuil de similarité cosinus en dessous duquel un document est ignoré.
# Première barrière anti-hallucination (cf. src/rag.py).
SIMILARITY_THRESHOLD = 0.5

# --- Réponse de repli (anti-hallucination) ----------------------------------
FALLBACK_ANSWER = (
    "Je ne dispose pas d'informations suffisantes dans le wiki pour répondre "
    "à cette question."
)

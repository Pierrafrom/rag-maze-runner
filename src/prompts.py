"""Prompts du RAG Maze Runner (tous rédigés en français).

Regroupés dans un module dédié pour être facilement relus et modifiés. On y
trouve le prompt de génération ainsi que les prompts de contrôle des étapes
avancées : reformulation multi-query, évaluation CRAG et auto-évaluation
Self-RAG (+ correction).
"""

from src.config import (
    CRAG_AMBIGUOUS,
    CRAG_IRRELEVANT,
    CRAG_RELEVANT,
    FALLBACK_ANSWER,
    NUM_QUERIES,
)

# ---------------------------------------------------------------------------
# 1. Génération de la réponse (ancrée au contexte)
# ---------------------------------------------------------------------------
# Variables attendues : {context} et {question}.
QA_PROMPT_TEMPLATE = (
    "Tu es un assistant expert de l'univers *Le Labyrinthe* (Maze Runner). "
    "Tu réponds en français, en t'appuyant EXCLUSIVEMENT sur les extraits du "
    "wiki Fandom FR fournis dans le contexte ci-dessous.\n\n"
    "Règles impératives :\n"
    "- Utilise UNIQUEMENT les informations présentes dans le contexte. "
    "N'utilise JAMAIS tes connaissances générales sur les livres, les films "
    "ou la série Maze Runner en dehors de ce contexte.\n"
    "- Si la réponse n'est pas présente dans le contexte, réponds EXACTEMENT "
    'par la phrase suivante, sans rien ajouter : "' + FALLBACK_ANSWER + '"\n'
    "- Ne fabrique aucune information et ne fais aucune supposition.\n"
    "- Reste concis (5 phrases maximum) et appuie-toi sur les éléments du "
    "contexte.\n\n"
    "Contexte :\n{context}\n\n"
    "Question : {question}\n\n"
    "Réponse :"
)

# Avertissement injecté dans le contexte quand CRAG juge l'info AMBIGUË.
CRAG_INSUFFICIENT_NOTICE = (
    "[AVERTISSEMENT] Le contexte récupéré est peut-être incomplet ou seulement "
    "partiellement pertinent. Ne réponds que sur ce qui est explicitement "
    "présent ci-dessous ; pour tout élément manquant, indique clairement que "
    "le wiki ne le précise pas.\n\n"
)

# ---------------------------------------------------------------------------
# 2. Multi-Query : génération de reformulations (logique du notebook,
#    MultiQueryRetriever, mais avec un prompt français et un nombre fixe)
# ---------------------------------------------------------------------------
# Variables attendues : {num_queries} et {question}.
MULTIQUERY_PROMPT_TEMPLATE = (
    "Tu es un assistant de recherche documentaire sur l'univers du Labyrinthe "
    "(Maze Runner). À partir de la question de l'utilisateur, génère "
    "{num_queries} reformulations DIFFÉRENTES en français, qui expriment la "
    "même intention avec d'autres mots, synonymes ou angles, afin d'améliorer "
    "la recherche vectorielle.\n\n"
    "Contraintes :\n"
    "- Une reformulation par ligne, sans numérotation ni puce.\n"
    "- Pas de texte d'introduction ni de conclusion.\n"
    "- Reste fidèle au sens de la question d'origine.\n\n"
    "Question d'origine : {question}\n\n"
    "Reformulations :"
)
# Valeur par défaut utilisée si aucune n'est passée explicitement.
DEFAULT_NUM_QUERIES = NUM_QUERIES

# ---------------------------------------------------------------------------
# 3. CRAG : évaluation de la pertinence du contexte récupéré
# ---------------------------------------------------------------------------
# Variables attendues : {question} et {context}.
CRAG_GRADER_PROMPT_TEMPLATE = (
    "Tu es un évaluateur de pertinence pour un système de questions-réponses "
    "sur l'univers du Labyrinthe (Maze Runner). On te donne une question et "
    "des extraits de documents récupérés. Évalue si ces extraits suffisent à "
    "répondre correctement à la question.\n\n"
    "Réponds par UN SEUL mot, sans rien d'autre :\n"
    f"- {CRAG_RELEVANT} : les extraits contiennent clairement l'information "
    "nécessaire pour répondre.\n"
    f"- {CRAG_AMBIGUOUS} : les extraits sont partiellement liés mais "
    "incomplets ou imprécis.\n"
    f"- {CRAG_IRRELEVANT} : les extraits ne concernent pas la question.\n\n"
    "Question : {question}\n\n"
    "Extraits :\n{context}\n\n"
    "Évaluation (un seul mot) :"
)

# ---------------------------------------------------------------------------
# 4. Self-RAG : auto-évaluation de la réponse générée
# ---------------------------------------------------------------------------
# Variables attendues : {question}, {context}, {answer}.
# Réponse attendue : "OK" si la réponse est fidèle au contexte ET répond à la
# question ; sinon "A_CORRIGER : <raison courte>".
SELFRAG_PROMPT_TEMPLATE = (
    "Tu es un évaluateur qualité. On te donne une question, le contexte "
    "documentaire fourni, et une réponse générée. Vérifie DEUX choses :\n"
    "1. Fidélité (groundedness) : chaque affirmation de la réponse est-elle "
    "appuyée par le contexte (aucune information inventée) ?\n"
    "2. Pertinence : la réponse répond-elle réellement à la question posée ?\n\n"
    "Si les deux conditions sont remplies, réponds EXACTEMENT : OK\n"
    "Sinon, réponds : A_CORRIGER : <raison courte>\n\n"
    "Note : la phrase de repli indiquant l'absence d'information est considérée "
    "comme une réponse VALIDE (OK) si le contexte est effectivement insuffisant.\n\n"
    "Question : {question}\n\n"
    "Contexte :\n{context}\n\n"
    "Réponse générée :\n{answer}\n\n"
    "Verdict :"
)

# ---------------------------------------------------------------------------
# 5. Correction (déclenchée si Self-RAG échoue)
# ---------------------------------------------------------------------------
# Variables attendues : {question}, {context}, {answer}, {critique}.
CORRECTION_PROMPT_TEMPLATE = (
    "Tu es un assistant expert de l'univers du Labyrinthe (Maze Runner). Une "
    "première réponse a été jugée insuffisante. Corrige-la en respectant "
    "strictement le contexte fourni.\n\n"
    "Règles :\n"
    "- Utilise UNIQUEMENT le contexte ci-dessous ; n'invente rien.\n"
    "- Corrige le problème signalé par la critique.\n"
    "- Si le contexte ne permet pas de répondre, réponds EXACTEMENT : "
    '"' + FALLBACK_ANSWER + '"\n'
    "- Réponds en français, de manière concise (5 phrases maximum).\n\n"
    "Question : {question}\n\n"
    "Contexte :\n{context}\n\n"
    "Réponse à corriger :\n{answer}\n\n"
    "Critique : {critique}\n\n"
    "Réponse corrigée :"
)

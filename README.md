# RAG Maze Runner — LO17 / AI31 (UTC, Printemps 2026)

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)](pyproject.toml)
[![Chroma](https://img.shields.io/badge/vector%20store-Chroma-informational)](pyproject.toml)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](streamlit_app.py)
[![Docker](https://img.shields.io/badge/Docker-Ollama%20%2B%20Streamlit-2496ED?logo=docker&logoColor=fff)](docker-compose.yml)
[![Checks](https://img.shields.io/badge/ruff%20%7C%20mypy%20--strict%20%7C%20pytest-passing-brightgreen)](pyproject.toml)

Système de **RAG (Retrieval Augmented Generation)** qui répond **en français** à
des questions sur l'univers du **Labyrinthe (Maze Runner)**, à partir du contenu
du **wiki Fandom FR**, en s'appuyant sur **Google Gemini**, **LangChain** et la
base vectorielle **Chroma**.

> Projet de groupe (4 personnes). **Rendu : 14/06/2026.**

---

## 1. Objectif et attendus du sujet

D'après `Sujet/Consignes pour le Projet.pdf` :

- [x] **RAG sur un corpus en français** (choix motivé des données).
- [x] **Code reproductible** basé sur le notebook de référence.
- [x] **Gestion des hallucinations**.
- [x] **Évaluation du RAG** — jeu de questions de référence, RAGAS et
  comparaison hybride vs. naïf (`tests/evaluation/`).
- [x] **Application Streamlit** (`streamlit_app.py`).
- [x] **Bonus** : déploiement conteneurisé (Docker Compose — Streamlit + Ollama).

**Choix du corpus.** Le wiki Fandom FR du Labyrinthe est riche, structuré
(personnages, lieux, créatures, œuvres), entièrement en français, et stable —
idéal pour un RAG factuel et vérifiable.

---

## 2. Architecture

Pipeline **RAG avancé** organisé en deux couches.

### Couche A — Ingestion multi-représentation (`uv run python ingest.py`)

Indexation **Parent Document Retriever** : on découpe en petits **enfants**
(~250 car., embarqués dans Chroma `chroma_children/`) reliés à des **parents**
plus larges (~1500 car., stockés dans un docstore persistant `parent_docstore/`).
À la recherche, un enfant trouvé fait remonter son parent (plus de contexte).

1. **Chargement** (`src/loader.py`) — pages du wiki via l'**API MediaWiki**.
2. **Nettoyage** (BeautifulSoup) + **filtrage** des pages trop courtes.
3. **Indexation parent-enfant** (`src/vectorstore.py`) — embeddings Gemini des
   enfants **par lots avec pauses** (quota), parents dans le docstore.

### Couche B — Récupération + Génération avancée (`RagPipeline`, `src/rag.py`)

En **lecture seule**, 6 étapes (toutes activables/désactivables via `config.py`) :

1. **Multi-Query** (`src/retrieval.py`) — le LLM génère 3 reformulations FR de
   la question (logique du prototype `rag.ipynb`).
2. **RAG-Fusion** — recherche vectorielle pour chaque requête, remontée au
   parent, puis fusion des classements par **Reciprocal Rank Fusion (RRF)**.
3. **Re-Ranking** — **FlashRank** (modèle local multilingue) reclasse le top 15
   fusionné → **top 4** sémantique.
4. **CRAG** (anti-hallucination) — un LLM note le contexte
   `PERTINENT / AMBIGU / HORS-SUJET`. HORS-SUJET ⇒ repli sans génération ;
   AMBIGU ⇒ avertissement injecté dans le prompt.
5. **Génération** — réponse ancrée au contexte (Gemini), prompt contraint.
6. **Self-RAG** — auto-évaluation (fidélité + pertinence) ; si échec, une passe
   de correction avant renvoi.

```
Question
   │
   ▼
[Multi-Query] ─ 3 reformulations + question
   │
   ▼
[RAG-Fusion] ─ recherche enfants → parents → RRF (top 15)
   │
   ▼
[Re-Ranking] ─ FlashRank (top 15 → top 4)
   │
   ▼
[CRAG] ── HORS-SUJET ─▶ Réponse de repli (pas de génération)
   │ PERTINENT / AMBIGU
   ▼
[Génération] ─ LLM Gemini (contexte + prompt contraint)
   │
   ▼
[Self-RAG] ─ auto-évaluation → correction si besoin ─▶ Réponse + sources
```

---

## 3. Arborescence

```
.
├── ingest.py              # Construit la base vectorielle (couche A)
├── main.py                # CLI question → réponse (couche B)
├── pyproject.toml         # Dépendances du projet (gérées par uv)
├── uv.lock                # Versions verrouillées (reproductibilité)
├── .env.example           # Modèle de configuration (à copier en .env)
├── src/
│   ├── config.py          # Configuration centralisée (modèles, RRF, rerank, CRAG…)
│   ├── prompts.py         # Prompts FR (QA, multi-query, CRAG, Self-RAG, correction)
│   ├── loader.py          # Chargement wiki + nettoyage
│   ├── vectorstore.py     # Index parent-enfant (Chroma enfants + docstore parents)
│   ├── retrieval.py       # Multi-Query + RAG-Fusion (RRF) + Re-Ranking (FlashRank)
│   ├── generator.py       # Chaînes LLM (génération, multi-query, CRAG, Self-RAG)
│   └── rag.py             # RagPipeline (orchestration des 6 étapes)
├── chroma_children_gemini/ # Index « enfants » Gemini (construit par ingest.py)
│                           #   (chroma_children_ollama/ si embeddings locaux)
├── parent_docstore/       # Docstore persistant des « parents » (partagé)
├── Dockerfile             # Image Streamlit (LLM/index hors image)
├── docker-compose.yml     # Stack Streamlit + Ollama (LLM local)
├── chroma_maze_runner/    # Base du pipeline simple (legacy, comparaison naïf vs avancé)
├── rag.ipynb              # Notebook prototype d'origine
├── CLAUDE.md              # Guide interne (assistant) — non commité
└── suivi.md               # Journal des décisions et modifications
```

---

## 4. Installation

> Prérequis : [**uv**](https://docs.astral.sh/uv/) et une clé API Gemini
> gratuite (https://aistudio.google.com/app/apikey). `uv` gère lui-même Python
> (≥ 3.11) et l'environnement virtuel.

```bash
# 1. Dépendances : crée le .venv et installe tout depuis uv.lock
uv sync

# 2. Clé API : copier le modèle puis renseigner la clé
copy .env.example .env           # Windows  (cp .env.example .env sur Unix)
#   puis éditer .env :  GOOGLE_API_KEY=AIza...
```

> Toutes les commandes du projet se lancent ensuite avec **`uv run`**, qui
> utilise automatiquement l'environnement géré par uv (pas besoin de l'activer).
> Pour ajouter une dépendance : `uv add <paquet>` (met à jour `pyproject.toml`
> et `uv.lock`).

---

## 5. Utilisation

### Construire la base (une seule fois)

Les index vectoriels ne sont pas versionnés (générés, pas des sources) —
à construire avant la première utilisation :

```bash
uv run python ingest.py
```

> L'indexation est volontairement lente (batchs + pauses) pour respecter le
> quota de l'API d'embeddings Gemini.

### Poser une question

```bash
uv run python main.py "Quel est le but de WICKED ?"
```

Exemple de sortie (les étapes sont tracées en temps réel) :

```
[Multi-Query] 3 variante(s) générée(s) (+ question originale)
[RAG-Fusion] 15 parents fusionnés via RRF
[Re-Ranking] top 4/15 retenus (FlashRank)
[CRAG] Statut: PERTINENT
[Génération] réponse produite
[Self-RAG] Validation: OK
========================================================================
Question     : Quel est le but de WICKED ?
Statut CRAG  : PERTINENT
Self-RAG     : OK

Réponse      : WICKED cherche à étudier les réactions cérébrales des sujets
immunisés pour élaborer un remède contre la Braise...

Sources (wiki Fandom FR) :
  - Quartier_général_du_WICKED score=0.98 — https://mazerunner.fandom.com/fr/wiki/...
  ...
```

Pour une question hors corpus (« Quelle est la capitale de la France ? »), le
système renvoie : *« Je ne dispose pas d'informations suffisantes dans le wiki
pour répondre à cette question. »*

### Application Streamlit

```bash
uv run streamlit run streamlit_app.py
```

Interface chat au-dessus de `RagPipeline` : sidebar pour activer/désactiver
chaque étape (Multi-Query, Re-Ranking, CRAG, Self-RAG — utile pour comparer
RAG naïf vs avancé), affichage des statuts CRAG/Self-RAG, des sources et des
reformulations en option. Le pipeline est mis en cache (`@st.cache_resource`)
par combinaison d'interrupteurs : pas de reconnexion à l'index à chaque message.

Le **sélecteur de modèle** (sidebar) permet de basculer entre Gemini (API) et
des modèles **locaux via Ollama** (sans quota) — voir les 3 modes ci-dessous.

### Utilisation programmatique

```python
from src.rag import RagPipeline

pipeline = RagPipeline()                 # connexion lecture seule à la base
res = pipeline.answer("Qui est Newt ?")
print(res["answer"], res["grounded"], res["sources"])

# Modèle local (sans quota) :
local = RagPipeline(provider="ollama", model="mistral")
```

---

## 5 bis. Trois modes de lancement

| Mode | LLM | Embeddings | Quota / offline |
|---|---|---|---|
| **A — API Gemini** (défaut) | Gemini 2.5 Flash | Gemini | quota free tier |
| **B — LLM local** | Ollama (mistral…) | Gemini | LLM sans quota ; embeddings API |
| **C — 100 % offline** | Ollama | Ollama (`nomic-embed-text`) | aucun quota, sans clé |

### Mode A — API Gemini

```bash
uv sync && cp .env.example .env       # renseigner GOOGLE_API_KEY
uv run streamlit run streamlit_app.py
```

### Mode B — LLM local (Ollama), embeddings Gemini

```bash
ollama serve &              # démarrer Ollama
ollama pull mistral         # tirer un modèle
# .env : LLM_PROVIDER=ollama  (GOOGLE_API_KEY reste requise pour les embeddings)
uv run streamlit run streamlit_app.py
```

### Mode C — 100 % offline (LLM + embeddings locaux)

L'index étant lié au provider d'embedding, on le **reconstruit une fois** en
local (rapide : aucune pause anti-quota). L'index Gemini existant est conservé
dans `chroma_children_gemini/` ; le local va dans `chroma_children_ollama/`.

```bash
ollama pull mistral && ollama pull nomic-embed-text
# .env : LLM_PROVIDER=ollama  et  EMBEDDING_PROVIDER=ollama
uv run python ingest.py --force          # construit chroma_children_ollama/
uv run streamlit run streamlit_app.py    # plus aucune clé API requise
```

### Mode Docker (compose : Streamlit + Ollama)

```bash
docker compose up --build                       # http://localhost:8501
docker compose --profile bootstrap up ollama-init   # (1re fois) tire les modèles
```

Les index sont montés en volume (jamais dans l'image) ; Ollama tourne dans un
service dédié. Pour le mode 100 % offline en conteneur, mettre
`EMBEDDING_PROVIDER=ollama` dans `.env` et lancer l'ingestion une fois :
`docker compose run --rm rag-app uv run python ingest.py --force`.

---

## 6. Configuration

Tous les paramètres sont dans `src/config.py` :

| Paramètre | Rôle | Défaut |
|---|---|---|
| `EMBEDDING_MODEL` | Modèle d'embedding | `models/gemini-embedding-001` |
| `LLM_MODEL` / `LLM_TEMPERATURE` | Génération | `gemini-2.5-flash` / `0.3` |
| `PARENT_CHUNK_SIZE` / `CHILD_CHUNK_SIZE` | Découpage parent / enfant | `1500` / `250` |
| `NUM_QUERIES` | Reformulations multi-query | `3` |
| `RRF_K` / `FUSION_TOP_N` | Fusion RRF / top fusionné | `60` / `15` |
| `RERANK_TOP_N` / `RERANKER_MODEL` | Top rerank / modèle FlashRank | `4` / `ms-marco-MultiBERT-L-12` |
| `MAX_CORRECTIONS` | Passes de correction Self-RAG | `1` |
| `USE_MULTIQUERY/RERANK/CRAG/SELF_RAG` | Interrupteurs d'étapes | `True` |
| `FALLBACK_ANSWER` | Message de repli | *(voir fichier)* |

Les prompts sont isolés dans `src/prompts.py` pour être modifiés facilement.
Les interrupteurs `USE_*` permettent d'activer/désactiver chaque étape (utile
pour l'évaluation comparative et le RAG « naïf »).

---

## 7. Gestion des hallucinations

Plusieurs mécanismes complémentaires :

1. **CRAG (Corrective RAG)** — un LLM juge si le contexte récupéré est
   `PERTINENT / AMBIGU / HORS-SUJET`. Si HORS-SUJET, on renvoie le repli **sans
   appeler le LLM de génération** ; si AMBIGU, un avertissement est injecté dans
   le prompt.
2. **Prompt contraint** — interdit au LLM d'utiliser ses connaissances
   générales et lui impose de signaler explicitement l'absence d'information.
3. **Self-RAG (auto-évaluation)** — après génération, un LLM vérifie la
   **fidélité** (la réponse est appuyée par le contexte) et la **pertinence** ;
   en cas d'échec, une passe de correction est appliquée avant le renvoi.

---

## 8. Dépannage

| Symptôme | Cause | Solution |
|---|---|---|
| `module 'langchain' has no attribute 'verbose'` | Versions LangChain incohérentes dans l'environnement | Repartir d'un environnement propre : `uv sync` |
| Crash à l'import de `langchain_text_splitters` (Keras 3 / transformers) | Conflit TensorFlow/Keras | Géré automatiquement (`USE_TF=0` dans `loader.py`) |
| `400 API_KEY_INVALID` | Clé absente ou placeholder dans `.env` | Renseigner une vraie clé `GOOGLE_API_KEY` |
| `429 RESOURCE_EXHAUSTED` (`limit: 0`) | Quota du modèle épuisé / indisponible | Changer `LLM_MODEL` dans `src/config.py` (ex. `gemini-2.5-flash`) |
| `FileNotFoundError: Index parent-enfant introuvable` | Index avancé non construit | Lancer `uv run python ingest.py` |
| Téléchargement `ms-marco-MultiBERT-L-12...` au 1er lancement | FlashRank récupère le modèle de reranking (~100 Mo) | Normal, une seule fois (mis en cache ensuite) |

---

## 9. Suivi du projet

Toutes les décisions et modifications sont consignées dans **`suivi.md`**
(journal versionné). Toute évolution du code doit y être ajoutée.

---

## 10. Étapes suivantes (TODO)

- [x] **Application Streamlit** (`streamlit_app.py`) au-dessus de `RagPipeline`.
- [x] **Évaluation du RAG** : jeu de questions/réponses de référence + métriques
  RAGAS, comparaison hybride vs. naïf via les interrupteurs `USE_*`
  (`tests/evaluation/`).
- [x] **Versionnement des index** : décidé — non versionnés (générés par
  `uv run python ingest.py`, ignorés par git).
- [x] (Bonus) **Déploiement** : Docker Compose (Streamlit + Ollama, GPU
  optionnel).

---

## 11. Modèles utilisés

| Rôle | Modèle |
|---|---|
| Embeddings | `models/gemini-embedding-001` |
| LLM (génération) | `gemini-2.5-flash` (temp 0.3) |
| LLM (graders CRAG / Self-RAG) | `gemini-2.5-flash` (temp 0) |
| Re-ranking | `ms-marco-MultiBERT-L-12` (FlashRank, local multilingue) |

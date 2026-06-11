# RAG Maze Runner — LO17 / AI31 (UTC, Printemps 2026)

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
- [ ] **Évaluation du RAG** (à faire).
- [ ] **Application Streamlit** (à faire).
- [ ] **Bonus** : déploiement de l'application.

**Choix du corpus.** Le wiki Fandom FR du Labyrinthe est riche, structuré
(personnages, lieux, créatures, œuvres), entièrement en français, et stable —
idéal pour un RAG factuel et vérifiable.

---

## 2. Architecture

Le projet est organisé en **deux couches**.

### Couche A — Ingestion (construction de la base)

Reproductible via `python ingest.py` :

1. **Chargement** (`src/loader.py`) — récupération des pages via l'**API
   MediaWiki** (`action=parse`) du wiki, plus propre que le scraping HTML brut.
2. **Nettoyage** (BeautifulSoup) — suppression des `table`/`script`/`style`,
   extraction des paragraphes `<p>`.
3. **Filtrage** — pages/documents trop courts écartés (peu d'information).
4. **Découpage** — `RecursiveCharacterTextSplitter` (`chunk_size=1000`,
   `overlap=200`).
5. **Indexation** (`src/vectorstore.py`) — embeddings Gemini, stockés dans
   Chroma **par batchs avec pauses** (respect du quota de l'API).

### Couche B — Récupération + Génération (réponse aux questions)

Orchestrée par `RagPipeline` (`src/rag.py`), en **lecture seule** sur la base :

1. **Récupération** des `k` chunks les plus proches + calcul du **score de
   similarité cosinus** requête/chunk.
2. **Filtre par seuil** (anti-hallucination nº1) — on écarte les chunks dont le
   cosinus est inférieur au seuil (`SIMILARITY_THRESHOLD`).
3. **Réponse de repli** (anti-hallucination nº2) — si aucun chunk fiable, on
   renvoie un message standard **sans appeler le LLM**.
4. **Génération** — sinon, le contexte est assemblé et envoyé au LLM Gemini via
   une chaîne LangChain `prompt → llm → parser`.

```
Question ──▶ embeddings ──▶ Chroma (k plus proches) ──▶ filtre cosinus
                                                            │
                          ┌── aucun chunk fiable ───────────┤
                          ▼                                  ▼
                    Réponse de repli                  Contexte + prompt
                  (LLM non appelé)                          │
                                                            ▼
                                                     LLM Gemini ──▶ Réponse + sources
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
│   ├── config.py          # Configuration centralisée (modèles, URLs, seuils…)
│   ├── prompts.py         # Prompt de question-réponse (ancré au domaine)
│   ├── loader.py          # Chargement wiki + nettoyage + chunking
│   ├── vectorstore.py     # Embeddings + build/load Chroma
│   ├── generator.py       # LLM Gemini + chaînes LangChain
│   └── rag.py             # RagPipeline (retrieval + anti-hallucinations + génération)
├── chroma_maze_runner/    # Base Chroma persistée (323 vecteurs)
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

La base `chroma_maze_runner/` est **déjà fournie** (323 vecteurs). Pour la
reconstruire de zéro (supprimer d'abord le dossier) :

```bash
uv run python ingest.py
```

> L'indexation est volontairement lente (batchs + pauses) pour respecter le
> quota de l'API d'embeddings Gemini.

### Poser une question

```bash
uv run python main.py "Qui est Thomas ?"
```

Exemple de sortie :

```
Question : Qui est Thomas ?

Réponse  : Thomas, auparavant Stephen, est un ancien Blocard du groupe A et
un des créateurs du Labyrinthe...

Sources (wiki Fandom FR) :
  - Thomas (cosinus=0.741) — https://mazerunner.fandom.com/fr/wiki/Thomas
  ...
```

Pour une question hors corpus (« Quelle est la capitale de la France ? »), le
système renvoie : *« Je ne dispose pas d'informations suffisantes dans le wiki
pour répondre à cette question. »*

### Utilisation programmatique

```python
from src.rag import RagPipeline

pipeline = RagPipeline()                 # connexion lecture seule à la base
res = pipeline.answer("Qui est Newt ?")
print(res["answer"], res["grounded"], res["sources"])
```

---

## 6. Configuration

Tous les paramètres sont dans `src/config.py` :

| Paramètre | Rôle | Défaut |
|---|---|---|
| `EMBEDDING_MODEL` | Modèle d'embedding | `models/gemini-embedding-001` |
| `LLM_MODEL` | Modèle de génération | `gemini-2.5-flash` |
| `RETRIEVER_K` | Nombre de chunks récupérés | `4` |
| `SIMILARITY_THRESHOLD` | Seuil cosinus minimal | `0.5` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Découpage | `1000` / `200` |
| `FALLBACK_ANSWER` | Message de repli | *(voir fichier)* |

Le prompt est isolé dans `src/prompts.py` pour être modifié facilement.

---

## 7. Gestion des hallucinations

Deux mécanismes complémentaires :

1. **Filtre par score cosinus** — calculé à partir des vecteurs réellement
   stockés (et non du score de distance brut de Chroma), pour que le seuil de
   `0.5` ait une signification stable. Si aucun chunk ne dépasse le seuil, le
   LLM n'est pas appelé.
2. **Réponse de repli** + **prompt contraint** — le prompt interdit au LLM
   d'utiliser ses connaissances générales et lui impose de renvoyer le message
   de repli si la réponse n'est pas dans le contexte (seconde barrière).

---

## 8. Dépannage

| Symptôme | Cause | Solution |
|---|---|---|
| `module 'langchain' has no attribute 'verbose'` | Versions LangChain incohérentes dans l'environnement | Repartir d'un environnement propre : `uv sync` |
| Crash à l'import de `langchain_text_splitters` (Keras 3 / transformers) | Conflit TensorFlow/Keras | Géré automatiquement (`USE_TF=0` dans `loader.py`) |
| `400 API_KEY_INVALID` | Clé absente ou placeholder dans `.env` | Renseigner une vraie clé `GOOGLE_API_KEY` |
| `429 RESOURCE_EXHAUSTED` (`limit: 0`) | Quota du modèle épuisé / indisponible | Changer `LLM_MODEL` dans `src/config.py` (ex. `gemini-2.5-flash`) |
| `FileNotFoundError: Base Chroma introuvable` | Base non construite | Lancer `python ingest.py` |

---

## 9. Suivi du projet

Toutes les décisions et modifications sont consignées dans **`suivi.md`**
(journal versionné). Toute évolution du code doit y être ajoutée.

---

## 10. Étapes suivantes (TODO)

- [ ] **Application Streamlit** au-dessus de `RagPipeline`.
- [ ] **Évaluation du RAG** : jeu de questions/réponses de référence + métriques
  (pertinence des passages, fidélité des réponses, taux de repli correct).
- [ ] Décider de **versionner ou non** `chroma_maze_runner/`.
- [ ] (Bonus) **Déploiement** de l'application.

---

## 11. Modèles utilisés

| Rôle | Modèle |
|---|---|
| Embeddings | `models/gemini-embedding-001` |
| LLM | `gemini-2.5-flash` |

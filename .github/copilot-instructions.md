# GitHub Copilot Instructions — RAG Maze Runner (LO17/AI31 UTC)

## Contexte du projet

Système de **RAG (Retrieval Augmented Generation)** qui répond en **français** à
des questions sur l'univers du **Labyrinthe (Maze Runner)**, à partir du wiki
Fandom FR (`mazerunner.fandom.com/fr`). Stack : **Python 3.11**, **LangChain**,
**Google Gemini** (`gemini-2.5-flash`), **ChromaDB**, **FlashRank**, **Streamlit**.

---

## Commandes de développement

```bash
uv sync --group dev              # installe tout (prod + dev)
uv run python main.py "..."      # question CLI (pipeline avancé)
uv run python ingest.py          # (re)construire l'index parent-enfant
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv run mypy src/                 # vérification de types
uv run pytest                    # tests unitaires
uv run pytest -m integration     # tests d'intégration (nécessite .env + index)
uv run pytest --cov=src          # tests + couverture
```

---

## Architecture

### Pipeline RAG avancé (6 étapes)

```
Question
  │
  ▼
[Multi-Query]  src/retrieval.py  ── 3 reformulations FR via LLM
  │
  ▼
[RAG-Fusion]   src/retrieval.py  ── recherche enfants → parents → RRF (top 15)
  │
  ▼
[Re-Ranking]   src/retrieval.py  ── FlashRank local (top 15 → top 4)
  │
  ▼
[CRAG]         src/rag.py        ── PERTINENT | AMBIGU | HORS-SUJET
  │
  ▼
[Génération]   src/generator.py  ── Gemini, prompt contraint
  │
  ▼
[Self-RAG]     src/rag.py        ── auto-évaluation + correction si besoin
  │
  ▼
Réponse + sources
```

### Modules `src/`

| Module | Rôle |
|---|---|
| `config.py` | **Source unique de vérité** pour tous les paramètres |
| `prompts.py` | Tous les prompts FR (QA, multi-query, CRAG, Self-RAG, correction) |
| `loader.py` | Scraping wiki via API MediaWiki + nettoyage BeautifulSoup |
| `vectorstore.py` | Index Chroma enfants + docstore parents ; rotation clés Gemini |
| `retrieval.py` | Multi-Query, RAG-Fusion (RRF), Re-Ranking (FlashRank) |
| `generator.py` | Chaînes LLM LCEL (génération, graders CRAG/Self-RAG) |
| `rag.py` | `RagPipeline` — orchestre les 6 étapes |
| `logging_config.py` | `setup_logging()` — configuration centralisée des logs |

### Index persistés

| Répertoire | Contenu |
|---|---|
| `chroma_children/` | Petits chunks « enfants » (~250 car.) avec embeddings Gemini |
| `parent_docstore/` | Documents « parents » (~1500 car.) via LocalFileStore |
| `chroma_maze_runner/` | Pipeline simple legacy (323 chunks 1000 car.) |

---

## Conventions de code (à respecter impérativement)

### Typage Python 3.11 — 100 % strict

- **Toute fonction doit avoir des annotations de types complètes** (paramètres + retour).
- Types LangChain : utiliser `Runnable[Input, Output]` depuis `langchain_core.runnables`.
- Imports de types uniquement au runtime → blocs `TYPE_CHECKING` ou `TCH` ruff.
- Vérifier avec `uv run mypy src/` avant tout commit.

```python
# BON
from langchain_core.runnables import Runnable

def generate_query_variants(
    question: str,
    multiquery_chain: Runnable[dict[str, str | int], str],
    num_queries: int = 3,
) -> list[str]: ...

# MAUVAIS
def generate_query_variants(question, multiquery_chain, num_queries=3): ...
```

### Logging — convention stricte

- **Jamais de `print()` dans `src/`** — utiliser `logging.getLogger(__name__)`.
- Seule la sortie finale utilisateur dans `main.py` conserve `print()`.
- Format : `%(asctime)s [%(levelname)s] %(name)s | %(message)s`
- Chaque fonction importante : un log d'entrée (`INFO`) + un log de résultat (`INFO`).
- Détails internes : `DEBUG`. Cas limites/replis : `WARNING`. Exceptions : `ERROR`.
- **Préfixes d'étape obligatoires** dans les messages :

```python
logger.info("[Multi-Query] %d variante(s) générée(s)", n)
logger.info("[RAG-Fusion] %d parents après RRF", n)
logger.info("[Re-Ranking] top %d/%d retenus (FlashRank)", top_n, total)
logger.info("[CRAG] Statut: %s", status)
logger.info("[Génération] réponse produite")
logger.info("[Self-RAG] Validation: %s", verdict)
logger.info("[Ingestion] %d documents chargés", n)
logger.info("[Vectorstore] %d vecteurs stockés", n)
logger.info("[Loader] OK   : %s (%d car.)", title, n)
logger.warning("[Embeddings] Quota 429 — rotation vers la clé %d/%d", i, total)
```

- **Ne jamais utiliser `%` dans les messages de log** — passer les args séparément :
  ```python
  # BON
  logger.info("[CRAG] Statut: %s", status)
  # MAUVAIS
  logger.info(f"[CRAG] Statut: {status}")
  ```

### Configuration centralisée

- **Tout paramètre réglable** vit dans `src/config.py`.
- Ne jamais coder en dur un modèle, une URL, un seuil ou une taille de batch hors `config.py`.
- Les interrupteurs `USE_MULTIQUERY`, `USE_RERANK`, `USE_CRAG`, `USE_SELF_RAG`
  permettent l'évaluation comparative — ne pas les supprimer.

### Gestion des hallucinations

Le pipeline anti-hallucination est en 3 couches :

1. **CRAG** — grader LLM (temp 0) qui classe le contexte ; HORS-SUJET → repli immédiat.
2. **Prompt contraint** — interdit les connaissances hors-contexte, impose le `FALLBACK_ANSWER`.
3. **Self-RAG** — auto-évaluation + correction après génération.

Ne pas contourner ces mécanismes lors d'ajouts de fonctionnalités.

### Tests

- Tous les tests vivent dans `tests/`.
- Marqueurs : `@pytest.mark.unit` (pas d'API) / `@pytest.mark.integration`.
- Mocker `src.generator.get_llm` et `src.vectorstore.get_embeddings` pour isoler la logique.
- Tester en priorité les fonctions **pures** (pas d'I/O) : `reciprocal_rank_fusion`,
  `_split_parent_child`, `filter_documents`, `_page_name_from_url`, `generate_query_variants`.

### Ruff

- Lancer `uv run ruff check src/ tests/` avant commit.
- `uv run ruff format src/ tests/` pour formater.
- Pas de `# noqa` sans commentaire expliquant pourquoi.

---

## Patterns courants

### Ajouter une étape au pipeline

1. Ajouter les paramètres dans `src/config.py`.
2. Ajouter le prompt dans `src/prompts.py` si LLM.
3. Implémenter la logique dans le module approprié (`retrieval.py` pour récupération,
   `generator.py` pour LLM, `rag.py` pour l'orchestration).
4. Ajouter un interrupteur `USE_<STEP>` dans `config.py` et dans `RagPipeline.__init__`.
5. Logger toutes les étapes avec le préfixe adéquat.
6. Écrire les tests unitaires correspondants.
7. Documenter dans `suivi.md`.

### Accéder au pipeline en Python

```python
from src.rag import RagPipeline

pipeline = RagPipeline()
result = pipeline.answer("Qui est Newt ?")
# result = {
#     "answer": str,
#     "grounded": bool,
#     "crag_status": "PERTINENT" | "AMBIGU" | "HORS-SUJET",
#     "sources": [{"title": str, "source": str, "score": float | None}],
#     "queries": list[str],
#     "self_rag": "OK" | "A_CORRIGER" | "désactivé" | "n/a",
# }
```

### Ajouter des pages au corpus

Ajouter les URLs dans `_WIKI_PATHS` dans `src/config.py`, puis reconstruire l'index :
```bash
rm -rf chroma_children/ parent_docstore/
uv run python ingest.py
```

---

## À NE PAS FAIRE

- Ne jamais committer `.env` (contient la clé API).
- Ne jamais utiliser `print()` dans `src/` (hors `main.py`).
- Ne jamais coder en dur un modèle ou un seuil hors `config.py`.
- Ne jamais contourner les étapes CRAG/Self-RAG sans raison documentée dans `suivi.md`.
- Ne jamais lancer `build_vectorstore` depuis la couche requête (lecture seule uniquement).
- Ne jamais committer sans passer ruff + mypy.

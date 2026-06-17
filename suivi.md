# Suivi du projet RAG — LO17 (UTC)

Journal de **toutes les modifications et choix de conception** du projet.
Ce fichier est **suivi par git** et fait partie du livrable.

## Comment ajouter une entrée

- Ajouter la nouvelle entrée **en haut** de la section « Journal » (ordre antéchronologique).
- Format :

```
### AAAA-MM-JJ — Titre court

**Quoi** : ce qui a été modifié/ajouté/supprimé (fichiers concernés).
**Pourquoi** : la raison, les contraintes, les alternatives écartées.
```

---

## Journal

### 2026-06-17 — Évaluation complète en local (Ollama) + compare_hybrid + préparation démo

**Quoi** : `tests/evaluation/compare_hybrid.py` *(nouveau)*, résultats dans
`tests/evaluation/results/` (`eval_latest.json`, `compare_hybrid_latest.json`).

- **compare_hybrid.py** : script CLI comparatif dense seul vs hybride BM25 sur
  le jeu de 24 questions. Lance le pipeline deux fois, affiche un tableau ✅/⚠️
  et sauvegarde le JSON horodaté.
- **run_eval — résultats (llama3.2:3b, EMBEDDING_PROVIDER=ollama, 24 questions)**

  | Métrique | Score |
  |---|---|
  | Exactitude décision réponse | **95,8 %** (23/24) |
  | Taux de repli correct | **80,0 %** (4/5 hors-sujet) |
  | Accord CRAG | **91,7 %** |
  | Couverture mots-clés (moy.) | 36,8 % |
  | Latence (CPU, sans GPU) | ~111 s/question |

- **compare_hybrid — dense seul vs hybride BM25**

  | Métrique | Dense | Hybride | Δ |
  |---|---|---|---|
  | Exactitude décision réponse | 91,7 % | **95,8 %** | **+4,2 %** ✅ |
  | Taux de repli correct | 60,0 % | **80,0 %** | **+20,0 %** ✅ |
  | Accord CRAG | 91,7 % | **95,8 %** | **+4,2 %** ✅ |
  | Couverture mots-clés | **47,4 %** | 36,8 % | -10,5 % ⚠️ |
  | Latence | 117,3 s | 121,4 s | +4,1 s ⚠️ |

- **RAGAS local** : `llama3.2:3b` trop petit pour produire du JSON structuré
  fiable → échec. Scores de référence du PoC `main` :
  `faithfulness=0.45, answer_relevancy=0.73, precision=0.81, recall=0.50`.
- **Infrastructure Ollama** : modèle `mistral` (7B) OOM-killed (11,7 Go RSS sur
  15 Go host) → migration vers `llama3.2:3b` (2 Go). Index `chroma_children_ollama/`
  construit (1260 enfants, 201 parents).

**Pourquoi** : valider empiriquement l'apport du BM25 hybride. `USE_HYBRID=True`
confirmé comme défaut justifié (repli +20 %, décision +4,2 %).

### 2026-06-16 — Recherche hybride BM25 + RRF (axe 8) + interrupteur d'ablation

**Quoi** : `src/retrieval.py`, `src/config.py`, `src/rag.py`, `streamlit_app.py`,
`tests/evaluation/run_eval.py`, `tests/unit/test_retrieval.py`, `pyproject.toml`.

- **BM25 lexical fusionné à la RRF** : `bm25_rank_parents()` applique BM25
  (backend `rank-bm25`) sur les **chunks enfants** (tokenizer FR
  minuscules/alphanum, `_tokenize_fr`), puis remonte aux parents (dédup), comme
  la voie dense. Le classement lexical est **injecté comme liste supplémentaire**
  dans la RRF existante de `fusion_retrieve` (flag `use_hybrid`). Index BM25 mis
  en cache par base d'enfants. Config : `USE_HYBRID=True`, `BM25_K=10`.
- **Choix BM25 sur enfants (et non parents)** : les parents (~1500 car.) diluent
  les termes ; les enfants (~250 car.) sont plus discriminants, et la remontée
  enfant→parent réutilise le pattern parent-document.
- **Interrupteur d'ablation** : `RagPipeline(use_hybrid)`, checkbox Streamlit,
  `run_eval --no-hybrid` → on peut **mesurer** dense seul vs hybride.
- **Tests** : 4 tests `bm25_rank_parents` (faux child-vectorstore + `InMemoryStore`,
  hors réseau) : appariement nom propre, insensibilité à la casse, remontée
  enfant→parent, base vide. Total **63 → 67 tests**, mypy strict 0 erreur, ruff clean.

**Pourquoi & honnêteté sur le gain** : le corpus est dense en noms propres, mais
un *smoke test* sur l'index réel (1260 enfants) montre que BM25 **seul** est
**bruité ici** : les concepts clés (Braise, WICKED, Griffeur) sont **ubiquitaires**
(cités sur presque toutes les pages perso) donc peu discriminants lexicalement,
et les pages longues (beaucoup d'enfants) sont sur-représentées (« Newt » remonte
souvent en tête). BM25 reste néanmoins un **second signal de rappel** fusionné à
la voie dense, et le Re-Ranking final (FlashRank) filtre le top 15 → 4.
⚠️ **À mesurer avant de figer le défaut** : lancer `run_eval` avec puis sans
`--no-hybrid` (idéalement en `--provider ollama`, hors quota) et comparer
couverture mots-clés / accord CRAG. Si l'hybride n'apporte rien sur ce corpus,
basculer `USE_HYBRID=False`. La décision est désormais **outillée** (ablation +
métriques), conformément à la démarche d'évaluation (axe 6).

### 2026-06-16 — Dockerisation (compose Streamlit + Ollama) + index paramétré par provider (axe 11)

**Quoi** : `Dockerfile` *(nouveau)*, `docker-compose.yml` *(nouveau)*,
`.dockerignore` *(nouveau)*, `src/config.py`, `src/vectorstore.py`,
`README.md`, `.env.example` ; `chroma_children/` → `chroma_children_gemini/`
*(git mv)*.

- **Index paramétré par provider d'embedding** : `CHILD_CHROMA_DIR =
  f"./chroma_children_{EMBEDDING_PROVIDER}"`. L'index Gemini existant est
  renommé `chroma_children_gemini/` (git mv, données préservées) ; un futur
  index local ira dans `chroma_children_ollama/`. Le `parent_docstore/`
  (texte brut, indépendant du provider) reste **partagé**. On peut ainsi
  conserver l'index cloud ET un index offline sans que l'un écrase l'autre.
- **Embeddings Ollama** : `get_embeddings()` gère `EMBEDDING_PROVIDER=ollama`
  (`OllamaEmbeddings(nomic-embed-text)`) → retrieval 100 % local possible.
- **Dockerfile** : image Streamlit `python:3.11-slim` + uv (couche deps mise en
  cache séparément du code). **LLM (Ollama) et index restent hors de l'image**
  (service séparé + volumes) → image légère, livraison reproductible.
- **docker-compose.yml** : 2 services (`ollama` officiel + `rag-app`) avec
  `healthcheck`, volumes (`ollama_data`, index montés, cache FlashRank),
  `OLLAMA_BASE_URL=http://ollama:11434`. Service one-shot `ollama-init`
  (profil `bootstrap`) pour tirer les modèles. `env_file` marqué
  `required: false` → `docker compose` ne casse pas en l'absence de `.env`
  (mode local sans clé). Bloc GPU NVIDIA commenté (optionnel, WSL). Validé via
  `docker compose config`.
- **README** : tableau des 3 modes (A API Gemini / B LLM local / C 100 % offline)
  + mode Docker, avec les commandes exactes.

**Pourquoi** : portabilité « clone → `docker compose up` ». Décisions :
- **LLM/index hors image** : éviter une image de plusieurs Go (modèles) et
  garder l'index versionné modifiable sans rebuild.
- **Garder Gemini + local en parallèle** (choix utilisateur) plutôt que de
  basculer tout en local : la réindexation locale est rapide (pas de pauses
  anti-quota), mais conserver l'index Gemini préserve la version « qualité
  cloud » pour la démo. Le paramétrage par dossier rend la bascule triviale
  (`EMBEDDING_PROVIDER`), sans migration ni perte.

### 2026-06-16 — LLM locaux (Ollama) sélectionnables + évaluation hors quota (axes 9/10)

**Quoi** : `src/config.py`, `src/generator.py`, `src/rag.py`, `streamlit_app.py`,
`tests/evaluation/run_eval.py`, `tests/evaluation/run_ragas.py`,
`tests/unit/test_generator.py` *(nouveau)*, `pyproject.toml`, `.env.example`.

- **Provider Ollama** : `LLM_PROVIDER=ollama` ajouté. `config.py` :
  `OLLAMA_MODEL` (défaut `mistral`), `OLLAMA_BASE_URL`, `OLLAMA_EMBED_MODEL`
  (`nomic-embed-text`), `LOCAL_MODELS` (liste du sélecteur, surchargée par env).
  `generator.get_llm(model, temperature, provider)` : nouveau paramètre
  `provider` (override à l'exécution) + branche `ChatOllama` (import paresseux,
  `ImportError` explicite si `langchain-ollama` absent).
- **Threading provider/modèle** : `build_*_chain(provider, model)` et
  `RagPipeline(..., provider, model)` propagent le choix à toutes les chaînes
  (génération, multi-query, CRAG, Self-RAG, correction). Choix d'API : passer
  `provider/model` aux *builders* (et non des instances LLM partagées) — garde
  les tests `RagPipeline` inchangés (ils patchent déjà `build_*`) et évite à
  `rag.py` d'appeler `get_llm` directement.
- **Streamlit (axe 10)** : sélecteur de modèle en sidebar
  (`_model_choices()` → Gemini/Groq affichés si clé présente, modèles locaux
  toujours listés). Cache `@st.cache_resource` désormais clé sur
  `(provider, model, 4 interrupteurs)`. Gestion d'erreur : `ImportError`
  (paquet manquant) et toute exception de génération (Ollama éteint) →
  message clair avec rappel `ollama serve` / `ollama pull`, **pas de crash**.
  Suppression de `_missing_api_key` (remplacé par le filtrage des choix).
- **Évaluation hors quota (axe 9 × 6)** : `run_eval`/`run_ragas` reçoivent
  `--provider {gemini,groq,ollama}` et `--model`, passés à `RagPipeline`. Le
  **juge RAGAS** utilise `get_llm(provider=...)` et, en mode `ollama`, des
  embeddings **`OllamaEmbeddings`** locaux → évaluation 100 % locale.

**Pourquoi** : le quota Gemini free tier (génération + juge RAGAS) était le
point bloquant pour une évaluation répétée. Déporter tous les appels LLM sur
Ollama supprime ce coût.
- **Nuance importante documentée** : les *embeddings de requête* du retrieval
  restent Gemini car l'index `chroma_children/` a été **construit avec Gemini**
  (provider+modèle d'embedding doivent être identiques entre indexation et
  requête). Passer à des embeddings locaux pour le retrieval imposerait de
  reconstruire l'index — hors périmètre. Donc en `--provider ollama` : LLM 100 %
  local, seuls subsistent ~4 embeddings de requête Gemini par question (quota
  *embeddings* séparé et léger, avec rotation de clés sur 429).
- **Modèles recommandés** : `mistral` (7B, ~4.4 Go) et `gemma3:4b` (~3.3 Go)
  pour la démo ; `llama3.1:8b` conseillé comme **juge** RAGAS (plus robuste).
- **Tests** : `test_generator.py` (3) verrouille le routage de provider
  (Ollama, défaut, fallback) hors-ligne. Total **60 → 63 tests**, mypy strict
  0 erreur, ruff clean. `langchain-ollama` ajouté en dépendance prod +
  override mypy `langchain_ollama.*`.

### 2026-06-16 — Évaluation du RAG (axe 6) : jeu de référence + RAGAS porté sur `RagPipeline`

**Quoi** : `tests/evaluation/` *(nouveau)* — `eval_dataset.json`, `dataset.py`,
`run_eval.py`, `run_ragas.py`, `__init__.py`, `results/.gitignore` ;
`tests/unit/test_rag.py` *(nouveau)*, `tests/unit/test_eval_dataset.py` *(nouveau)* ;
`src/rag.py`, `pyproject.toml`.

- **`src/rag.py`** : ajout du champ `contexts: list[str]` au `TypedDict`
  `RagAnswer` (page_content des documents rerankés réellement fournis au
  générateur ; `[]` en repli). Prérequis indispensable à RAGAS, qui exige
  `retrieved_contexts` — jusqu'ici seules les métadonnées `sources`
  (titre/url/score) étaient exposées, pas le texte du contexte.
- **`eval_dataset.json`** : 24 questions FR de référence (19 du domaine avec
  `reference` + `expected_keywords` + `expected_crag_status=PERTINENT`, 5
  hors-sujet `should_answer=false` / `HORS-SUJET`). Étend la PoC de 4 questions.
- **`run_eval.py`** (évaluation comportementale, **sans juge LLM**) : exécute le
  *vrai* `RagPipeline` et mesure taux de repli correct, exactitude de décision
  réponse/repli, accord CRAG, couverture des mots-clés et latence. Résultats
  versionnés (`results/*_latest.json`). Interrupteurs `--no-multiquery/--no-rerank/
  --no-crag/--no-self-rag` pour les études d'ablation (axe 8).
- **`run_ragas.py`** (les 4 métriques de l'énoncé : Faithfulness,
  ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall) :
  **portage** de la PoC trouvée sur `origin/main` (cellules 27-34 de `rag.ipynb`).

**Pourquoi** : la PoC RAGAS existante (a) vivait uniquement dans un notebook sur
`origin/main` (historique git indépendant — absente de notre branche), (b)
évaluait une chaîne *naïve* Ollama (sans RAG-Fusion/Re-Ranking/CRAG/Self-RAG),
donc ses scores (`faithfulness=0.45`…) ne disaient rien de notre pipeline, et
(c) dépendait d'un juge `llama3.1:8b` non reproductible chez nous.
Décisions du portage :
- **Juge = `get_llm()` + embeddings = `get_embeddings()`** (Gemini/Groq, nos
  providers existants) au lieu d'Ollama → reproductible sans dépendance serveur
  locale supplémentaire.
- **Évaluation du contexte réellement généré** (`RagAnswer["contexts"]`) plutôt
  qu'un retriever séparé → la fidélité/précision mesurent bien *notre* pipeline.
- **RAGAS = dépendance optionnelle** (groupe `eval`, `uv sync --group eval`)
  car lourde ; import paresseux sous `try/except` avec message clair si absente,
  + shim `ChatVertexAI` pour contourner un import cassé de certaines versions
  RAGAS. `override` mypy `ragas.*` ajouté.
- **Deux niveaux** : `run_eval` (bon marché, déterministe, mesure le repli
  anti-hallucination sans quota juge) complète `run_ragas` (qualité fine mais
  coûteuse en quota). `--limit` sur les deux pour les essais.
- **Tests** : `test_rag.py` (9 tests `RagPipeline` mockés — combinaisons CRAG
  PERTINENT/AMBIGU/HORS-SUJET, repli sans génération, correction Self-RAG,
  exposition `contexts`) comble le trou « aucun test sur l'orchestrateur » ;
  `test_eval_dataset.py` (6 tests) garantit l'intégrité du jeu sans réseau.
  Total **45 → 60 tests unitaires**, mypy strict 0 erreur, ruff clean.

### 2026-06-16 — Application Streamlit + typage 100 % mypy strict

**Quoi** : `streamlit_app.py` *(nouveau)*, `src/rag.py`, `src/generator.py`,
`src/retrieval.py`, `src/vectorstore.py`, `pyproject.toml`, `README.md`,
`CLAUDE.md`

- **`streamlit_app.py`** : interface chat (`st.chat_input`/`st.chat_message`)
  au-dessus de `RagPipeline`. Sidebar avec interrupteurs Multi-Query/
  Re-Ranking/CRAG/Self-RAG (évaluation comparative en direct), affichage des
  statuts CRAG/Self-RAG, sources en `st.expander`, reformulations en option.
  Pipeline mis en cache via `@st.cache_resource` (clé = combinaison des 4
  interrupteurs) pour ne jamais rouvrir l'index à chaque message.
  Réécrit **from scratch** plutôt que repris du `app.py` trouvé sur la branche
  `main` (historique git indépendant de `test`, sans ancêtre commun) : ce
  dernier dupliquait une logique RAG naïve avec Ollama, sans réutiliser notre
  pipeline avancé (Multi-Query/RAG-Fusion/Re-Ranking/CRAG/Self-RAG) — repartir
  de `RagPipeline` était plus cohérent que d'adapter du code incompatible.
- **`src/rag.py`** : ajout des `TypedDict` `SourceInfo` et `RagAnswer` —
  remplace les `dict` non paramétrés en retour de `RagPipeline.answer`,
  `_fallback`, `_sources`, `answer_question`.
- **Typage 100 % mypy strict (0 erreur sur `src/`)** :
  - `pyproject.toml` : ajout de `plugins = ["pydantic.mypy"]` (sans ce plugin,
    mypy ne reconnaît pas les champs pydantic des classes LangChain comme
    kwargs valides du `__init__` — ex. `GoogleGenerativeAIEmbeddings(google_api_key=...)`).
  - `src/generator.py` : type alias `StrChain = Runnable[dict[str, Any], str]`
    appliqué à `get_llm`, `format_docs`, toutes les `build_*_chain`,
    `build_rag_chain`. Correctif `ChatGroq(model_name=...)` (le champ pydantic
    réel est `model_name`, alias `model` non reconnu par le plugin mypy).
  - `src/retrieval.py` : signatures `Chroma`/`BaseStore[str, Document]`/
    `StrChain` explicites. `FlashrankRerank(...)` marqué
    `# type: ignore[call-arg]` (le champ `client` est déclaré obligatoire
    mais construit dynamiquement par un validateur pydantic `mode="before"`).
  - `src/vectorstore.py` : `_call_with_rotation` retype en
    `list[list[float]] | list[float]` (+ `cast` ciblés dans `embed_documents`/
    `embed_query`) au lieu d'un retour `Any` implicite ; `get_parent_docstore`
    et `get_parent_document_retriever` typés avec `BaseStore[str, Document]`.

**Pourquoi** : le sujet exige une application Streamlit ; la version trouvée
sur `main` n'était pas réutilisable sans régresser sur les anti-hallucinations
(CRAG/Self-RAG absents de cette version). Le typage strict était l'objectif
explicite de la session précédente (tooling mypy/ruff/pytest) — il restait
26 erreurs mypy non résolues qu'il fallait traiter avant de considérer le
code « 100 % typé ».

**Validation** : `uv run mypy src/ streamlit_app.py` → 0 erreur.
`uv run ruff check` → clean. `uv run pytest -m unit` → 45/45.
`uv run streamlit run streamlit_app.py` démarre sans exception (HTTP 200) ;
le flux question→réponse complet n'a **pas** pu être testé dans un navigateur
faute de clé API (`.env` absent dans cet environnement) — à vérifier par un
membre de l'équipe disposant d'une clé Gemini valide.

### 2026-06-13 — Flag --force pour reconstruction de l'index dans ingest.py

**Quoi** : `ingest.py`

- Ajout du flag `--force` / `-f` via `argparse` : si l'index existe et que le
  flag est passé, les dossiers `chroma_children/` et `parent_docstore/` sont
  supprimés (`shutil.rmtree`) avant reconstruction.
- Sans `--force`, le comportement précédent est conservé (skip + message).
- Import de `CHILD_CHROMA_DIR` et `PARENT_DOCSTORE_DIR` depuis `src/config.py`.

**Pourquoi** : la suppression manuelle des dossiers était fastidieuse.
Alternative écartée : suppression automatique sans flag — trop destructive par
défaut (quota Gemini consommé sans avertissement).

**Choix de versionnement des stores** : `chroma_children/`, `parent_docstore/`
et `chroma_maze_runner/` sont **versionnés** (non ignorés) afin de livrer un
index fonctionnel sans avoir à relancer `ingest.py` (coûteux en quota). Le
dépôt est donc plus lourd mais directement utilisable après `git clone`.

---

### 2026-06-13 — Tooling complet : ruff, mypy strict, pytest, copilot instructions

**Quoi** : `pyproject.toml`, `.gitignore`, `.github/copilot-instructions.md`,
`CLAUDE.md`, `src/loader.py`, `src/rag.py`, `src/retrieval.py`,
`src/vectorstore.py`, `tests/` *(nouveau)*

- **`pyproject.toml`** : ajout du groupe de dépendances dev (`mypy>=1.10`,
  `ruff>=0.6`, `pytest>=8.0`, `pytest-cov>=5.0`, `pytest-mock>=3.14`,
  `types-requests`, `types-beautifulsoup4`) + configurations `[tool.mypy]`
  (strict), `[tool.ruff]` (lint + format), `[tool.pytest.ini_options]`
  (marqueurs `unit`/`integration`, testpaths), `[tool.coverage]`.
- **`.gitignore`** : `.github/` retiré — `copilot-instructions.md` doit être
  versionné pour être lu par GitHub Copilot.
- **`.github/copilot-instructions.md`** : réécriture complète (architecture
  du pipeline, commandes dev, conventions typage/logging/tests, patterns
  courants, liste des choses à ne pas faire).
- **`CLAUDE.md`** : refonte — expertise par domaine (LangChain, Gemini, Chroma,
  FlashRank, Streamlit, patterns RAG avancés), stratégie de debug économe en
  tokens (4 niveaux), conventions typage strict Python 3.11, logs normés,
  structure de tests obligatoire.
- **Corrections ruff** : `zip(strict=False)` dans `retrieval.py` (B905),
  `rsplit(maxsplit=1)` dans `loader.py` (PLC0207), singleton sans `global`
  dans `rag.py` (PLW0603), suppression noqa obsolète dans `vectorstore.py`.
- **`tests/`** : 45 tests unitaires (`test_loader.py`, `test_retrieval.py`,
  `test_vectorstore.py`) + `conftest.py` avec fixtures mocks (LLM, embeddings,
  documents factices). Marqueurs `@pytest.mark.unit` / `@pytest.mark.integration`.

**Pourquoi** : outillage qualité demandé — mypy strict pour la détection
précoce d'erreurs de type, ruff pour la cohérence du style, pytest avec mocks
pour tester la logique pure sans appel API. Les tests unitaires permettent
d'isoler rapidement les régressions lors des ajouts futurs (Streamlit, éval).

### 2026-06-11 — Support multi-provider LLM (Gemini / Groq) + système de logs

**Quoi** : `src/config.py`, `src/generator.py`, `src/logging_config.py` *(nouveau)*,
`src/rag.py`, `src/retrieval.py`, `src/loader.py`, `src/vectorstore.py`,
`main.py`, `ingest.py`, `.env.example`, `pyproject.toml` + `uv.lock`

- **LLM provider** (`src/config.py`) : ajout de `LLM_PROVIDER` (`"gemini"` |
  `"groq"`), `GROQ_API_KEY`, `GROQ_MODEL` (défaut `llama3-8b-8192`).
- **`src/generator.py`** : `get_llm()` rendu agnostique — instancie
  `ChatGoogleGenerativeAI` ou `ChatGroq` selon `LLM_PROVIDER` ; import lazy de
  `langchain_groq` avec message d'erreur clair si absent.
- **`src/logging_config.py`** : nouveau module, `setup_logging(level)` configure
  le logger racine (format : `AAAA-MM-JJ HH:MM:SS [LEVEL] nom.module | message`,
  bruit des libs tierces réduit au niveau WARNING).
- **Migration logging** : tous les `print()` internes des modules `src/` migrés
  vers `logger.info()` / `logger.warning()` / `logger.debug()`. Règle : seule
  la sortie finale destinée à l'utilisateur dans `main.py` conserve `print()`.
  `src/rag.py._log()` route sur `logger.info()` (verbose=True) ou
  `logger.debug()` (verbose=False).
- **`main.py` + `ingest.py`** : appel `setup_logging()` au démarrage ; vérification
  de clé adaptée selon `LLM_PROVIDER` / `EMBEDDING_PROVIDER`.
- **`langchain-groq==1.1.3`** ajouté via `uv add langchain-groq`.

**Pourquoi** : (1) flexibilité infrastructure — pouvoir basculer Gemini ↔ Groq
via une variable d'env sans modifier le code ; (2) conformité à la convention
de logs CLAUDE.md (module `logging`, pas de `print()` dans `src/`).

**Alternatives écartées** : support Ollama (local, mais trop lourd pour
déploiement) ; `langchain-openai` (payant, hors périmètre).

### 2026-06-11 — Rotation de clés API et alternatives d'embedding (quota 429 journalier)

**Quoi** : `src/config.py`, `src/vectorstore.py`, `.env.example`
- `src/config.py` : ajout de `GOOGLE_API_KEYS` (lit `GOOGLE_API_KEY` +
  `GOOGLE_API_KEY_2/3/4`), `EMBEDDING_PROVIDER` (`"gemini"` | `"huggingface"`),
  `HF_EMBEDDING_MODEL` (modèle sentence-transformers à utiliser en local).
- `src/vectorstore.py` : nouvelle classe `RotatingGeminiEmbeddings` (implémente
  `langchain_core.embeddings.Embeddings`) qui bascule vers la clé suivante sur
  `429 RESOURCE_EXHAUSTED` ; `get_embeddings()` mis à jour pour gérer les trois
  cas : Gemini mono-clé (comportement inchangé), Gemini multi-clés (rotation),
  HuggingFace local (aucun quota).
- `.env.example` : documentation des nouvelles variables.

**Pourquoi** : quota free tier atteint (`embed_content_free_tier_requests`,
1 000 req./jour/clé). La rotation permet d'utiliser plusieurs comptes Google
pour multiplier le quota. La voie HuggingFace (`langchain-huggingface` +
`sentence-transformers`) est offerte comme alternative sans quota, locale.

**Alternatives écartées** : OpenAI / Cohere (autres API payantes, hors scope) ;
attendre la réinitialisation journalière (bloquant pour l'ingestion complète).

**Remarque** : changer de provider ou de modèle d'embedding impose de
reconstruire l'index (`chroma_children/` + `parent_docstore/`).

### 2026-06-11 — Correctif quota embeddings à l'ingestion (429 RESOURCE_EXHAUSTED)

**Blocage rencontré**
`uv run python ingest.py` échouait sur `429 RESOURCE_EXHAUSTED` —
`embed_content_free_tier_requests, limit: 100` (soit **100 requêtes
d'embedding par minute** en free tier).

**Cause**
La 1re version de `build_parent_document_index` regroupait par **documents**
(10 docs/lot) et déléguait à `ParentDocumentRetriever.add_documents`, qui
embarque **tous les enfants du lot d'un seul coup** : un lot de 10 documents =
plusieurs centaines d'enfants embarqués en une fois → dépassement immédiat des
100/min. (La base simple passait car elle indexait par lots de 80 *chunks* avec
pause de 65 s.)

**Correctif (`src/vectorstore.py`)**
- `_split_parent_child()` : découpage parent/enfant **local** (aucun appel
  réseau), reproduisant la logique interne du retriever (parent_id par parent,
  `metadata['doc_id']` sur chaque enfant, titre/source propagés).
- Parents rangés dans le docstore via `docstore.mset()` (aucun appel API).
- Enfants embarqués **par lots de 80 avec pauses de 65 s** (≈ requêtes/minute,
  comme la base simple qui passait) via `_add_children_with_retry()` qui
  **réessaie automatiquement** en cas de 429 (jusqu'à 5 fois).
- Logs préfixés `[Ingestion]` / `[Vectorstore]`.

**Nettoyage**
- Le run échoué avait laissé un index **partiel** (`chroma_children/` 184 K,
  `parent_docstore/` absent). Supprimé pour repartir d'une base propre.

**Validation**
- `_split_parent_child` testé sans API : 1 doc ~2760 car → 2 parents (~1494) +
  16 enfants (~243), `doc_id` cohérents, titre propagé. ✅

### 2026-06-11 — Convention de logs ajoutée à CLAUDE.md

**Quoi**
- Ajout d'une section **« Convention de logs (OBLIGATOIRE) »** dans `CLAUDE.md` :
  usage du module `logging` (logger par module), configuration centralisée
  prévue dans `src/logging_config.py` (`setup_logging`), format standard,
  préfixes d'étape FR (`[Multi-Query]`, `[CRAG]`, …), niveaux INFO/DEBUG/
  WARNING/ERROR, et règle « un log d'entrée + un log de résultat par fonction
  importante ».

**Pourquoi**
- Demande de l'utilisateur : garantir que tout code créé soit facilement
  traçable. Centralise une règle claire que je suivrai pour le code futur.

**Reste à faire**
- Implémenter `src/logging_config.py` et migrer les `print` préfixés actuels du
  `RagPipeline` vers ce logger (proposé à l'utilisateur).

### 2026-06-11 — Pipeline RAG avancé (multi-représentation, RAG-Fusion, re-ranking, CRAG, Self-RAG)

**Objectif**
Faire évoluer le RAG « simple » (cosinus + seuil) vers une architecture avancée
en 6 étapes, répartie dans les modules existants + un nouveau module
`src/retrieval.py`.

**Analyse préalable de `rag.ipynb` (imposée)**
Le notebook implémente le multi-query via
`from langchain_classic.retrievers.multi_query import MultiQueryRetriever` puis
`MultiQueryRetriever.from_llm(retriever, llm)` (cellules 19-23), avec
`ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)`. Ce
retriever génère des reformulations par LLM puis fait l'**union** des résultats.
Je m'en suis inspiré pour la *génération des reformulations* (prompt FR dédié,
nombre fixe) mais j'ai remplacé l'union par une **fusion RRF** (vrai RAG-Fusion,
demandé).

**Ce qui a été fait, par composant**
1. **Indexation multi-représentation (Parent Document Retriever)** —
   `src/vectorstore.py` : enfants ~250 car. embarqués dans Chroma
   `chroma_children/` + parents ~1500 car. dans un docstore **persistant**
   `parent_docstore/` (`LocalFileStore` + `create_kv_docstore`). `src/loader.py`
   ajoute `load_clean_documents()` (docs complets, le découpage parent/enfant
   est délégué au retriever). `ingest.py` construit cet index par lots avec
   pauses (quota embeddings). Au retrieval, un enfant sélectionné fait remonter
   son parent (via `metadata['doc_id']`).
2. **Multi-Query / RAG-Fusion (RRF)** — `src/retrieval.py` :
   `generate_query_variants` (3 reformulations FR + question d'origine),
   `search_parents_for_query` (recherche enfants → parents dédupliqués),
   `reciprocal_rank_fusion` (score = Σ 1/(k+rang), k=60), `fusion_retrieve`
   (top 15).
3. **Re-Ranking** — `src/retrieval.py` : `FlashrankRerank`
   (`ms-marco-MultiBERT-L-12`, local, multilingue) ; top 15 → top 4.
4. **CRAG** — `src/rag.py` + `src/prompts.py` : un LLM (temp 0) note le contexte
   `PERTINENT | AMBIGU | HORS-SUJET`. HORS-SUJET → repli sans appeler le LLM de
   génération ; AMBIGU → avertissement injecté dans le prompt.
5. **Self-RAG** — `src/rag.py` + `src/generator.py` + `src/prompts.py` :
   auto-évaluation (fidélité + pertinence) après génération ; si `A_CORRIGER`,
   une passe de correction (`MAX_CORRECTIONS=1`) avant renvoi.
   - `src/config.py` : tous les nouveaux paramètres + interrupteurs
     `USE_MULTIQUERY/RERANK/CRAG/SELF_RAG` (utiles pour l'évaluation comparative).
   - `src/generator.py` : chaînes LCEL génération / multi-query / CRAG /
     Self-RAG / correction.
   - `main.py` : affichage enrichi (statut CRAG, Self-RAG, sources + score).
   - Logs élégants à chaque étape : `[Multi-Query]`, `[RAG-Fusion]`,
     `[Re-Ranking]`, `[CRAG] Statut: …`, `[Génération]`, `[Self-RAG] Validation: …`.

**Dépendances**
- `uv add flashrank` (→ `flashrank>=0.2.10` dans `pyproject.toml` + `uv.lock`).

**Découvertes / points techniques (env langchain 1.x)**
- Le méta-paquet `langchain` n'est pas installé : les retrievers/stores
  « classiques » vivent dans **`langchain_classic`** (`langchain_classic.retrievers`
  pour `MultiQueryRetriever` et `ParentDocumentRetriever`,
  `langchain_classic.storage` pour `LocalFileStore`/`create_kv_docstore`).
- `LocalFileStore` n'est PAS dans `langchain_core.stores` ici → importé depuis
  `langchain_classic.storage`.
- Garde-fou `USE_TF=0` déplacé dans `src/__init__.py` (s'applique avant tout
  sous-module ; `vectorstore.py` importe désormais aussi le text-splitter).

**Validation effectuée (sans l'index complet)**
- `py_compile` + chaîne d'imports complète OK (env uv).
- RRF : ordre correct (un parent présent dans 2 listes remonte 1er).
- Multi-Query : parsing robuste (puces/ lignes vides retirées, question
  d'origine en tête, déduplication).
- FlashRank : modèle multilingue téléchargé (98,7 Mo, mis en cache) ;
  reranking correct (« Qui a créé le Labyrinthe ? » → « Thomas » 0.999, doc
  hors-sujet écarté).

**Reste à faire (côté utilisateur)**
- Lancer `uv run python ingest.py` pour construire l'index parent-enfant
  (~1000+ enfants à embarquer → ~15 min + quota embeddings Gemini), puis
  `uv run python main.py "Quel est le but de WICKED ?"` pour le test end-to-end.
- L'ancienne base `chroma_maze_runner/` (pipeline simple) reste disponible pour
  comparer « RAG naïf vs avancé » dans le rapport.

---

### 2026-06-11 — Passage à uv pour la gestion des dépendances

**Quoi**
- Adoption de **uv** (0.10.8) comme gestionnaire de dépendances unique.
- Création de **`pyproject.toml`** (dépendances déclaratives, `requires-python>=3.11`) et de **`uv.lock`** (versions verrouillées, 129 paquets résolus).
- Ajout de **`streamlit>=1.40`** aux dépendances (anticipe l'app, étape suivante).
- **Suppression de `requirements.txt`** (remplacé par `pyproject.toml` + `uv.lock`).
- Recréation de l'environnement avec `uv sync` (remplace le `.venv` pip précédent).
- Mise à jour de `README.md` et `CLAUDE.md` : toutes les commandes passent désormais par `uv run` (ex. `uv run python main.py "..."`), install via `uv sync`, ajout de dépendance via `uv add`.

**Pourquoi**
- Demande explicite de l'utilisateur ; aligne aussi le projet sur le standard du projet de référence `Pasteuraize` (pyproject + uv.lock).
- `uv.lock` garantit la **reproductibilité exacte** (attendu « code source reproductible » du sujet), bien mieux que des bornes `>=` dans un `requirements.txt`. uv gère en plus la version de Python et le venv.

**Validation**
- `uv lock` puis `uv sync` OK (exit 0).
- **Test end-to-end via `uv run python main.py "Qui est Newt ?"`** : réponse correcte et sourcée (4 chunks « Newt », cosinus ~0.76–0.78). Pipeline complet fonctionnel sous l'environnement uv.

---

### 2026-06-11 — Documentation : README + .env.example

**Quoi**
- Création de **`README.md`** détaillé (objectif, attendus du sujet, architecture en 2 couches, arborescence, installation, utilisation, configuration, gestion des hallucinations, dépannage, TODO).
- Création de **`.env.example`** (modèle de configuration sans clé, **versionnable**) ; instructions pour le copier en `.env`.
- Détail accru de ce journal (déroulé chronologique + blocages, ci-dessous).

**Pourquoi**
- Le sujet demande un **code source reproductible** : un README clair est indispensable pour qu'un correcteur (ou un coéquipier) installe et lance le projet sans contexte.
- `.env.example` permet de committer la *structure* de configuration sans jamais exposer la vraie clé (le `.env` reste ignoré par git).

---

### 2026-06-11 — Migration du notebook vers `src/` + couche RAG (P2) avec anti-hallucinations

#### Objectif
Porter la logique du notebook `rag.ipynb` en modules Python réutilisables
(Étape 1) puis construire la couche de récupération/génération avec gestion des
hallucinations (Étape 2), en se branchant **en lecture seule** sur la base
Chroma déjà construite par P1.

#### Fichiers créés / modifiés
- **Réécrits** (ex-pipeline blog Gemini anglais → pipeline Maze Runner) :
  `src/config.py`, `src/loader.py`, `src/vectorstore.py`, `src/generator.py`,
  `main.py`, `requirements.txt`.
- **Nouveaux** : `src/prompts.py` (prompt FR ancré au domaine),
  `src/rag.py` (`RagPipeline`), `ingest.py` (build reproductible).

#### Déroulé chronologique détaillé & blocages rencontrés

1. **Lecture (Étape 0).** Lecture du sujet PDF, du notebook `rag.ipynb`, des
   modules `src/` existants et du projet de référence `Pasteuraize`. Constat :
   `src/` contenait encore l'ancien pipeline (blog Gemini, anglais,
   `./chroma_db`, `WebBaseLoader`, `k=1`), désynchronisé du notebook
   Maze Runner. Patterns retenus de Pasteuraize : config centralisée, prompts
   comme constantes dédiées, séparation modulaire, messages d'erreur/repli.

2. **Écriture des modules.** Migration de l'ingestion (API MediaWiki + BS4 +
   filtrage + chunking) et écriture de la couche requête. Choix de conception
   notables :
   - **Score cosinus recalculé manuellement** depuis les vecteurs stockés
     (`vectorstore._collection.query(..., include=["embeddings"])`) plutôt que
     via le score de distance brut de Chroma : la collection P1 ayant été créée
     en distance **L2** par défaut, le score de pertinence natif de LangChain
     serait mal calibré ; le cosinus recalculé garde un seuil de `0.5` lisible.
   - **Séparation `ingest.py` (build) / `main.py` (requête)** : l'Étape 2a
     impose une connexion **lecture seule** ; `load_vectorstore` ne reconstruit
     jamais la base (lève une erreur explicite si absente).

3. **Vérifs statiques.** `py_compile` OK sur tous les modules.

4. **🔴 BLOCAGE 1 — import de `langchain_text_splitters` (env local).**
   `import src.loader` plantait :
   `ValueError: Your currently installed version of Keras is Keras 3, but this
   is not yet supported in Transformers...`. **Cause** : la version installée de
   `langchain_text_splitters` importe *eagerly* l'intégration
   `sentence-transformers` → `transformers` → backend TensorFlow, cassé par
   Keras 3 sur cette machine. **Résolution** : garde-fou
   `os.environ.setdefault("USE_TF", "0")` placé **avant** l'import, en tête de
   `src/loader.py` (désactive le backend TF de transformers ; `setdefault`
   n'écrase rien si l'utilisateur a déjà fixé la variable). Test : le splitter
   s'importe et découpe correctement. Note : ce blocage ne touche que la couche
   ingestion ; la couche requête (`main.py`) n'importe pas `loader`.

5. **Test connexion base.** `load_vectorstore()._collection.count()` →
   **323 vecteurs**, conforme au notebook. La base P1 n'est jamais modifiée.

6. **🔴 BLOCAGE 2 — versions LangChain incohérentes (env global).**
   Premier test end-to-end : échec à l'instanciation du LLM —
   `AttributeError: module 'langchain' has no attribute 'verbose'`.
   **Diagnostic** (`pip show`) : environnement global incohérent —
   `langchain-core 0.1.23` (très ancien, hérité des pins du notebook) +
   `langchain-google-genai 0.0.6` (ancien, pydantic v1) coexistant avec
   `langchain 1.3.2` / `langchain-community 0.4.2` (récents). Le vieux
   `langchain-google-genai` appelle un chemin de compat qui n'existe plus.
   **Résolution** : sur décision de l'utilisateur, création d'un **`.venv`
   dédié** + `pip install -r requirements.txt` (versions cohérentes). Dans le
   venv, embeddings et LLM s'instancient et atteignent réellement l'API Gemini.

7. **🔴 BLOCAGE 3 — clé API placeholder.**
   `400 INVALID_ARGUMENT : API key not valid`. **Cause** : `.env` contenait
   `GOOGLE_API_KEY=your_api_key_here` (placeholder, 17 caractères, ne commence
   pas par `AIza`). **Résolution** : l'utilisateur a fourni une vraie clé,
   inscrite dans `.env` (fichier ignoré par git).

8. **🔴 BLOCAGE 4 — quota du modèle LLM.**
   `429 RESOURCE_EXHAUSTED` avec `limit: 0` pour `gemini-2.0-flash`
   (`generate_content_free_tier_requests`). **Cause** : ce modèle n'est pas
   accordé sur le free tier de ce projet. **Résolution** : sonde de plusieurs
   modèles → `gemini-2.5-flash` disponible. `LLM_MODEL` mis à jour dans
   `src/config.py`. (À noter : la partie *embeddings* / récupération, elle,
   fonctionnait déjà — seul l'appel de génération butait sur le quota.)

9. **✅ Test end-to-end réussi** (venv + vraie clé + `gemini-2.5-flash`) :
   - « Qui est Thomas ? » → réponse correcte et sourcée (cosinus ~0.74) ;
   - « Quel est le rôle de Newt ? » → réponse correcte, 4 sources « Newt » (~0.78) ;
   - « Quelle est la capitale de la France ? » → **réponse de repli** correcte.

#### Observations & réglages
- **Calibrage du seuil.** Avec `gemini-embedding-001`, les chunks pertinents
  scorent ~0.72–0.78 et les hors-sujet ~0.55. Le seuil reste à **0.5** (valeur
  explicitement demandée par le sujet) : sur la question hors-sujet, les chunks
  passent de justesse le filtre, mais le **prompt** (2e barrière) impose quand
  même le repli — d'où un cas `grounded=True` mais avec la réponse de repli.
  Pour que le **filtre cosinus** (1re barrière) tranche seul ces cas
  (et économise un appel LLM), on pourrait monter le seuil à ~0.6.
- **Avertissement cosmétique** : `langchain_community.Chroma` est déprécié au
  profit de `langchain-chroma` (conservé pour rester aligné avec P1).
- **Encodage console** : les accents s'affichent mal dans le terminal Windows
  (cp1252) à l'exécution, mais les chaînes stockées/renvoyées sont en UTF-8
  correct — purement cosmétique.

#### Points ouverts
- Application Streamlit + évaluation du RAG (attendus du sujet) : à faire.
- Migration éventuelle `langchain_community.Chroma` → `langchain-chroma`.
- Versionner ou ignorer `chroma_maze_runner/`.

### 2026-06-11 — Mise en place du suivi et mise à jour de la doc

**Quoi**
- Création de ce fichier `suivi.md` (journal de suivi du projet).
- Réécriture de `CLAUDE.md` pour refléter les dernières avancées : ajout de la règle de suivi obligatoire (toute modif doit être consignée ici), documentation du nouveau pipeline `rag.ipynb` (RAG sur le wiki Fandom FR du *Labyrinthe* / Maze Runner), tableau des stores Chroma, notes de dépendances.

**Pourquoi**
- Le `CLAUDE.md` ne décrivait que le scaffolding initial (`src/` + `main.py`, RAG sur le blog Gemini en anglais), alors que le travail actuel se fait dans `rag.ipynb` sur un nouveau corpus français — la doc était désynchronisée du code.
- Centraliser l'historique des décisions dans un fichier versionné facilite le rendu du projet et la reprise du travail entre séances.

### (Antérieur) — État du projet repris dans ce journal

Reconstitué a posteriori à partir du dépôt :

- **Scaffolding initial** (`src/loader.py`, `src/vectorstore.py`, `src/generator.py`, `src/config.py`, `main.py`) : RAG sur **une page** du blog Google Gemini (anglais). Chargement `WebBaseLoader`, découpe entre deux délimiteurs, store `./chroma_db`, retriever `k=1`, LLM `gemini-2.0-flash`, embeddings `gemini-embedding-001`.
- **Pivot vers le corpus Maze Runner** (`rag.ipynb`) :
  - Sources : ~39 pages du wiki `mazerunner.fandom.com/fr` (personnages, lieux, créatures/concepts, livres & films).
  - Choix de l'**API MediaWiki** (`action=parse`) au lieu de `WebBaseLoader` → HTML plus propre, nettoyage BS4 (suppression `table`/`script`/`style`, extraction des `<p>`).
  - Filtrage des pages trop courtes (< 500 puis < 600 caractères) ; pause 0,3 s entre requêtes. 38 documents retenus sur 39 (« Braise » vide).
  - **Chunking** `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200) → 323 chunks.
  - **Indexation par batchs** de 80 avec pauses de 65 s pour respecter le quota de l'API d'embeddings. Store `./chroma_maze_runner` (323 vecteurs).
  - Retriever `k=4`.

**Points ouverts à trancher**
- Versionner ou ignorer `chroma_maze_runner/` (base lourde, régénérable mais coûteuse en quota).
- Le notebook utilise `time.sleep` sans import explicite de `time` visible dans la cellule d'imports — à vérifier/corriger.

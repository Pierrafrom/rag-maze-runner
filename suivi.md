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

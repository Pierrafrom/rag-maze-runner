## 1. Commandes à lancer avant la démo (15-20 min avant de passer)

```bash
cd lo17_rag

# 1. Démarrer le service Ollama (modèles déjà tirés : mistral, llama3.2:3b, nomic-embed-text)
docker compose up -d ollama

# 2. Vérifier qu'il est sain et que les modèles sont là
docker compose ps
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

Créer/vérifier le fichier `.env` à la racine (aucune clé API requise, mais ces
3 variables pilotent le mode offline et la liste des modèles proposés dans la
sidebar) :

```fish
printf "LLM_PROVIDER=ollama\nEMBEDDING_PROVIDER=ollama\nLOCAL_MODELS=mistral,llama3.2:3b\n" > .env
```

```bash
# 3. (Re)construire l'image si le code a changé depuis le dernier build
docker compose build rag-app

# 4. Lancer l'appli Streamlit conteneurisée
docker compose up -d rag-app
```

→ Ouvre `http://localhost:8501` dans le navigateur.

### Checklist avant de passer

- [ ] `docker compose ps` → `maze-ollama` et `maze-rag-app` tous les deux
      `healthy`/`running`
- [ ] Sidebar → Modèle de génération : **💻 mistral (local)** (meilleure
      qualité que llama3.2:3b)
- [ ] Sidebar → Multi-Query, Recherche hybride (BM25), Re-Ranking, CRAG : **ON**
      · **Self-RAG : OFF** — voir encadré ci-dessous
- [ ] Sidebar → "Afficher les sources" ON, "Afficher les reformulations" OFF
      (on l'activera en live)
- [ ] **Poser Q1 et Q2 maintenant** (voir section 2) pour qu'elles soient déjà
      dans l'historique du chat au moment de présenter — on ne les repose pas
      en live, on scrolle et on commente
- [ ] **Ne pas vider l'historique** avant de présenter (contrairement à un
      mode rapide type Gemini) : on a besoin que Q1/Q2 restent visibles

> **Pourquoi Self-RAG est désactivé en offline :** Self-RAG utilise le même
> modèle que la génération comme juge de sa propre réponse. Avec `mistral`
> (7B) comme juge, ce double-check est trop sévère et instable : il a rejeté
> une réponse correcte et bien sourcée sur le WICKED et fini en repli après
> la limite de corrections. C'est documenté comme une limite connue (cf.
> `suivi.md`, qui recommandait déjà un juge plus robuste pour RAGAS). Avec
> Gemini (juge de prod), Self-RAG est fiable — on le garde ON en mode Gemini,
> OFF en mode Ollama offline. CRAG reste actif (déterministe, temp=0, fiable
> dans les deux cas).

---

## 2. Questions à poser

### Q1 — Question dans le domaine (à poser AVANT de présenter)

```
Qu'est-ce que le WICKED ?
```

**En présentant (scroller jusqu'à cette réponse déjà affichée), dire :**
"la question a été reformulée 3 fois (Multi-Query), recherchée en dense +
lexical BM25, fusionnée par RRF, puis re-classée par FlashRank."

- Ouvrir 🔄 **Reformulations** → montre les 3 reformulations Multi-Query
- Ouvrir 🔗 **Sources** → montre les pages FR du wiki citées (preuve "données
  en français" + traçabilité)
- Pointer le badge `🟢 CRAG : PERTINENT · ⚪ Self-RAG : désactivé`

### Q2 — Question hors-sujet (à poser AVANT de présenter)

```
Qui est Harry Potter ?
```

**Résultat attendu (déjà visible) :** réponse de repli immédiate, badge
`🔴 CRAG : HORS-SUJET`.

**Dire :** "Le grader CRAG bloque avant même d'appeler le LLM de génération —
zéro tentative d'hallucination, pas juste un post-filtrage."

### Q3 — Même question hors-sujet, sans CRAG (EN LIVE, le seul live)

1. Décocher **CRAG** dans la sidebar (devant le jury) — Self-RAG reste OFF
2. Sélectionner **💻 llama3.2:3b (local)** dans la sidebar (plus rapide — ici
   on n'a pas besoin de qualité, juste du contraste)
3. Reposer exactement la même question :

```
Qui est Harry Potter ?
```

4. **Dès que l'attente commence**, basculer immédiatement sur le terminal pour
   présenter les métriques (section 3) — le temps de génération (~30-60 s)
   sert à montrer les chiffres, pas à attendre en silence
5. Revenir au navigateur : montrer la réponse générée sans CRAG — pas de
   badge de repli, le LLM a tenté une réponse sans garde-fou
6. **Recocher CRAG** (revenir à la config de référence)

> Si une question sur le Self-RAG arrive en Q&A : "on l'a désactivé en
> offline car le juge 7B local est trop instable pour s'auto-évaluer de
> façon fiable — avec Gemini comme juge de prod, on le garde actif." C'est
> un choix documenté, pas un abandon de la couche.

---

## 3. Métriques d'évaluation à montrer (pendant l'attente de Q3)

```bash
cat tests/evaluation/results/eval_latest.json | python3 -m json.tool | head -20
cat tests/evaluation/results/compare_hybrid_latest.json
```

**Chiffres clés à dire :**
- 24 questions de référence évaluées (19 dans le domaine, 5 hors-sujet)
- **95,8 %** d'exactitude de décision (répondre vs se replier)
- **80 %** de taux de repli correct sur les questions hors-sujet
- Ablation dense seul vs hybride BM25 : **+20 %** sur le taux de repli,
  **+4,2 %** sur l'exactitude → décision data-driven d'activer le BM25 par défaut

---

## 4. Bonus déploiement

Déjà démontré par le simple fait que toute la démo tourne offline. Si besoin
d'un rappel explicite :

```bash
docker compose ps
```

**Dire :** "Toute la démo que vous venez de voir tourne 100 % offline, sans
clé API ni connexion internet — `maze-ollama` sert le LLM et les embeddings,
l'index Chroma est monté en volume dans le conteneur Streamlit."

---

## 5. Repli si Docker ne démarre pas le jour J

```bash
docker compose logs ollama --tail 30
docker compose logs rag-app --tail 30
```

Si vraiment bloqué et qu'il y a du wifi : `LLM_PROVIDER=gemini` dans `.env`
+ `GOOGLE_API_KEY` valide, puis `uv run streamlit run streamlit_app.py` en
local (hors Docker) — bien plus rapide mais nécessite une clé API et du quota.

---

## 6. Mapping aux consignes officielles du sujet

| Consigne (PDF officiel) | Où c'est montré dans ce scénario |
|---|---|
| Données en français motivées | Sources citées à la Q1 (wiki Fandom FR) |
| Évaluation du RAG | Section 3 (métriques chiffrées) |
| Gestion des hallucinations | Q2 et Q3 (CRAG avec/sans) ; Self-RAG démontré et sa limite (juge local) expliquée |
| Application Streamlit | Tout le scénario tourne dans l'appli |
| Bonus déploiement | Toute la démo tourne 100 % offline (Docker/Ollama) |

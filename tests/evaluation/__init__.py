"""Évaluation du RAG Maze Runner.

Deux niveaux complémentaires, tous deux exécutés sur le *vrai* pipeline avancé
``RagPipeline`` (et non sur une chaîne naïve) :

* ``run_eval.py``  — évaluation comportementale légère, sans juge LLM :
  taux de repli correct, accord CRAG, présence des mots-clés attendus, latence.
* ``run_ragas.py`` — évaluation RAGAS (4 métriques de l'énoncé) ; le juge et
  les embeddings réutilisent les providers du projet (Gemini/Groq), pas Ollama.

Le jeu de référence partagé est ``eval_dataset.json``.
"""

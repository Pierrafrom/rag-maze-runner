"""Configuration centralisée des logs du RAG Maze Runner.

Appeler ``setup_logging()`` **une seule fois** au démarrage de l'application
(``main.py``, ``ingest.py``) pour configurer tous les loggers du projet.

Format imposé : ``AAAA-MM-JJ HH:MM:SS [LEVEL] nom.module | message``

Les préfixes d'étape (``[Multi-Query]``, ``[CRAG]``, …) sont portés par les
messages eux-mêmes ; chaque module déclare son logger en tête :
``logger = logging.getLogger(__name__)``.
"""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure le logger racine avec le format et le niveau demandés.

    Args:
        level: niveau minimum de journalisation (``logging.INFO`` par défaut).
               Passer ``logging.DEBUG`` pour voir les détails internes.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Réduire le bruit des bibliothèques tierces.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

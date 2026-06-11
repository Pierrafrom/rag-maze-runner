"""Package source du RAG Maze Runner.

Garde-fou d'import : certaines installations exposent ``langchain_text_splitters``
à un conflit Keras 3 / transformers (import eager de sentence-transformers).
Désactiver le backend TensorFlow de transformers évite ce crash. Placé ici, il
s'applique avant l'import de n'importe quel sous-module de ``src``.
``setdefault`` n'écrase pas une valeur déjà choisie par l'utilisateur.
"""

import os

os.environ.setdefault("USE_TF", "0")

"""Chargement et préparation des documents du wiki Maze Runner.

Réimplémentation, sous forme de module Python, de la logique d'ingestion du
notebook ``rag.ipynb`` (partie P1) :

    1. récupération du contenu via l'API MediaWiki (``action=parse``) ;
    2. nettoyage HTML avec BeautifulSoup (suppression tables/scripts/styles,
       extraction des paragraphes) ;
    3. filtrage des pages/documents trop courts ;
    4. découpage en chunks avec ``RecursiveCharacterTextSplitter``.

Ce module ne touche pas à ChromaDB : il produit uniquement des ``Document``.
"""

import os
import time
from urllib.parse import unquote

# Garde-fou : certaines installations exposent `langchain_text_splitters` à un
# conflit Keras 3 / transformers (import eager de sentence-transformers).
# Désactiver le backend TensorFlow de transformers évite ce crash à l'import.
# `setdefault` n'écrase pas une valeur déjà choisie par l'utilisateur.
os.environ.setdefault("USE_TF", "0")

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    MIN_DOC_CHARS,
    MIN_PAGE_CHARS,
    REQUEST_DELAY,
    USER_AGENT,
    WIKI_API_URL,
    WIKI_URLS,
)


def _page_name_from_url(url: str) -> str:
    """Extrait le nom de page MediaWiki (décodé) depuis une URL ``/fr/wiki/...``."""
    return unquote(url.split("/fr/wiki/")[-1])


def load_wiki_page(url: str, min_chars: int = MIN_PAGE_CHARS) -> Document | None:
    """Charge une page du wiki via l'API MediaWiki et renvoie un ``Document``.

    Args:
        url: URL complète de la page (``https://mazerunner.fandom.com/fr/wiki/...``).
        min_chars: longueur minimale de texte exploitable ; en dessous la page
            est considérée comme vide et ``None`` est renvoyé.

    Returns:
        Un ``Document`` (``page_content`` = texte nettoyé, ``metadata`` =
        ``source`` et ``title``) ou ``None`` si la page est absente/vide.
    """
    page_name = _page_name_from_url(url)
    params = {
        "action": "parse",
        "page": page_name,
        "prop": "text",
        "format": "json",
    }

    response = requests.get(
        WIKI_API_URL, params=params, headers={"User-Agent": USER_AGENT}
    )
    data = response.json()

    # L'API renvoie une clé "error" ou pas de "parse" quand la page n'existe pas.
    if "error" in data or "parse" not in data:
        return None

    html_content = data["parse"]["text"]["*"]
    soup = BeautifulSoup(html_content, "html.parser")

    # On retire les éléments non rédactionnels (tableaux, scripts, styles).
    for tag in soup.find_all(["table", "script", "style"]):
        tag.decompose()

    # On ne garde que le texte des paragraphes.
    paragraphs = soup.find_all("p")
    text = "\n".join(
        p.get_text(separator=" ", strip=True)
        for p in paragraphs
        if p.get_text(strip=True)
    )

    if len(text) < min_chars or "aucun texte sur cette page" in text.lower():
        return None

    return Document(
        page_content=text,
        metadata={"source": url, "title": page_name},
    )


def load_all_documents(
    urls: list[str] = WIKI_URLS,
    delay: float = REQUEST_DELAY,
    verbose: bool = True,
) -> list[Document]:
    """Charge toutes les pages du wiki, avec un délai entre les requêtes.

    Args:
        urls: liste d'URLs de pages à charger.
        delay: pause (s) entre deux requêtes pour ménager l'API.
        verbose: si vrai, journalise la progression sur la sortie standard.

    Returns:
        La liste des ``Document`` chargés avec succès.
    """
    documents: list[Document] = []
    for url in urls:
        doc = load_wiki_page(url)
        if doc is not None:
            documents.append(doc)
            if verbose:
                print(f"OK   : {doc.metadata['title']} ({len(doc.page_content)} car.)")
        elif verbose:
            print(f"VIDE : {_page_name_from_url(url)}")
        time.sleep(delay)

    if verbose:
        print(f"\n{len(documents)} documents chargés sur {len(urls)}")
    return documents


def filter_documents(
    documents: list[Document], min_chars: int = MIN_DOC_CHARS
) -> list[Document]:
    """Écarte les documents trop courts (faible valeur informative)."""
    return [doc for doc in documents if len(doc.page_content) >= min_chars]


def split_documents(documents: list[Document]) -> list[Document]:
    """Découpe les documents en chunks avec ``RecursiveCharacterTextSplitter``."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    return splitter.split_documents(documents)


def prepare_chunks(urls: list[str] = WIKI_URLS, verbose: bool = True) -> list[Document]:
    """Pipeline complet d'ingestion : chargement → filtrage → chunking.

    Returns:
        La liste des chunks prêts à être indexés dans Chroma.
    """
    documents = load_all_documents(urls, verbose=verbose)
    documents = filter_documents(documents)
    chunks = split_documents(documents)
    if verbose:
        print(f"{len(chunks)} chunks produits à partir de {len(documents)} documents")
    return chunks

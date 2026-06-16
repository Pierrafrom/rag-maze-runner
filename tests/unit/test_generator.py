"""Tests unitaires de la sélection de provider LLM (``get_llm``).

Hors-ligne : on vérifie uniquement le *routage* vers la bonne classe de chat,
la construction n'ouvrant aucune connexion réseau pour Ollama.
"""

import pytest

from src.config import OLLAMA_MODEL
from src.generator import get_llm


@pytest.mark.unit
def test_get_llm_ollama_provider_routes_to_chatollama() -> None:
    llm = get_llm(provider="ollama", model="mistral", temperature=0.0)
    assert type(llm).__name__ == "ChatOllama"
    assert llm.model == "mistral"


@pytest.mark.unit
def test_get_llm_ollama_uses_default_model_when_unspecified() -> None:
    llm = get_llm(provider="ollama")
    assert llm.model == OLLAMA_MODEL


@pytest.mark.unit
def test_get_llm_unknown_provider_falls_back_to_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provider non reconnu → branche Gemini (par défaut). On évite tout réseau
    # en interceptant le constructeur Gemini.
    captured: dict[str, object] = {}

    class _FakeGemini:
        def __init__(self, model: str, temperature: float) -> None:
            captured["model"] = model
            captured["temperature"] = temperature

    monkeypatch.setattr("src.generator.ChatGoogleGenerativeAI", _FakeGemini)
    get_llm(provider="provider-inexistant", temperature=0.2)
    assert captured["temperature"] == 0.2

from types import SimpleNamespace

import pytest

from answer import (
    DEFAULT_GENERATION_MODEL,
    SYSTEM_INSTRUCTION,
    format_context,
    generate_answer,
    print_answer,
)


MATCHES = [
    {
        "champion": "Malphite",
        "opponent": "Yone",
        "role": "top",
        "source_url": "https://example.com/one",
        "chunk_text": "Keep the wave near your tower.",
    },
    {
        "champion": "Malphite",
        "opponent": "Yone",
        "role": "top",
        "source_url": "https://example.com/two",
        "chunk_text": "Trade only when it is safe.",
    },
]


class FakeModels:
    def __init__(self, text="Use short trades. [Source 1]"):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


class FakeClient:
    def __init__(self, text="Use short trades. [Source 1]"):
        self.models = FakeModels(text)


def test_format_context_numbers_sources_and_excludes_urls():
    context = format_context(MATCHES)

    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "Keep the wave near your tower." in context
    assert "https://example.com" not in context
    assert context.count("<retrieved_context>") == 2


def test_generate_answer_sends_grounded_prompt_and_configuration():
    client = FakeClient()

    answer = generate_answer(client, " How should I play this lane? ", MATCHES)

    assert answer == "Use short trades. [Source 1]"
    [call] = client.models.calls
    assert call["model"] == DEFAULT_GENERATION_MODEL
    assert "How should I play this lane?" in call["contents"]
    assert "Keep the wave near your tower." in call["contents"]
    assert call["config"].system_instruction == SYSTEM_INSTRUCTION
    assert call["config"].temperature == 0.0
    assert call["config"].automatic_function_calling.disable is True


def test_generate_answer_rejects_missing_context_without_api_call():
    client = FakeClient()

    with pytest.raises(ValueError, match="without retrieved context"):
        generate_answer(client, "How should I play?", [])

    assert client.models.calls == []


def test_generate_answer_rejects_empty_response():
    with pytest.raises(ValueError, match="Gemini returned an empty answer"):
        generate_answer(FakeClient("   "), "How should I play?", MATCHES)


def test_print_answer_uses_database_source_urls(capsys):
    print_answer("Grounded answer. [Source 2]", MATCHES)

    output = capsys.readouterr().out
    assert "Grounded answer. [Source 2]" in output
    assert "[1] https://example.com/one" in output
    assert "[2] https://example.com/two" in output

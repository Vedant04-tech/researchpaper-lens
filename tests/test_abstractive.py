"""Unit tests for src/abstractive.py (TDD: RED -> GREEN -> REFACTOR)."""

from unittest.mock import MagicMock, patch
import pytest

from src.abstractive import (
    AbstractiveError,
    AbstractiveResult,
    HuggingFaceAdapter,
    MockAbstractiveAdapter,
    OpenAICompatibleAdapter,
    MissingApiKeyError,
    summarize_abstractive,
)


def test_mock_adapter():
    adapter = MockAbstractiveAdapter()
    text = "Information retrieval systems process queries against indexed documents."
    result = adapter.summarize(text)
    assert isinstance(result, AbstractiveResult)
    assert result.is_mock is True
    assert "Abstractive summary" in result.summary
    assert result.latency_seconds >= 0.0


def test_missing_api_key_hf():
    adapter = HuggingFaceAdapter(api_key="")
    with pytest.raises(MissingApiKeyError):
        adapter.summarize("Short sample text.")


def test_missing_api_key_openai():
    adapter = OpenAICompatibleAdapter(api_key="")
    with pytest.raises(MissingApiKeyError):
        adapter.summarize("Short sample text.")


@patch("requests.post")
def test_hf_adapter_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"summary_text": "This is an AI generated research summary."}]
    mock_post.return_value = mock_response

    adapter = HuggingFaceAdapter(api_key="hf_test_token")
    result = adapter.summarize("Sample document text describing an algorithm in detail.")

    assert isinstance(result, AbstractiveResult)
    assert result.summary == "This is an AI generated research summary."
    assert "bart" in result.model_name.lower()
    assert result.is_mock is False


@patch("requests.post")
def test_hf_adapter_api_error(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "Model is currently loading"
    mock_post.return_value = mock_response

    adapter = HuggingFaceAdapter(api_key="hf_test_token")
    with pytest.raises(AbstractiveError, match="Hugging Face API error"):
        adapter.summarize("Sample text.")


@patch("requests.post")
def test_openai_adapter_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Compact executive summary of the paper."}}]
    }
    mock_post.return_value = mock_response

    adapter = OpenAICompatibleAdapter(api_key="test_key", model="gpt-3.5-turbo")
    result = adapter.summarize("Sample text.")

    assert result.summary == "Compact executive summary of the paper."
    assert result.provider == "openai-compatible"


def test_summarize_abstractive_helper_mock():
    result = summarize_abstractive(
        "Information retrieval is the science of searching.",
        provider="mock",
    )
    assert isinstance(result, AbstractiveResult)
    assert result.is_mock is True


@patch("requests.post")
def test_summarize_abstractive_helper_hf(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"summary_text": "HF abstractive summary output."}]
    mock_post.return_value = mock_resp

    result = summarize_abstractive(
        "Document about PageRank centrality and graph algorithms.",
        provider="huggingface",
        api_key="hf_test_token",
    )
    assert result.summary == "HF abstractive summary output."
    assert result.provider == "huggingface"


@patch("requests.post")
def test_summarize_abstractive_helper_openai(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "OpenAI abstractive summary output."}}]
    }
    mock_post.return_value = mock_resp

    result = summarize_abstractive(
        "Document about vector space retrieval models.",
        provider="openai",
        api_key="test_sk",
    )
    assert result.summary == "OpenAI abstractive summary output."
    assert result.provider == "openai-compatible"


def test_summarize_abstractive_invalid_provider():
    with pytest.raises(AbstractiveError, match="Unsupported abstractive provider"):
        summarize_abstractive("Sample text.", provider="unsupported_provider")


def test_abstractive_empty_text():
    mock_adapter = MockAbstractiveAdapter()
    result = mock_adapter.summarize("")
    assert result.summary == ""
    assert result.is_mock is True


@patch("requests.post")
def test_abstractive_long_text_truncation(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"summary_text": "Summary of long text."}]
    mock_post.return_value = mock_resp

    adapter = HuggingFaceAdapter(api_key="hf_test_token")
    long_text = "Word " * 2000  # 10000 characters
    result = adapter.summarize(long_text)
    assert result.summary == "Summary of long text."

    # Verify posted payload inputs was truncated to <= 4000 chars
    called_payload = mock_post.call_args[1]["json"]
    assert len(called_payload["inputs"]) <= 4000


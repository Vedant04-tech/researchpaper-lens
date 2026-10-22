"""ResearchPaper Lens — Abstractive Summarization Adapter (Phase 6 Enhancement).

Provides a decoupled, pluggable architecture for abstractive summarization.
Supports:
1. MockAbstractiveAdapter (offline, deterministic demonstration & testing)
2. HuggingFaceAdapter (Hugging Face Inference API e.g. facebook/bart-large-cnn)
3. OpenAICompatibleAdapter (OpenAI / Groq / Local Ollama v1 endpoints)

Guarantees:
- Safe error handling: provider timeouts, missing keys, or HTTP errors do NOT crash the app.
- Clear attribution: outputs are explicitly marked as model-generated.
- Privacy awareness: requires explicit user activation and key entry before making outbound requests.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import requests

DEFAULT_HF_MODEL: str = "facebook/bart-large-cnn"
DEFAULT_OPENAI_MODEL: str = "gpt-3.5-turbo"
DEFAULT_REQUEST_TIMEOUT: int = 30
MAX_INPUT_CHARS: int = 4000  # Conservative limit to avoid token overflow on small models


class AbstractiveError(Exception):
    """Base exception for abstractive summarization failures."""


class MissingApiKeyError(AbstractiveError):
    """Raised when an API key is required but missing."""


class ModelTimeoutError(AbstractiveError):
    """Raised when provider request times out."""


@dataclass
class AbstractiveResult:
    """Structured response from abstractive summarization."""

    summary: str
    model_name: str
    provider: str
    latency_seconds: float
    is_mock: bool = False
    warning: Optional[str] = None


class AbstractiveAdapter(ABC):
    """Base interface for abstractive summarization adapters."""

    @abstractmethod
    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> AbstractiveResult:
        """Generates an abstractive summary from input text."""


class MockAbstractiveAdapter(AbstractiveAdapter):
    """Deterministic offline mock adapter for testing and zero-cost demonstration."""

    def __init__(self, model_name: str = "mock-abstractive-v1") -> None:
        self.model_name = model_name

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> AbstractiveResult:
        start_time = time.perf_counter()
        cleaned = text.strip()
        if not cleaned:
            return AbstractiveResult(
                summary="",
                model_name=self.model_name,
                provider="mock",
                latency_seconds=0.0,
                is_mock=True,
            )

        # Generate a synthetic executive summary from the first few key points
        words = cleaned.split()
        lead = " ".join(words[:min(40, len(words))])
        summary_text = (
            f"Abstractive summary (synthesized representation): The document investigates foundational methodology "
            f"and experimental results regarding {lead}... In conclusion, the authors validate core performance "
            f"and propose practical extensions for future research."
        )

        latency = time.perf_counter() - start_time
        return AbstractiveResult(
            summary=summary_text,
            model_name=self.model_name,
            provider="mock",
            latency_seconds=round(latency, 3),
            is_mock=True,
        )


class HuggingFaceAdapter(AbstractiveAdapter):
    """Adapter for Hugging Face Inference API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_HF_MODEL,
        model: Optional[str] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self.api_key = api_key.strip() if api_key else ""
        self.model_name = model or model_name
        self.timeout = timeout

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> AbstractiveResult:
        if not self.api_key:
            raise MissingApiKeyError("Hugging Face API token is required. Please supply a valid HF_TOKEN.")

        truncated_text = text[:MAX_INPUT_CHARS].strip()
        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": truncated_text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False,
            },
        }

        start_time = time.perf_counter()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.Timeout as e:
            raise ModelTimeoutError(f"Hugging Face API request timed out after {self.timeout}s: {e}") from e
        except requests.RequestException as e:
            raise AbstractiveError(f"Hugging Face network request failed: {e}") from e

        latency = time.perf_counter() - start_time

        if response.status_code != 200:
            raise AbstractiveError(
                f"Hugging Face API error (status {response.status_code}): {response.text}"
            )

        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0 and "summary_text" in data[0]:
                summary_text = data[0]["summary_text"].strip()
            elif isinstance(data, dict) and "summary_text" in data:
                summary_text = data["summary_text"].strip()
            else:
                summary_text = str(data)
        except Exception as e:
            raise AbstractiveError(f"Failed to parse Hugging Face JSON response: {e}") from e

        return AbstractiveResult(
            summary=summary_text,
            model_name=self.model_name,
            provider="huggingface",
            latency_seconds=round(latency, 3),
            is_mock=False,
        )


class OpenAICompatibleAdapter(AbstractiveAdapter):
    """Adapter for OpenAI-compatible Chat Completions endpoints (OpenAI, Groq, Ollama)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_OPENAI_MODEL,
        model: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self.api_key = api_key.strip() if api_key else ""
        self.model_name = model or model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> AbstractiveResult:
        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise MissingApiKeyError("API key is required for external OpenAI-compatible endpoints.")

        truncated_text = text[:MAX_INPUT_CHARS].strip()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an academic researcher. Provide a concise, factual executive summary "
                        "of the following text. Do not invent findings or cite unmentioned sources."
                    ),
                },
                {"role": "user", "content": f"Summarize this research paper excerpt:\n\n{truncated_text}"},
            ],
            "max_tokens": max_length * 2,
            "temperature": 0.3,
        }

        start_time = time.perf_counter()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.Timeout as e:
            raise ModelTimeoutError(f"Request timed out after {self.timeout}s: {e}") from e
        except requests.RequestException as e:
            raise AbstractiveError(f"Network request to {self.base_url} failed: {e}") from e

        latency = time.perf_counter() - start_time

        if response.status_code != 200:
            raise AbstractiveError(
                f"OpenAI-compatible API error (status {response.status_code}): {response.text}"
            )

        try:
            data = response.json()
            summary_text = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise AbstractiveError(f"Failed to parse API response: {e}") from e

        return AbstractiveResult(
            summary=summary_text,
            model_name=self.model_name,
            provider="openai-compatible",
            latency_seconds=round(latency, 3),
            is_mock=False,
        )


def summarize_abstractive(
    text: str,
    provider: str = "mock",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
) -> AbstractiveResult:
    """Convenience factory function to run abstractive summarization.

    Args:
        text: Input document or extractive summary text.
        provider: Provider identifier ('mock', 'huggingface', 'openai').
        api_key: API token or key.
        model: Custom model identifier if overriding defaults.
        base_url: Custom API base URL for OpenAI-compatible endpoints.

    Returns:
        AbstractiveResult with generated summary and execution metadata.
    """
    prov = provider.lower()
    if prov == "mock":
        adapter = MockAbstractiveAdapter(model_name=model or "mock-abstractive-v1")
    elif prov in ("huggingface", "hf"):
        adapter = HuggingFaceAdapter(api_key=api_key, model_name=model or DEFAULT_HF_MODEL)
    elif prov in ("openai", "openai-compatible", "ollama", "groq"):
        adapter = OpenAICompatibleAdapter(
            api_key=api_key,
            model_name=model or DEFAULT_OPENAI_MODEL,
            base_url=base_url or "https://api.openai.com/v1",
        )
    else:
        raise AbstractiveError(f"Unsupported abstractive provider: '{provider}'")

    return adapter.summarize(text)

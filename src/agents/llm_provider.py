"""Shared LLM factory: OpenAI, Gemini, Groq (Llama 3), or Ollama (local Llama)."""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

Provider = Literal["openai", "gemini", "groq", "ollama"]


def resolve_llm_provider() -> Provider:
    name = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if name in ("llama", "llama3", "groq"):
        return "groq"
    if name == "ollama":
        return "ollama"
    if name == "gemini":
        return "gemini"
    return "openai"


def llm_credentials_ok() -> bool:
    p = resolve_llm_provider()
    if p == "gemini":
        return bool(os.getenv("GOOGLE_API_KEY", "").strip())
    if p == "groq":
        return bool(os.getenv("GROQ_API_KEY", "").strip())
    if p == "ollama":
        return True
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def make_chat_llm(*, temperature: float, max_tokens: int) -> BaseChatModel:
    """Return a chat model for explanation / judge chains."""
    p = resolve_llm_provider()

    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not os.getenv("GOOGLE_API_KEY", "").strip():
            raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if p == "groq":
        from langchain_groq import ChatGroq

        if not os.getenv("GROQ_API_KEY", "").strip():
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq (or llama3)")
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if p == "ollama":
        from langchain_community.chat_models import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3")
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        return ChatOllama(
            model=model,
            base_url=base,
            temperature=temperature,
            num_predict=max_tokens,
        )

    from langchain_openai import ChatOpenAI

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        max_tokens=max_tokens,
    )

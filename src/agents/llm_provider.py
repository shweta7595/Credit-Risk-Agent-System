"""Shared LLM factory: Groq (Llama 3)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)


def llm_credentials_ok() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def make_chat_llm(*, temperature: float, max_tokens: int) -> BaseChatModel:
    from langchain_groq import ChatGroq

    if not llm_credentials_ok():
        raise RuntimeError("GROQ_API_KEY is not set.")

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
        max_tokens=max_tokens,
    )

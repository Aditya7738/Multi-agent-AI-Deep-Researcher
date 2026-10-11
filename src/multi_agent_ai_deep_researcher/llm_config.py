import os

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


def get_llm() -> LLM:
    """Create the shared CrewAI LLM using OpenRouter."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/deepseek/deepseek-chat",
    )

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing from the .env file."
        )

    return LLM(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        timeout=120,
        max_tokens=2048,
    )

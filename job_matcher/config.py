"""Settings loaded from the .env file.

Every secret and tunable value lives here, so no other module reads
environment variables directly. One place to change, one place to debug.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # reads .env in the current folder into environment variables


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    max_jobs_to_score: int = 8      # how many jobs the LLM scores (free models have rate limits)
    request_timeout: float = 60.0   # seconds to wait for the LLM
    pause_between_calls: float = 2.0  # be gentle with free-tier rate limits


def get_settings() -> Settings:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("MODEL", "").strip()

    if not api_key or api_key == "your-key-here":
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Copy .env.example to .env and paste your key."
        )
    if not model:
        raise RuntimeError(
            "MODEL is missing in .env. Pick a free model at https://openrouter.ai/models "
            "(filter: free) and paste its ID."
        )
    # Hard rule for this project: free models only, so you never get billed by accident.
    if not model.endswith(":free"):
        raise RuntimeError(
            f"MODEL '{model}' is not a free model. Use an OpenRouter model ID ending in ':free'."
        )

    return Settings(
        api_key=api_key,
        model=model,
        max_jobs_to_score=int(os.getenv("MAX_JOBS_TO_SCORE", "8")),
    )

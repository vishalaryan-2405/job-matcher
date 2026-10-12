"""Data shapes used across the project (Pydantic models).

LLM output is untrusted: it can miss fields or use wrong types.
Validating it through these models turns silent bugs into clear errors.
"""
from pydantic import BaseModel, Field, field_validator


class Job(BaseModel):
    title: str
    location: str = ""
    url: str = ""
    department: str = ""
    description: str = ""


class CandidateProfile(BaseModel):
    name: str = ""
    years_experience: float = 0
    skills: list[str] = []
    roles: list[str] = []
    education: str = ""
    summary: str = ""

    def as_text(self) -> str:
        """A compact text version used for TF-IDF matching and prompts."""
        return (
            f"Roles: {', '.join(self.roles)}. "
            f"Skills: {', '.join(self.skills)}. "
            f"Experience: {self.years_experience} years. "
            f"Education: {self.education}. {self.summary}"
        )


class MatchResult(BaseModel):
    job: Job
    score: int = Field(ge=0, le=100)
    verdict: str = ""
    reasons: list[str] = []
    missing_skills: list[str] = []

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, value):
        # Models sometimes return "85", 85.5 or 120. Normalise to an int in 0..100.
        try:
            return max(0, min(100, int(float(value))))
        except (TypeError, ValueError):
            return 0


def verdict_for(score: int) -> str:
    """Business rule lives in code, not in the prompt: the model gives a number,
    we decide what the number means."""
    if score >= 75:
        return "strong match"
    if score >= 50:
        return "possible match"
    return "weak match"

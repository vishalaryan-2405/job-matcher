"""Read a resume file and turn it into a structured CandidateProfile.

The profile is cached on disk: your resume rarely changes, and every LLM
call on a free model counts against your daily limit.
"""
import hashlib
import json
import logging
import re
from pathlib import Path

from pypdf import PdfReader

from .config import Settings
from .llm import chat_json
from .models import CandidateProfile

log = logging.getLogger(__name__)
CACHE_DIR = Path(".cache")

PROFILE_SYSTEM = """You extract structured data from resumes.
Return ONLY a JSON object with these keys:
{
  "name": string,
  "years_experience": number (total professional experience incl. internships, e.g. 0.5),
  "skills": [string] (technical skills, tools, frameworks; max 30),
  "roles": [string] (job titles the person is suited for, e.g. "ML Engineer"),
  "education": string (highest degree, field, year),
  "summary": string (2 sentences on strengths)
}
Use only facts in the resume. Do not invent anything."""


def read_resume_text(path: str) -> str:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Resume not found: {file}")

    suffix = file.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".txt", ".md"):
        text = file.read_text(encoding="utf-8")
    else:
        raise ValueError("Resume must be a .pdf, .txt or .md file")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 200:
        raise ValueError(
            "Very little text was extracted from the resume. "
            "If it's a scanned image PDF, export it as a text PDF or save it as .txt."
        )
    return text


def build_profile(settings: Settings, resume_text: str, use_cache: bool = True) -> CandidateProfile:
    key = hashlib.sha256((settings.model + resume_text).encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"profile_{key}.json"

    if use_cache and cache_file.exists():
        log.info("Using cached resume profile (%s)", cache_file)
        return CandidateProfile.model_validate_json(cache_file.read_text(encoding="utf-8"))

    log.info("Parsing resume with the LLM...")
    data = chat_json(settings, PROFILE_SYSTEM, resume_text[:12000])
    profile = CandidateProfile.model_validate(data)

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(profile.model_dump(), indent=2), encoding="utf-8")
    return profile

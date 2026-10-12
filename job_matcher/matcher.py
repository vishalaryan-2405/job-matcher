"""Match a candidate profile against jobs in two stages.

Stage 1 (cheap, local, instant): TF-IDF similarity ranks ALL jobs and keeps
the top few. No API calls, so a company with 300 jobs costs nothing here.

Stage 2 (expensive, smart): the LLM scores only those few, with reasons.

This "cheap filter, then expensive reasoning" pattern is used everywhere in
real AI systems (search, RAG, recommendations).
"""
import json
import logging
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Settings
from .llm import chat_json
from .models import CandidateProfile, Job, MatchResult, verdict_for

log = logging.getLogger(__name__)


def filter_by_location(jobs: list[Job], location: str | None) -> list[Job]:
    """Keep jobs in the wanted location, plus remote jobs and jobs with no location."""
    if not location:
        return jobs
    wanted = location.lower()
    return [j for j in jobs if not j.location
            or wanted in j.location.lower()
            or "remote" in j.location.lower()]


def prefilter(profile: CandidateProfile, jobs: list[Job], top_k: int) -> list[Job]:
    if len(jobs) <= top_k:
        return jobs
    # Title is repeated so it weighs more than a long description.
    docs = [f"{j.title} {j.title} {j.department} {j.description[:3000]}" for j in jobs]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([profile.as_text()] + docs)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    ranked = sorted(range(len(jobs)), key=lambda i: similarities[i], reverse=True)
    return [jobs[i] for i in ranked[:top_k]]


SCORE_SYSTEM = """You are a strict, honest technical recruiter.
Score how well the candidate fits the job from 0 to 100 using this rubric:
- Skills overlap with the job's REQUIRED skills: 50 points
- Experience level fit (years, seniority): 30 points
- Role and domain fit: 20 points
Rules:
- If the job needs far more years than the candidate has (e.g. 5+ years vs 1 year), score at most 40.
- Only count skills that appear in the candidate profile. Never assume.
- If the job description is empty, judge from the title only and say so in reasons.
Return ONLY JSON:
{"score": integer, "reasons": [up to 3 short strings], "missing_skills": [up to 5 strings]}"""


def score_job(settings: Settings, profile: CandidateProfile, job: Job) -> MatchResult:
    user = (
        "CANDIDATE PROFILE:\n" + json.dumps(profile.model_dump(), indent=2)
        + f"\n\nJOB TITLE: {job.title}\nLOCATION: {job.location}\n"
        + f"JOB DESCRIPTION:\n{job.description[:6000]}"
    )
    data = chat_json(settings, SCORE_SYSTEM, user)
    if not isinstance(data, dict):
        data = {}
    result = MatchResult(
        job=job,
        score=data.get("score", 0),
        reasons=[str(r) for r in data.get("reasons", [])][:3],
        missing_skills=[str(s) for s in data.get("missing_skills", [])][:5],
    )
    result.verdict = verdict_for(result.score)
    return result


def match_jobs(settings: Settings, profile: CandidateProfile, jobs: list[Job],
               describe=None, on_progress=None) -> list[MatchResult]:
    """Prefilter, then LLM-score the shortlist.

    describe(job)            -> fills an empty job description
    on_progress(done, total, title) -> lets a caller (the web app) show progress
    """
    shortlist = prefilter(profile, jobs, settings.max_jobs_to_score)
    log.info("Shortlisted %d of %d jobs for LLM scoring", len(shortlist), len(jobs))

    results = []
    for index, job in enumerate(shortlist, start=1):
        if on_progress:
            on_progress(index - 1, len(shortlist), job.title)
        if not job.description and describe:
            job.description = describe(job)
        log.info("[%d/%d] Scoring: %s", index, len(shortlist), job.title)
        try:
            results.append(score_job(settings, profile, job))
        except Exception as exc:  # one bad job shouldn't stop the whole run
            log.warning("Could not score '%s': %s", job.title, exc)
        time.sleep(settings.pause_between_calls)
    return sorted(results, key=lambda r: r.score, reverse=True)

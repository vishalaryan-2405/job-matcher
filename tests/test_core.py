"""Tests that run without internet or an API key.

The LLM is replaced by a fake function ("mocking"), so tests are fast,
free and give the same result every time.
"""
import pytest

from job_matcher import config, fetchers, matcher
from job_matcher.llm import extract_json
from job_matcher.models import CandidateProfile, Job, MatchResult, verdict_for


# ---------- config ----------

def test_settings_reject_paid_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "openai/gpt-4o")
    with pytest.raises(RuntimeError, match="not a free model"):
        config.get_settings()


def test_settings_accept_free_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "some-org/some-model:free")
    assert config.get_settings().model.endswith(":free")


# ---------- JSON extraction ----------

@pytest.mark.parametrize("reply", [
    '{"score": 80}',
    '```json\n{"score": 80}\n```',
    'Sure! Here is the result:\n{"score": 80}\nHope this helps.',
])
def test_extract_json_handles_messy_replies(reply):
    assert extract_json(reply) == {"score": 80}


def test_extract_json_raises_when_missing():
    with pytest.raises(ValueError):
        extract_json("I cannot help with that.")


# ---------- models ----------

@pytest.mark.parametrize("raw,expected", [("85", 85), (120, 100), (-5, 0), ("abc", 0), (72.6, 72)])
def test_score_is_clamped(raw, expected):
    assert MatchResult(job=Job(title="x"), score=raw).score == expected


def test_verdict_rules():
    assert verdict_for(80) == "strong match"
    assert verdict_for(60) == "possible match"
    assert verdict_for(20) == "weak match"


# ---------- platform detection ----------

@pytest.mark.parametrize("url,expected", [
    ("https://boards.greenhouse.io/acme", ("greenhouse", "acme")),
    ("https://job-boards.greenhouse.io/acme/jobs/123", ("greenhouse", "acme")),
    ("https://boards.greenhouse.io/embed/job_board?for=acme", ("greenhouse", "acme")),
    ("https://jobs.lever.co/acme", ("lever", "acme")),
    ("https://jobs.ashbyhq.com/acme", ("ashby", "acme")),
    ("https://acme.com/careers", ("generic", None)),
])
def test_detect_platform(url, expected):
    assert fetchers.detect_platform(url) == expected


def test_find_embedded_board():
    page = '<div id="grnhse_app"></div><script src="https://boards.greenhouse.io/embed/job_board/js?for=acme"></script>'
    assert fetchers.find_embedded_board(page) == ("greenhouse", "acme")
    assert fetchers.find_embedded_board("<a href='https://jobs.lever.co/acme/123'>Job</a>") == ("lever", "acme")
    assert fetchers.find_embedded_board("<p>no board here</p>") is None


# ---------- JSON-LD parsing ----------

JSONLD_PAGE = """
<html><head><script type="application/ld+json">
{"@context": "https://schema.org", "@graph": [
  {"@type": "Organization", "name": "Acme"},
  {"@type": "JobPosting", "title": "ML Engineer", "url": "/jobs/ml",
   "description": "<p>Build <b>Python</b> ML pipelines</p>",
   "jobLocation": {"address": {"addressLocality": "Hyderabad", "addressCountry": "IN"}}},
  {"@type": "JobPosting", "title": "Support Agent", "jobLocationType": "TELECOMMUTE"}
]}
</script></head><body></body></html>
"""


def test_parse_jsonld_jobs():
    jobs = fetchers.parse_jsonld_jobs(JSONLD_PAGE, "https://acme.com/careers")
    assert [j.title for j in jobs] == ["ML Engineer", "Support Agent"]
    assert jobs[0].url == "https://acme.com/jobs/ml"
    assert jobs[0].location == "Hyderabad, IN"
    assert "Python" in jobs[0].description and "<b>" not in jobs[0].description
    assert jobs[1].location == "Remote"


def test_broken_jsonld_is_skipped():
    page = '<script type="application/ld+json">{not valid json</script>'
    assert fetchers.parse_jsonld_jobs(page, "https://x.com") == []


# ---------- matching ----------

PROFILE = CandidateProfile(
    name="Test", years_experience=1, skills=["Python", "scikit-learn", "FastAPI", "LLM", "RAG"],
    roles=["Machine Learning Engineer"], education="B.Tech CSE",
)


def test_prefilter_ranks_relevant_jobs_first():
    jobs = [
        Job(title="Head Chef", description="Cook meals, manage kitchen staff"),
        Job(title="Machine Learning Engineer", description="Python, scikit-learn, LLM and RAG systems"),
        Job(title="Sales Executive", description="Cold calls and client meetings"),
    ]
    assert matcher.prefilter(PROFILE, jobs, top_k=1)[0].title == "Machine Learning Engineer"


def test_location_filter_keeps_remote_and_unknown():
    jobs = [Job(title="a", location="Hyderabad"), Job(title="b", location="London"),
            Job(title="c", location="Remote"), Job(title="d")]
    assert [j.title for j in matcher.filter_by_location(jobs, "hyderabad")] == ["a", "c", "d"]


def test_score_job_with_fake_llm(monkeypatch):
    def fake_chat_json(settings, system, user):
        assert "Machine Learning Engineer" in user  # the job reached the prompt
        return {"score": "82", "reasons": ["Python and ML match"], "missing_skills": ["Docker"]}

    monkeypatch.setattr(matcher, "chat_json", fake_chat_json)
    settings = config.Settings(api_key="k", model="m:free", pause_between_calls=0)
    result = matcher.score_job(settings, PROFILE, Job(title="Machine Learning Engineer"))
    assert result.score == 82
    assert result.verdict == "strong match"
    assert result.missing_skills == ["Docker"]


def test_match_jobs_survives_one_failure(monkeypatch):
    def flaky(settings, system, user):
        if "Broken" in user:
            raise RuntimeError("model hiccup")
        return {"score": 60, "reasons": [], "missing_skills": []}

    monkeypatch.setattr(matcher, "chat_json", flaky)
    settings = config.Settings(api_key="k", model="m:free", pause_between_calls=0)
    results = matcher.match_jobs(settings, PROFILE, [Job(title="Broken role"), Job(title="ML role")])
    assert [r.job.title for r in results] == ["ML role"]

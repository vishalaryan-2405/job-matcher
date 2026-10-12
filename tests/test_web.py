"""Tests for the web API, with the LLM and the internet replaced by fakes."""
import time

import pytest
from fastapi.testclient import TestClient

from job_matcher import web
from job_matcher.config import Settings
from job_matcher.models import CandidateProfile, Job, MatchResult

RESUME_TXT = ("Jane Doe. Machine learning engineer. Python, scikit-learn, FastAPI, LLM, RAG. " * 5).encode()


@pytest.fixture
def client(monkeypatch):
    settings = Settings(api_key="k", model="m:free", pause_between_calls=0)
    monkeypatch.setattr(web, "APP_PASSWORD", "")  # tests run unlocked even if your .env sets one
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    monkeypatch.setattr(web, "build_profile",
                        lambda s, text: CandidateProfile(name="Jane", years_experience=1, skills=["Python"]))
    monkeypatch.setattr(web, "fetch_jobs", lambda s, url, c: (
        [Job(title="ML Engineer", location="Remote", description="Python"),
         Job(title="Chef", location="London", description="Cooking")], "lever API"))
    monkeypatch.setattr(web, "match_jobs", lambda s, p, jobs, describe=None, on_progress=None: [
        MatchResult(job=j, score=80 if "ML" in j.title else 20, verdict="x") for j in jobs])
    monkeypatch.setattr(web, "score_job", lambda s, p, job: MatchResult(job=job, score=66, verdict="possible match"))
    return TestClient(web.app)


def wait_for(client, job_id):
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_index_page_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Which of their open roles fit you?" in resp.text


def test_health_reports_model(client):
    assert client.get("/api/health").json() == {"ok": True, "model": "m:free", "locked": False}


def test_match_flow(client):
    resp = client.post("/api/match", files={"resume": ("cv.txt", RESUME_TXT)},
                       data={"url": "https://jobs.lever.co/acme", "location": ""})
    assert resp.status_code == 200
    job = wait_for(client, resp.json()["job_id"])
    assert job["status"] == "done"
    assert job["found"] == 2 and job["method"] == "lever API"
    assert [r["job"]["title"] for r in job["results"]] == ["ML Engineer", "Chef"]


def test_location_filter_applies(client):
    resp = client.post("/api/match", files={"resume": ("cv.txt", RESUME_TXT)},
                       data={"url": "https://jobs.lever.co/acme", "location": "India"})
    job = wait_for(client, resp.json()["job_id"])
    assert job["kept"] == 1  # London dropped, Remote kept


def test_score_flow(client):
    resp = client.post("/api/score", files={"resume": ("cv.txt", RESUME_TXT)},
                       data={"description": "We need a Python ML engineer.\n" * 5, "title": ""})
    job = wait_for(client, resp.json()["job_id"])
    assert job["results"][0]["score"] == 66
    assert job["results"][0]["job"]["title"] == "We need a Python ML engineer."


def test_rejects_bad_inputs(client):
    bad_url = client.post("/api/match", files={"resume": ("cv.txt", RESUME_TXT)}, data={"url": "acme.com"})
    assert bad_url.status_code == 400
    bad_file = client.post("/api/match", files={"resume": ("cv.docx", b"x")}, data={"url": "https://x.com"})
    assert bad_file.status_code == 400
    short_jd = client.post("/api/score", files={"resume": ("cv.txt", RESUME_TXT)}, data={"description": "hi"})
    assert short_jd.status_code == 400


def test_errors_reach_the_page(client, monkeypatch):
    def boom(s, url, c):
        raise web.PageNotReadable("No job openings found on this page.")
    monkeypatch.setattr(web, "fetch_jobs", boom)
    resp = client.post("/api/match", files={"resume": ("cv.txt", RESUME_TXT)}, data={"url": "https://x.com/careers"})
    job = wait_for(client, resp.json()["job_id"])
    assert job["status"] == "error" and "No job openings" in job["error"]
    assert job["can_paste"] is True


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404

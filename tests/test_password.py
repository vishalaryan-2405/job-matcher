"""The deployed site is locked with APP_PASSWORD; locally it stays open."""
from fastapi.testclient import TestClient

from job_matcher import web
from job_matcher.config import Settings


def make_client(monkeypatch, password):
    monkeypatch.setattr(web, "APP_PASSWORD", password)
    monkeypatch.setattr(web, "get_settings", lambda: Settings(api_key="k", model="secret-model:free"))
    return TestClient(web.app)


def test_open_when_no_password(monkeypatch):
    client = make_client(monkeypatch, "")
    assert client.get("/api/jobs/nope").status_code == 404  # reached the app, no lock
    assert client.get("/api/health").json()["model"] == "secret-model:free"


def test_locked_site_rejects_missing_or_wrong_password(monkeypatch):
    client = make_client(monkeypatch, "letmein")
    assert client.get("/api/jobs/nope").status_code == 401
    assert client.get("/api/jobs/nope", headers={"X-App-Password": "wrong"}).status_code == 401
    assert client.get("/api/jobs/nope", headers={"X-App-Password": "letmein"}).status_code == 404


def test_locked_site_still_serves_page_and_hides_model(monkeypatch):
    client = make_client(monkeypatch, "letmein")
    assert client.get("/").status_code == 200
    health = client.get("/api/health").json()
    assert health["locked"] is True and health["model"] is None

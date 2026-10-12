"""Every way a careers page or the LLM can fail should give a plain message.

httpx.MockTransport lets us fake any server response without the internet,
so we can test a 403 or a timeout as easily as a success.
"""
import httpx
import pytest

from job_matcher import fetchers, llm
from job_matcher.config import Settings

SETTINGS = Settings(api_key="k", model="m:free", pause_between_calls=0, request_timeout=1)


def client_returning(status=200, raise_exc=None):
    def handler(request):
        if raise_exc:
            raise raise_exc
        return httpx.Response(status, text="nope")
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("status,phrase", [
    (403, "doesn't allow automated access"),
    (401, "doesn't allow automated access"),
    (404, "wasn't found"),
    (429, "limiting requests"),
    (503, "having problems on its side"),
])
def test_page_status_codes_become_plain_messages(status, phrase):
    with pytest.raises(fetchers.PageNotReadable) as err:
        fetchers.fetch_jobs(SETTINGS, "https://acme.com/careers", client_returning(status))
    message = str(err.value)
    assert phrase in message
    assert str(status) not in message  # no raw codes shown to the user
    assert "acme.com" in message


def test_403_suggests_pasting_the_job():
    with pytest.raises(fetchers.PageNotReadable) as err:
        fetchers.fetch_jobs(SETTINGS, "https://acme.com/careers", client_returning(403))
    assert "Paste a job" in str(err.value)


def test_unknown_platform_board():
    with pytest.raises(fetchers.PageNotReadable) as err:
        fetchers.fetch_jobs(SETTINGS, "https://jobs.lever.co/not-a-company", client_returning(404))
    assert "No Lever job board was found" in str(err.value)


@pytest.mark.parametrize("exc,phrase", [
    (httpx.ConnectTimeout("slow"), "took too long"),
    (httpx.ConnectError("dns"), "Couldn't connect"),
])
def test_network_failures(exc, phrase):
    with pytest.raises(fetchers.PageNotReadable) as err:
        fetchers.fetch_jobs(SETTINGS, "https://acme.com/careers", client_returning(raise_exc=exc))
    assert phrase in str(err.value)


@pytest.mark.parametrize("status,phrase", [
    (401, "API key was rejected"),
    (404, "model wasn't found"),
    (402, "needs paid credits"),
])
def test_llm_errors_are_plain(monkeypatch, status, phrase):
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: httpx.Response(status, text='{"error":"raw"}'))
    with pytest.raises(llm.LLMError) as err:
        llm.chat(SETTINGS, [{"role": "user", "content": "hi"}])
    assert phrase in str(err.value)
    assert "raw" not in str(err.value)


def test_llm_rate_limit_after_retries(monkeypatch):
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: httpx.Response(429, text="slow down"))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)  # don't actually wait in tests
    with pytest.raises(llm.LLMError) as err:
        llm.chat(SETTINGS, [{"role": "user", "content": "hi"}])
    assert "usage limit" in str(err.value)

"""Web app: a FastAPI backend around the same engine the command line uses.

Run:  uvicorn job_matcher.web:app --reload
Open: http://127.0.0.1:8000

Why background jobs?
Scoring 8 jobs on a free model takes 30-90 seconds. A web request that hangs
that long feels broken and can time out. So POST /api/match starts the work in
a background thread and returns a job id at once; the page then polls
GET /api/jobs/{id} every second to show progress and, finally, the results.
This "start, then poll" pattern is how real apps handle slow work.
"""
import hmac
import logging
import os
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .config import get_settings
from .fetchers import PageNotReadable, fetch_job_description, fetch_jobs, make_client
from .llm import LLMError
from .matcher import filter_by_location, match_jobs, score_job
from .models import Job
from .resume import build_profile, read_resume_text

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Job Matcher")

# Optional password for the deployed site. Without it, anyone who finds the
# link could run matches on YOUR OpenRouter key and use up its free quota.
# Locally you can leave APP_PASSWORD unset and nothing is asked.
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
OPEN_PATHS = {"/", "/api/health"}


@app.middleware("http")
async def require_password(request: Request, call_next):
    """Middleware runs before every request. If a password is set, /api/*
    requests must send it in the X-App-Password header."""
    if APP_PASSWORD and request.url.path not in OPEN_PATHS:
        sent = request.headers.get("x-app-password", "")
        # compare_digest takes the same time whether the guess is close or not,
        # so the password can't be guessed by measuring response times.
        if not hmac.compare_digest(sent.encode(), APP_PASSWORD.encode()):
            return JSONResponse({"detail": "Wrong or missing password."}, status_code=401)
    return await call_next(request)
STATIC = Path(__file__).parent / "static"
UPLOADS = Path(".cache") / "uploads"
MAX_RESUME_BYTES = 5 * 1024 * 1024

# In-memory job store. Fine for one person on one machine; a real multi-user
# deployment would use a database or Redis, because this dict is lost on restart.
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()

CRASH_MESSAGE = ("Something went wrong while matching. The details are in the terminal "
                 "where the server is running; try again, and share them if it keeps happening.")
EXPECTED_ERRORS = (RuntimeError, FileNotFoundError, ValueError, PageNotReadable, LLMError)


def update(job_id: str, **fields) -> None:
    with LOCK:
        JOBS[job_id].update(fields)


def save_upload(resume: UploadFile) -> Path:
    suffix = Path(resume.filename or "").suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "Upload your resume as a PDF or a .txt file.")
    data = resume.file.read()
    if not data:
        raise HTTPException(400, "The resume file is empty.")
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(400, "The resume file is larger than 5 MB.")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)
    return path


def run_match(job_id: str, resume_path: Path, url: str, location: str | None) -> None:
    try:
        settings = get_settings()
        update(job_id, step="Reading your resume")
        profile = build_profile(settings, read_resume_text(str(resume_path)))
        update(job_id, profile=profile.model_dump(), step="Reading the careers page")

        client = make_client()
        jobs, method = fetch_jobs(settings, url, client)
        found = len(jobs)
        jobs = filter_by_location(jobs, location)
        update(job_id, found=found, kept=len(jobs), method=method)
        if not jobs:
            update(job_id, status="done", results=[], step="No roles to score")
            return

        def progress(done, total, title):
            update(job_id, done=done, total=total, step=f"Scoring: {title}")

        results = match_jobs(settings, profile, jobs,
                             describe=lambda j: fetch_job_description(client, j),
                             on_progress=progress)
        update(job_id, status="done", step="Done", done=len(results),
               results=[r.model_dump() for r in results])
    except PageNotReadable as exc:
        # The page couldn't be read: offer the "Paste a job" route on the page.
        update(job_id, status="error", error=str(exc), can_paste=True)
    except EXPECTED_ERRORS as exc:
        update(job_id, status="error", error=str(exc))
    except Exception:  # unexpected: full traceback goes to the terminal, a calm message to the page
        log.exception("Match job %s crashed", job_id)
        update(job_id, status="error", error=CRASH_MESSAGE)
    finally:
        resume_path.unlink(missing_ok=True)


def run_score(job_id: str, resume_path: Path, title: str, description: str) -> None:
    try:
        settings = get_settings()
        update(job_id, step="Reading your resume")
        profile = build_profile(settings, read_resume_text(str(resume_path)))
        update(job_id, profile=profile.model_dump(), step="Scoring the job", total=1)
        result = score_job(settings, profile, Job(title=title, description=description))
        update(job_id, status="done", step="Done", done=1, found=1, kept=1,
               method="pasted job description", results=[result.model_dump()])
    except EXPECTED_ERRORS as exc:
        update(job_id, status="error", error=str(exc))
    except Exception:
        log.exception("Score job %s crashed", job_id)
        update(job_id, status="error", error=CRASH_MESSAGE)
    finally:
        resume_path.unlink(missing_ok=True)


def start(target, *args) -> dict:
    job_id = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[job_id] = {"id": job_id, "status": "running", "step": "Starting",
                        "done": 0, "total": 0, "results": None, "error": None}
    threading.Thread(target=target, args=(job_id, *args), daemon=True).start()
    return {"job_id": job_id}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    """Checks the .env settings without calling the LLM."""
    locked = bool(APP_PASSWORD)
    try:
        settings = get_settings()
        # On a password-protected (deployed) site, don't reveal which model is used.
        return {"ok": True, "model": None if locked else settings.model, "locked": locked}
    except RuntimeError as exc:
        error = "The server isn't set up yet: its API key or model setting is missing." if locked else str(exc)
        return {"ok": False, "error": error, "locked": locked}


@app.post("/api/match")
def start_match(resume: UploadFile = File(...), url: str = Form(...), location: str = Form("")):
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "Paste the full careers page link, starting with https://")
    return start(run_match, save_upload(resume), url, location.strip() or None)


@app.post("/api/score")
def start_score(resume: UploadFile = File(...), description: str = Form(...), title: str = Form("")):
    description = description.strip()
    if len(description) < 50:
        raise HTTPException(400, "Paste the full job description (at least a few lines).")
    title = title.strip() or description.splitlines()[0][:100]
    return start(run_score, save_upload(resume), title, description)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "This run was not found. The server may have restarted.")
        return dict(job)

# Careers Page Job Matcher

Give it your resume and one company's careers page URL. It reads the current
openings and shows only the roles that match your skills and experience, with a
score, the reasons, the skills you're missing, and the apply link.

Uses **free LLMs only** (OpenRouter models ending in `:free`). No LangChain: every
LLM call is a plain HTTP request you can read and debug.

```
Found 41 open roles (read via lever API)

=== 3 matching roles (score >= 50) ===

1. Machine Learning Engineer I (Hyderabad)
   Score: 84/100  ->  strong match
   + Python, scikit-learn and FastAPI match the core requirements
   + 0-2 years required fits a 2026 graduate with an internship
   Missing: Docker, AWS
   Apply: https://jobs.lever.co/...
```

## Setup (one time)

```bash
cd job-matcher
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then fill in:
- `OPENROUTER_API_KEY`: from https://openrouter.ai/keys
- `MODEL`: a free model ID from https://openrouter.ai/models (filter: free), ending in `:free`

Check everything works (no internet or key needed):

```bash
python -m pytest -q
```

## Use it

Score your resume against one job description (fastest first test):

```bash
python -m job_matcher score --resume my_resume.pdf --jd sample_jd.txt
```

Match against a company's careers page:

```bash
python -m job_matcher match --resume my_resume.pdf --url https://jobs.lever.co/COMPANY
python -m job_matcher match --resume my_resume.pdf --url https://company.com/careers --location India
```

Options: `--location` keeps jobs in that place plus remote ones, `--min-score 60`
hides weaker matches, `--no-cache` re-reads your resume, `-v` shows every step.
Full results are saved to `results/` as JSON.

## Web app

```bash
pip install -r requirements.txt
uvicorn job_matcher.web:app --reload
```

Open http://127.0.0.1:8000 in your browser. Upload your resume, paste a careers
page link (or switch to "Paste a job"), and press **Find matching roles**.
Stop the server with Ctrl+C.

How it works: the page sends your resume and link to `POST /api/match`, which
starts the work in a background thread and returns a job id at once. The page
then asks `GET /api/jobs/{id}` every second for progress and results, because
scoring takes about a minute on free models. `GET /api/health` checks your `.env`.

## Deploy (Render, free)

The code goes to GitHub without `.env`, so your key and model never leave your
computer. Render gets them as private settings.

1. **Upload to GitHub:** in VS Code, open Source Control and click **Publish to GitHub**.
2. **Create the service:** on render.com, sign in with GitHub, click
   **New → Blueprint**, and pick this repository. Render reads `render.yaml`.
3. **Fill in the three secrets** Render asks for:
   - `OPENROUTER_API_KEY`: your key
   - `MODEL`: your free model ID
   - `APP_PASSWORD`: a password visitors must type (protects your free quota)
4. Click **Apply**. The first build takes a few minutes; then you get a
   `https://job-matcher-xxxx.onrender.com` link.

Every `git push` redeploys automatically.
Free Render services sleep after about 15
minutes without visitors, so the first visit after a break takes up to a minute.

## How it works

```
resume.pdf ──► read text ──► LLM extracts profile (cached in .cache/)
                                          │
careers URL ──► fetch jobs (fallback chain below)
                                          ▼
              TF-IDF ranks ALL jobs locally (free, instant) ──► keep top 8
                                          ▼
              LLM scores each of the 8 with a rubric ──► code assigns verdict
                                          ▼
                           ranked results + saved JSON
```

**Reading careers pages: the fallback chain** (`fetchers.py`). The LLM never visits
websites; this code fetches pages and hands the text to the LLM.

1. URL is a Greenhouse / Lever / Ashby board → their public JSON API
2. Company page embeds one of those boards → find it in the page source, use the API
3. Page contains schema.org `JobPosting` data (put there for Google Jobs) → parse it
4. Otherwise → send the page's links to the LLM to pick out job postings, then open each shortlisted job page for its description
5. Nothing found → clear error message (JavaScript-only pages come in a later step)

## Code tour: read in this order

| File | What to learn from it |
|---|---|
| `config.py` | Secrets from `.env`, failing early with clear messages, the free-model guard |
| `llm.py` | A raw LLM API call, retries with exponential backoff, fixing messy JSON replies |
| `models.py` | Validating untrusted LLM output with Pydantic; business rules in code |
| `resume.py` | PDF text extraction, structured extraction prompt, caching to save API calls |
| `fetchers.py` | Calling public APIs, HTML parsing, JSON-LD, the fallback chain |
| `matcher.py` | Two-stage retrieval: cheap TF-IDF filter, then expensive LLM scoring |
| `web.py` + `static/index.html` | FastAPI endpoints, file uploads, background jobs with polling, a page built with plain HTML/JS |
| `cli.py` | Commands and arguments, clean error messages instead of stack traces |
| `tests/test_core.py` | Testing without internet by faking the LLM (mocking) |

## Known limits (next steps)

- Pages that load jobs only with JavaScript (e.g. Workday) → add Playwright
- Matching uses TF-IDF keywords → upgrade to embeddings for meaning-based matching
- Package with Docker so it runs the same everywhere
- Free models have per-minute and daily request limits; the app waits and retries on 429

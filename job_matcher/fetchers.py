"""Read job openings from a careers page.

The LLM never visits websites. This code fetches pages; the LLM only reads
text we hand it. We try the most reliable source first and fall back:

  1. Hiring platform API (Greenhouse, Lever, Ashby)  -> clean JSON
  2. Company page that embeds one of those platforms  -> same APIs
  3. schema.org JobPosting data inside the page       -> clean JSON
  4. Links on the page, classified by the LLM          -> titles + links
  5. Nothing readable -> clear error, user pastes a job description
"""
import html as html_lib
import json
import logging
import re
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .config import Settings
from .llm import chat_json
from .models import Job

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (job-matcher; personal job search tool)"}


class PageNotReadable(Exception):
    """No jobs could be read from the page."""


# ---------- helpers ----------

def html_to_text(raw_html: str) -> str:
    text = BeautifulSoup(raw_html or "", "html.parser").get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def detect_platform(url: str) -> tuple[str, str | None]:
    """Return (platform, board_name) for known hiring platforms, else ("generic", None)."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    parts = [p for p in parsed.path.split("/") if p]

    if host.endswith("greenhouse.io"):
        board = parse_qs(parsed.query).get("for", [None])[0]  # embed URLs: ?for=company
        if board:
            return "greenhouse", board
        if parts and parts[0] != "embed":
            return "greenhouse", parts[0]
    if host == "jobs.lever.co" and parts:
        return "lever", parts[0]
    if host == "jobs.ashbyhq.com" and parts:
        return "ashby", parts[0]
    return "generic", None


EMBED_PATTERNS = [
    ("greenhouse", re.compile(r"boards(?:-api)?\.greenhouse\.io/(?:embed/job_board(?:/js)?\?for=|v1/boards/)?([A-Za-z0-9_-]+)")),
    ("greenhouse", re.compile(r"job-boards\.greenhouse\.io/([A-Za-z0-9_-]+)")),
    ("lever", re.compile(r"jobs\.lever\.co/([A-Za-z0-9_-]+)")),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/([A-Za-z0-9_.-]+)")),
]


def find_embedded_board(page_html: str) -> tuple[str, str] | None:
    """Many company careers pages are just a wrapper around Greenhouse/Lever/Ashby.
    Find that board in the page source so we can use its clean API instead."""
    for platform, pattern in EMBED_PATTERNS:
        match = pattern.search(page_html)
        if match and match.group(1) not in ("embed", "v1"):
            return platform, match.group(1)
    return None


# ---------- 1. hiring platform APIs ----------

def fetch_greenhouse(client: httpx.Client, board: str) -> list[Job]:
    resp = client.get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs", params={"content": "true"})
    resp.raise_for_status()
    jobs = []
    for item in resp.json().get("jobs", []):
        jobs.append(Job(
            title=item.get("title", ""),
            location=(item.get("location") or {}).get("name", ""),
            url=item.get("absolute_url", ""),
            department=", ".join(d.get("name", "") for d in item.get("departments", [])),
            # Greenhouse returns HTML that is itself HTML-escaped, so unescape first.
            description=html_to_text(html_lib.unescape(item.get("content", ""))),
        ))
    return jobs


def fetch_lever(client: httpx.Client, company: str) -> list[Job]:
    resp = client.get(f"https://api.lever.co/v0/postings/{company}", params={"mode": "json"})
    resp.raise_for_status()
    jobs = []
    for item in resp.json():
        categories = item.get("categories") or {}
        parts = [item.get("descriptionPlain") or html_to_text(item.get("description", ""))]
        for section in item.get("lists", []):  # "Requirements", "Responsibilities" etc.
            parts.append(f"{section.get('text', '')}:\n{html_to_text(section.get('content', ''))}")
        jobs.append(Job(
            title=item.get("text", ""),
            location=categories.get("location", "") or "",
            url=item.get("hostedUrl", ""),
            department=categories.get("team", "") or "",
            description="\n".join(parts),
        ))
    return jobs


def fetch_ashby(client: httpx.Client, board: str) -> list[Job]:
    resp = client.get(f"https://api.ashbyhq.com/posting-api/job-board/{board}")
    resp.raise_for_status()
    jobs = []
    for item in resp.json().get("jobs", []):
        jobs.append(Job(
            title=item.get("title", ""),
            location=item.get("location", "") or "",
            url=item.get("jobUrl", ""),
            department=item.get("department", "") or "",
            description=item.get("descriptionPlain") or html_to_text(item.get("descriptionHtml", "")),
        ))
    return jobs


PLATFORM_FETCHERS = {"greenhouse": fetch_greenhouse, "lever": fetch_lever, "ashby": fetch_ashby}


# ---------- 3. schema.org JobPosting ----------

def _iter_jsonld_items(data):
    if isinstance(data, list):
        for entry in data:
            yield from _iter_jsonld_items(entry)
    elif isinstance(data, dict):
        if "@graph" in data:
            yield from _iter_jsonld_items(data["@graph"])
        yield data


def _jsonld_location(item: dict) -> str:
    if item.get("jobLocationType") == "TELECOMMUTE":
        return "Remote"
    locations = item.get("jobLocation") or []
    if isinstance(locations, dict):
        locations = [locations]
    names = []
    for loc in locations:
        address = (loc or {}).get("address") or {}
        if isinstance(address, str):
            names.append(address)
            continue
        country = address.get("addressCountry", "")
        if isinstance(country, dict):
            country = country.get("name", "")
        parts = [address.get("addressLocality", ""), address.get("addressRegion", ""), country]
        names.append(", ".join(p for p in parts if p))
    return " | ".join(n for n in names if n)


def parse_jsonld_jobs(page_html: str, base_url: str) -> list[Job]:
    soup = BeautifulSoup(page_html, "html.parser")
    jobs = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except json.JSONDecodeError:
            continue
        for item in _iter_jsonld_items(data):
            kind = item.get("@type")
            if kind == "JobPosting" or (isinstance(kind, list) and "JobPosting" in kind):
                jobs.append(Job(
                    title=item.get("title", ""),
                    location=_jsonld_location(item),
                    url=urljoin(base_url, item.get("url", "") or base_url),
                    description=html_to_text(item.get("description", "")),
                ))
    return [j for j in jobs if j.title]


# ---------- 4. links classified by the LLM ----------

LINKS_SYSTEM = """You look at links from a company careers page and pick the ones that
are individual job postings (one specific open role each).
Ignore navigation, blog, benefits, login, social media and "all jobs" links.
Return ONLY JSON: {"jobs": [{"title": string, "url": string, "location": string}]}
Use "" for an unknown location. If there are no job postings, return {"jobs": []}."""


def extract_jobs_with_llm(settings: Settings, page_html: str, base_url: str) -> list[Job]:
    soup = BeautifulSoup(page_html, "html.parser")
    links, seen = [], set()
    for a in soup.find_all("a", href=True):
        text = " ".join(a.get_text(" ").split())
        href = urljoin(base_url, a["href"])
        if 3 <= len(text) <= 150 and href not in seen and not href.startswith(("mailto:", "tel:", "javascript:")):
            seen.add(href)
            links.append(f"{text} -> {href}")
    if not links:
        return []

    data = chat_json(settings, LINKS_SYSTEM, "\n".join(links[:300]))
    items = data.get("jobs", []) if isinstance(data, dict) else []
    return [Job(**{k: str(i.get(k, "") or "") for k in ("title", "url", "location")})
            for i in items if isinstance(i, dict) and i.get("title")]


def fetch_job_description(client: httpx.Client, job: Job) -> str:
    """For jobs found by link only: open the job page and read its description."""
    try:
        resp = client.get(job.url)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("Could not open %s: %s", job.url, exc)
        return ""
    structured = parse_jsonld_jobs(resp.text, str(resp.url))
    if structured and structured[0].description:
        return structured[0].description
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return html_to_text(str(soup))[:6000]


# ---------- the fallback chain ----------

def make_client() -> httpx.Client:
    return httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True)


def fetch_jobs(settings: Settings, url: str, client: httpx.Client | None = None) -> tuple[list[Job], str]:
    """Return (jobs, method_used). Raises PageNotReadable if nothing works."""
    client = client or make_client()

    platform, board = detect_platform(url)
    if platform in PLATFORM_FETCHERS:
        log.info("Detected %s board '%s', using its public API", platform, board)
        return PLATFORM_FETCHERS[platform](client, board), f"{platform} API"

    log.info("Fetching page %s", url)
    try:
        resp = client.get(url)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise PageNotReadable(f"The site returned {exc.response.status_code} for this page.") from exc
    except httpx.HTTPError as exc:
        raise PageNotReadable(f"Could not reach the page: {exc}") from exc
    page_html, final_url = resp.text, str(resp.url)

    embedded = find_embedded_board(page_html)
    if embedded:
        platform, board = embedded
        log.info("Page embeds a %s board '%s', using its public API", platform, board)
        try:
            jobs = PLATFORM_FETCHERS[platform](client, board)
            if jobs:
                return jobs, f"embedded {platform} API"
        except httpx.HTTPError as exc:
            log.warning("Embedded board lookup failed (%s); trying other methods", exc)

    jobs = parse_jsonld_jobs(page_html, final_url)
    if jobs:
        return jobs, "JobPosting data in page"

    jobs = extract_jobs_with_llm(settings, page_html, final_url)
    if jobs:
        return jobs, "page links read by LLM"

    raise PageNotReadable(
        "No job openings found on this page. It may load jobs with JavaScript "
        "(support coming in a later step) or need a login. "
        "Try the company's 'View all jobs' link, or use the score command with a pasted job description."
    )

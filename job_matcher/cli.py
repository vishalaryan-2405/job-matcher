"""Command-line entry point.

  python -m job_matcher match --resume resume.pdf --url https://jobs.lever.co/company
  python -m job_matcher score --resume resume.pdf --jd job.txt
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from .config import get_settings
from .fetchers import PageNotReadable, fetch_job_description, fetch_jobs, make_client
from .llm import LLMError
from .matcher import filter_by_location, match_jobs, score_job
from .models import Job, MatchResult
from .resume import build_profile, read_resume_text

log = logging.getLogger("job_matcher")


def print_result(rank: int, result: MatchResult) -> None:
    job = result.job
    print(f"\n{rank}. {job.title}" + (f" ({job.location})" if job.location else ""))
    print(f"   Score: {result.score}/100  ->  {result.verdict}")
    for reason in result.reasons:
        print(f"   + {reason}")
    if result.missing_skills:
        print(f"   Missing: {', '.join(result.missing_skills)}")
    if job.url:
        print(f"   Apply: {job.url}")


def save_results(name: str, results: list[MatchResult]) -> Path:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}_{datetime.now():%Y%m%d_%H%M}.json"
    path.write_text(json.dumps([r.model_dump() for r in results], indent=2), encoding="utf-8")
    return path


def cmd_match(args) -> int:
    settings = get_settings()
    profile = build_profile(settings, read_resume_text(args.resume), use_cache=not args.no_cache)
    print(f"Profile: {profile.name or 'candidate'} | {profile.years_experience} yrs | "
          f"{', '.join(profile.skills[:8])}...")

    client = make_client()
    jobs, method = fetch_jobs(settings, args.url, client)
    print(f"Found {len(jobs)} open roles (read via {method})")

    jobs = filter_by_location(jobs, args.location)
    if args.location:
        print(f"{len(jobs)} roles left after location filter '{args.location}'")
    if not jobs:
        print("No jobs to score.")
        return 0

    results = match_jobs(settings, profile, jobs, describe=lambda j: fetch_job_description(client, j))
    shown = [r for r in results if r.score >= args.min_score]

    print(f"\n=== {len(shown)} matching roles (score >= {args.min_score}) ===")
    for rank, result in enumerate(shown, start=1):
        print_result(rank, result)

    company = (urlparse(args.url).netloc + urlparse(args.url).path).replace("/", "_").strip("_")
    print(f"\nFull results saved to {save_results(company[:60], results)}")
    return 0


def cmd_score(args) -> int:
    settings = get_settings()
    profile = build_profile(settings, read_resume_text(args.resume), use_cache=not args.no_cache)
    description = Path(args.jd).read_text(encoding="utf-8")
    title = args.title or description.strip().splitlines()[0][:100]
    result = score_job(settings, profile, Job(title=title, description=description))
    print_result(1, result)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="job_matcher", description="Match your resume to a company's open jobs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show detailed logs")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("match", help="match against a careers page")
    m.add_argument("--resume", required=True, help="path to your resume (.pdf or .txt)")
    m.add_argument("--url", required=True, help="company careers page URL")
    m.add_argument("--location", help="keep only jobs in this location (remote jobs are kept)")
    m.add_argument("--min-score", type=int, default=50, help="hide results below this score")
    m.add_argument("--no-cache", action="store_true", help="re-parse the resume")
    m.set_defaults(func=cmd_match)

    s = sub.add_parser("score", help="score against one pasted job description")
    s.add_argument("--resume", required=True)
    s.add_argument("--jd", required=True, help="text file with the job description")
    s.add_argument("--title", help="job title (default: first line of the file)")
    s.add_argument("--no-cache", action="store_true")
    s.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (RuntimeError, FileNotFoundError, ValueError, PageNotReadable, LLMError) as exc:
        # Expected problems: show a clean message, not a stack trace.
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

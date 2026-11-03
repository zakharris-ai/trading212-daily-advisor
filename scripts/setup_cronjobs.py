#!/usr/bin/env python3
"""Create or update the cron-job.org triggers for GitHub Actions (09:00 digest + weekday intraday watch).

Uses only the Python standard library and reads credentials from `.env` (or environment variables) without ever
printing secret values:
  * GITHUB_DISPATCH_TOKEN  Fine-grained GitHub token scoped to this repository with `Actions: Read and write`
  * CRONJOB_API_KEY        API key from https://console.cron-job.org/settings

Idempotent: if a job with the same GitHub workflow dispatch URL already exists on cron-job.org, it is updated in place
(`PATCH /jobs/<jobId>`) instead of creating a duplicate; otherwise it is created (`PUT /jobs`).

Usage:
  python scripts/setup_cronjobs.py                  # verify token + create/update both cron-job.org jobs
  python scripts/setup_cronjobs.py --only watch     # create/update only the intraday_watch.yml job
  python scripts/setup_cronjobs.py --dry-run        # print redacted payloads without calling external APIs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
GITHUB_API = "https://api.github.com"
CRONJOB_API = "https://api.cron-job.org"
REQUEST_METHOD_POST = 1


class SetupError(RuntimeError):
    """User-facing configuration or API error (never contains raw tokens)."""


def parse_github_repo(remote_url: str) -> Optional[str]:
    """Extracts 'owner/repo' from an SSH or HTTPS GitHub remote URL."""
    s = (remote_url or "").strip()
    m = re.search(r"github\.com[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$", s)
    return m.group(1) if m else None


def detect_repo(explicit: Optional[str] = None, cwd: Optional[Path] = None) -> str:
    """Resolves 'owner/repo' from --repo, GITHUB_REPOSITORY, or `git remote get-url origin`."""
    for candidate in (explicit, os.getenv("GITHUB_REPOSITORY")):
        if candidate and re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", candidate.strip()):
            return candidate.strip()
    try:
        out = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=str(cwd or REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        parsed = parse_github_repo(out)
        if parsed:
            return parsed
    except Exception:
        pass
    raise SetupError("Could not detect GitHub 'owner/repo'. Pass --repo owner/repo or set GITHUB_REPOSITORY.")


def load_env_file(path: Path, environ: Optional[Dict[str, str]] = None) -> None:
    """Minimal `.env` loader (does not overwrite variables already set in `environ`)."""
    target = environ if environ is not None else os.environ
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key and key not in target:
            target[key] = val


def resolve_credentials(environ: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """Returns `(github_dispatch_token, cronjob_api_key)` or raises `SetupError` naming what is missing."""
    env = environ if environ is not None else os.environ
    gh_token = (env.get("GITHUB_DISPATCH_TOKEN") or env.get("GH_DISPATCH_TOKEN") or "").strip()
    cj_key = (env.get("CRONJOB_API_KEY") or "").strip()
    missing = []
    if not gh_token:
        missing.append("GITHUB_DISPATCH_TOKEN")
    if not cj_key:
        missing.append("CRONJOB_API_KEY")
    if missing:
        raise SetupError(
            f"Missing {', '.join(missing)} in {REPO_ROOT / '.env'} (or ~/.env). "
            "Add them safely via hidden terminal input and re-run."
        )
    return gh_token, cj_key


def github_dispatch_headers(gh_token: str) -> Dict[str, str]:
    """Headers stored on cron-job.org (and used when verifying workflow_dispatch against GitHub)."""
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {gh_token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }


def build_job_specs(repo: str, gh_token: str, ref: str = "main", only: str = "both") -> List[Tuple[str, Dict[str, Any]]]:
    """Builds the cron-job.org `DetailedJob` payloads for the 09:00 digest and weekday intraday watch."""
    body = json.dumps({"ref": ref}, separators=(",", ":"))
    headers = github_dispatch_headers(gh_token)

    digest_job: Dict[str, Any] = {
        "title": "Trading 212 digest 09:00 London",
        "url": f"{GITHUB_API}/repos/{repo}/actions/workflows/daily_trading_digest.yml/dispatches",
        "enabled": True,
        "saveResponses": False,
        "requestMethod": REQUEST_METHOD_POST,
        "schedule": {
            "timezone": "Europe/London",
            "expiresAt": 0,
            "hours": [9],
            "mdays": [-1],
            "minutes": [0],
            "months": [-1],
            "wdays": [-1],
        },
        "notification": {
            "onFailure": True,
            "onFailureCount": 1,
            "onSuccess": False,
            "onDisable": True,
        },
        "extendedData": {
            "headers": dict(headers),
            "body": body,
        },
    }

    watch_job: Dict[str, Any] = {
        "title": "Trading 212 intraday watch",
        "url": f"{GITHUB_API}/repos/{repo}/actions/workflows/intraday_watch.yml/dispatches",
        "enabled": True,
        "saveResponses": False,
        "requestMethod": REQUEST_METHOD_POST,
        "schedule": {
            "timezone": "Europe/London",
            "expiresAt": 0,
            "hours": list(range(8, 21)),  # 08:00 through 20:30 London
            "mdays": [-1],
            "minutes": [0, 30],
            "months": [-1],
            "wdays": [1, 2, 3, 4, 5],     # Monday (1) through Friday (5)
        },
        "notification": {
            "onFailure": True,
            "onFailureCount": 2,
            "onSuccess": False,
            "onDisable": True,
        },
        "extendedData": {
            "headers": dict(headers),
            "body": body,
        },
    }

    specs: List[Tuple[str, Dict[str, Any]]] = []
    if only in ("both", "digest"):
        specs.append(("digest", digest_job))
    if only in ("both", "watch"):
        specs.append(("watch", watch_job))
    return specs


def redact_job_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copies a job spec with Authorization header value redacted for safe terminal printing."""
    out = json.loads(json.dumps(spec))
    hdrs = ((out.get("extendedData") or {}).get("headers")) or {}
    if "Authorization" in hdrs:
        hdrs["Authorization"] = "Bearer <REDACTED>"
    return out


HttpFn = Callable[[str, str, Dict[str, str], Optional[bytes]], Tuple[int, str]]


def _default_http(method: str, url: str, headers: Dict[str, str], body: Optional[bytes] = None) -> Tuple[int, str]:
    req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return int(e.code), raw
    except urllib.error.URLError as e:
        raise SetupError(f"Network error contacting {url.split('/')[2]}: {e.reason}") from None


def verify_github_token(
    repo: str,
    gh_token: str,
    ref: str = "main",
    verify_dispatch: bool = True,
    http: HttpFn = _default_http,
) -> List[str]:
    """Verifies repo workflow visibility and (optionally) `Actions: Read and write` via a live dispatch of watch."""
    headers = github_dispatch_headers(gh_token)
    status, raw = http("GET", f"{GITHUB_API}/repos/{repo}/actions/workflows", headers, None)
    if status == 401:
        raise SetupError("GitHub returned 401 Bad credentials: GITHUB_DISPATCH_TOKEN is invalid or expired.")
    if status == 403:
        raise SetupError(
            f"GitHub returned 403 Forbidden on {repo}: grant the fine-grained token 'Actions: Read and write' on {repo}."
        )
    if status == 404:
        raise SetupError(
            f"GitHub returned 404 Not Found on {repo}: make sure the token has repository access to '{repo}'."
        )
    if status != 200:
        raise SetupError(f"Unexpected GitHub HTTP {status} listing workflows on {repo}.")

    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        payload = {}
    paths = [str(w.get("path") or "") for w in (payload.get("workflows") or []) if isinstance(w, dict)]

    if verify_dispatch:
        body = json.dumps({"ref": ref}, separators=(",", ":")).encode("utf-8")
        d_status, _ = http(
            "POST",
            f"{GITHUB_API}/repos/{repo}/actions/workflows/intraday_watch.yml/dispatches",
            headers,
            body,
        )
        if d_status == 403:
            raise SetupError(
                "GitHub token can read workflows but returned 403 on dispatch: change repository permission "
                "'Actions' from Read-only to 'Read and write'."
            )
        if d_status == 422:
            raise SetupError(f"GitHub returned 422 dispatching intraday_watch.yml on ref '{ref}'.")
        if d_status != 204:
            raise SetupError(f"Unexpected GitHub HTTP {d_status} testing intraday_watch.yml dispatch.")
    return paths


def _norm_url(url: Any) -> str:
    return str(url or "").strip().rstrip("/")


def upsert_cronjobs(
    specs: List[Tuple[str, Dict[str, Any]]],
    cronjob_api_key: str,
    http: HttpFn = _default_http,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> List[Dict[str, Any]]:
    """Lists existing cron-job.org jobs and creates (`PUT /jobs`) or updates (`PATCH /jobs/<id>`) each target URL."""
    headers = {
        "Authorization": f"Bearer {cronjob_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    status, raw = http("GET", f"{CRONJOB_API}/jobs", headers, None)
    if status == 401:
        raise SetupError("cron-job.org returned 401 Unauthorized: CRONJOB_API_KEY is invalid or revoked.")
    if status == 403:
        raise SetupError("cron-job.org returned 403 Forbidden: check IP/origin restrictions on your CRONJOB_API_KEY.")
    if status != 200:
        raise SetupError(f"cron-job.org returned HTTP {status} when listing existing jobs.")

    try:
        existing_jobs = (json.loads(raw or "{}").get("jobs")) or []
    except json.JSONDecodeError as e:
        raise SetupError(f"Malformed JSON from cron-job.org /jobs: {e}") from None

    by_url: Dict[str, int] = {}
    for j in existing_jobs:
        if isinstance(j, dict) and j.get("jobId") is not None and j.get("url"):
            by_url[_norm_url(j["url"])] = int(j["jobId"])

    results: List[Dict[str, Any]] = []
    for idx, (kind, spec) in enumerate(specs):
        if idx > 0:
            sleep_fn(1.5)  # cron-job.org PUT /jobs rate limit is 1 req/s (5 req/min)
        target_url = _norm_url(spec["url"])
        body = json.dumps({"job": spec}).encode("utf-8")
        if target_url in by_url:
            job_id = by_url[target_url]
            u_status, _ = http("PATCH", f"{CRONJOB_API}/jobs/{job_id}", headers, body)
            if u_status != 200:
                raise SetupError(f"cron-job.org returned HTTP {u_status} updating job #{job_id} ({spec['title']}).")
            results.append({"kind": kind, "action": "updated", "jobId": job_id, "title": spec["title"], "url": spec["url"]})
        else:
            c_status, c_raw = http("PUT", f"{CRONJOB_API}/jobs", headers, body)
            if c_status != 200:
                raise SetupError(f"cron-job.org returned HTTP {c_status} creating job '{spec['title']}'.")
            try:
                job_id = int(json.loads(c_raw or "{}").get("jobId", 0))
            except (ValueError, TypeError, json.JSONDecodeError):
                job_id = 0
            results.append({"kind": kind, "action": "created", "jobId": job_id, "title": spec["title"], "url": spec["url"]})
    return results


def dispatch_digest_if_unsent(repo: str, gh_token: str, ref: str = "main", http: HttpFn = _default_http) -> bool:
    """Triggers `daily_trading_digest.yml` once if today's London digest hasn't been sent yet (dedupe-safe)."""
    marker = REPO_ROOT / "data" / "last_email_sent.json"
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        today_london = datetime.now(ZoneInfo("Europe/London")).date().isoformat()
        hour_london = datetime.now(ZoneInfo("Europe/London")).hour
    except Exception:
        return False
    if hour_london < 9:
        return False
    if marker.is_file():
        try:
            sent_date = json.loads(marker.read_text(encoding="utf-8")).get("date")
            if sent_date == today_london:
                return False
        except Exception:
            pass
    body = json.dumps({"ref": ref}, separators=(",", ":")).encode("utf-8")
    status, _ = http(
        "POST",
        f"{GITHUB_API}/repos/{repo}/actions/workflows/daily_trading_digest.yml/dispatches",
        github_dispatch_headers(gh_token),
        body,
    )
    return status == 204


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create or update cron-job.org triggers for GitHub Actions.")
    parser.add_argument("--repo", help="GitHub owner/repo (default: auto-detected from git remote origin)")
    parser.add_argument("--ref", default="main", help="Git branch ref to dispatch (default: main)")
    parser.add_argument("--only", choices=("both", "digest", "watch"), default="both",
                        help="Which cron-job.org job(s) to configure (default: both)")
    parser.add_argument("--no-verify-dispatch", action="store_true",
                        help="Skip live test dispatch of intraday_watch.yml against GitHub")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print redacted job payloads without calling GitHub or cron-job.org")
    args = parser.parse_args(argv)

    load_env_file(REPO_ROOT / ".env")
    load_env_file(Path.home() / ".env")

    try:
        repo = detect_repo(args.repo)
        if args.dry_run:
            specs = build_job_specs(repo, "<GITHUB_DISPATCH_TOKEN>", ref=args.ref, only=args.only)
            for _, spec in specs:
                print(json.dumps({"job": redact_job_spec(spec)}, indent=2))
            return 0

        gh_token, cj_key = resolve_credentials()
        verify_github_token(repo, gh_token, ref=args.ref, verify_dispatch=not args.no_verify_dispatch)
        print(f"Verified GitHub token on {repo} (Actions: Read and write confirmed).")

        specs = build_job_specs(repo, gh_token, ref=args.ref, only=args.only)
        outcomes = upsert_cronjobs(specs, cj_key)
        for item in outcomes:
            print(f"cron-job.org {item['action'].upper()}: #{item['jobId']} — {item['title']} -> {item['url']}")

        if args.only in ("both", "digest") and dispatch_digest_if_unsent(repo, gh_token, ref=args.ref):
            print("Also triggered today's daily_trading_digest.yml (today's 09:00 digest had not been sent yet).")
        return 0
    except SetupError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

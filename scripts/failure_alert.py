#!/usr/bin/env python3
"""Emails a failure alert when the daily digest workflow fails, so a missed digest is never silent.

Run by the `if: failure()` steps of .github/workflows/daily_trading_digest.yml. The workflow sends at
most one alert per London day (it remembers a sent alert with an Actions cache key), and later ticks
keep retrying the digest itself. Standard library only, so the alert still works when the dependency
install is the step that failed.

Environment (all optional; missing SMTP settings just skip the email):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAILS  same secrets the digest uses
  OUTCOME_TESTS, OUTCOME_DIGEST, OUTCOME_COMMIT             `steps.<id>.outcome` of the workflow steps
  RUN_URL, TRIGGER                                          link to this run's logs, and the event name
  GITHUB_OUTPUT                                             receives `sent=true` after a successful send
"""

import json
import os
import smtplib
import sys
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from config import _safe_int  # GEMINI.md: numeric env vars are parsed with config._safe_int
except Exception:  # config.py itself may be what broke this run; the alert path must stay independent

    def _safe_int(val: Any, default: int) -> int:
        """Same semantics as config._safe_int (used only when config.py cannot be imported)."""
        if val is None:
            return default
        try:
            s = str(val).strip()
            if not s or s.lower() in ("true", "false", "none", "null"):
                return default
            return int(s)
        except (ValueError, TypeError):
            return default


LONDON_TZ = ZoneInfo("Europe/London")
SENT_MARKER = REPO_ROOT / "data" / "last_email_sent.json"
ALERT_MARKER_DIR = REPO_ROOT / ".failure-alert"
LOG_TAIL_LINES = 40

# (env var holding `steps.<id>.outcome`, plain-English stage, log file that step writes)
STAGES: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("OUTCOME_TESTS", "the offline unit tests (a safety stop before any email)", "logs/unit_tests.log"),
    ("OUTCOME_DIGEST", "the portfolio analysis and email dispatch (main.py)", "logs/digest_run.log"),
    ("OUTCOME_COMMIT", "saving today's snapshot back to GitHub", None),
)
SETUP_STAGE: Tuple[str, str, Optional[str]] = (
    "SETUP", "workflow setup (checkout, Python, dependency install or the gatekeeper)", None,
)


def failed_stage(env: Mapping[str, str]) -> Tuple[str, str, Optional[str]]:
    """The first stage whose outcome is 'failure', as (key, plain-English label, log file or None)."""
    for stage in STAGES:
        if (env.get(stage[0]) or "").strip().lower() == "failure":
            return stage
    return SETUP_STAGE


def london_now() -> datetime:
    return datetime.now(LONDON_TZ)


def digest_sent_time(today: str, marker: Optional[Path] = None) -> Optional[str]:
    """The London send time from this checkout's sent-marker when it is dated today, else None."""
    try:
        data = json.loads((marker or SENT_MARKER).read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and data.get("date") == today:
        return str(data.get("time") or "earlier today")
    return None


def log_tail(path: Optional[Path], max_lines: int = LOG_TAIL_LINES) -> str:
    """The last `max_lines` lines of a log file (each capped at 300 chars), or '' when unavailable."""
    if path is None:
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(line[:300] for line in lines[-max_lines:])


def build_alert(env: Mapping[str, str], now: datetime, sent_time: Optional[str], tail: str) -> Tuple[str, str]:
    """Subject and plain-text body of the failure alert."""
    key, stage, _ = failed_stage(env)
    day = now.strftime("%a %d %b %Y")
    run_url = (env.get("RUN_URL") or "").strip() or "(run link unavailable)"
    trigger = (env.get("TRIGGER") or "").strip() or "unknown"

    if sent_time and key == "OUTCOME_COMMIT":
        subject = f"⚠️ Trading digest: emailed, but saving the snapshot failed ({day})"
        status = (
            f"Today's digest email WAS sent (at {sent_time} London), but saving the snapshot back to GitHub "
            "then failed. The sent-marker may not have reached GitHub, so a later run could email a duplicate."
        )
    elif sent_time:
        subject = f"⚠️ Trading digest: a run failed after today's email was sent ({day})"
        status = (
            f"Today's digest email was sent at {sent_time} London. This run then failed during {stage}, "
            "so today's snapshot may not have been saved."
        )
    else:
        subject = f"⚠️ Trading digest FAILED — no email sent by this run ({day})"
        status = f"This run failed during {stage}, so it did not email today's digest."

    lines: List[str] = [
        "Your automated daily trading digest hit a problem on GitHub Actions.",
        "",
        status,
        "",
        f"Trigger: {trigger}   |   Time: {now.strftime('%H:%M')} London",
        f"Full run log: {run_url}",
        "",
        "What happens next:",
        "- GitHub's later scheduled runs today retry automatically (they can be a few hours late).",
        "- The once-a-day sent-marker still applies, so you will get at most one digest today.",
        "- You get at most one of these alerts per day.",
        "",
        "Common causes and fixes:",
        "- 'Trading 212 positions unavailable': the API key may have expired or been revoked. Create a new",
        "  key in the Trading 212 app (Settings -> API) and update the T212_* secrets on GitHub.",
        "- Trading 212 or market data briefly unavailable: no action needed; a later run will retry.",
        "- 'Email NOT sent': the SMTP app password changed. Update SMTP_PASS on GitHub.",
        "- Unit tests failed: a code change broke a safety check. No recommendations are sent until it is fixed.",
        "",
        "To send today's digest by hand once fixed: GitHub -> Actions -> Daily Trading Intelligence Digest -> Run workflow.",
    ]
    if tail:
        lines += ["", f"Last {LOG_TAIL_LINES} log lines from the failed step:", "-" * 60, tail, "-" * 60]
    return subject, "\n".join(lines) + "\n"


def smtp_settings(env: Mapping[str, str]) -> Dict[str, Any]:
    """SMTP settings with the same defaults and recipient parsing as config.py."""
    return {
        "host": (env.get("SMTP_HOST") or "").strip() or "smtp.gmail.com",
        "port": _safe_int(env.get("SMTP_PORT"), 587),
        "user": (env.get("SMTP_USER") or "").strip(),
        "password": env.get("SMTP_PASS") or "",
        "recipients": [e.strip() for e in (env.get("ALERT_EMAILS") or "").split(",") if e.strip()],
    }


def send_alert(subject: str, body: str, settings: Dict[str, Any]) -> bool:
    """Sends the alert (STARTTLS, like mailer.send_email_digest). Returns True on success; never raises."""
    if not settings["user"] or not settings["password"] or not settings["recipients"]:
        print("Failure alert skipped: SMTP_USER, SMTP_PASS or ALERT_EMAILS is not configured.")
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"Trading 212 Assistant <{settings['user']}>"
    msg["To"] = ", ".join(settings["recipients"])
    msg.set_content(body)
    try:
        server = smtplib.SMTP(settings["host"], settings["port"], timeout=20)
        try:
            server.ehlo()
            server.starttls()
            server.login(settings["user"], settings["password"])
            server.send_message(msg)
        finally:
            try:
                server.quit()
            except Exception:
                pass
    except Exception as e:
        print(f"Failure alert could not be emailed: {e}")
        return False
    return True


def record_sent(env: Mapping[str, str], now: datetime) -> None:
    """Flags a delivered alert: `sent=true` step output plus the marker directory the cache step saves."""
    ALERT_MARKER_DIR.mkdir(parents=True, exist_ok=True)
    (ALERT_MARKER_DIR / "sent").write_text(now.isoformat() + "\n", encoding="utf-8")
    output_path = (env.get("GITHUB_OUTPUT") or "").strip()
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("sent=true\n")


def main(env: Optional[Mapping[str, str]] = None) -> int:
    """Builds and sends the alert. Always exits 0: the job has already failed, so no extra noise."""
    env = os.environ if env is None else env
    now = london_now()
    _, stage, log_file = failed_stage(env)
    sent_time = digest_sent_time(now.strftime("%Y-%m-%d"))
    tail = log_tail(REPO_ROOT / log_file if log_file else None)
    subject, body = build_alert(env, now, sent_time, tail)
    print(f"Failed stage: {stage}")
    if send_alert(subject, body, smtp_settings(env)):
        record_sent(env, now)
        print(f"Failure alert emailed: {subject}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

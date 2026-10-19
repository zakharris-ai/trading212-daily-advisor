import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Offline tests for scripts/failure_alert.py (the workflow's `if: failure()` alert): SMTP is always mocked.

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "failure_alert.py")
_spec = importlib.util.spec_from_file_location("failure_alert", _PATH)
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)

NOW = datetime(2026, 10, 19, 9, 3, tzinfo=ZoneInfo("Europe/London"))
SMTP_ENV = {
    "SMTP_HOST": "smtp.example.com",
    "SMTP_PORT": "587",
    "SMTP_USER": "bot@example.com",
    "SMTP_PASS": "app-password",
    "ALERT_EMAILS": "me@example.com, partner@example.com",
}


class FailedStageTests(unittest.TestCase):
    def test_first_failed_step_wins_and_setup_is_the_default(self):
        self.assertEqual(fa.failed_stage({"OUTCOME_TESTS": "failure"})[0], "OUTCOME_TESTS")
        self.assertEqual(fa.failed_stage({"OUTCOME_TESTS": "success", "OUTCOME_DIGEST": "failure"})[0], "OUTCOME_DIGEST")
        self.assertEqual(fa.failed_stage({"OUTCOME_DIGEST": "success", "OUTCOME_COMMIT": "Failure"})[0], "OUTCOME_COMMIT")
        self.assertEqual(fa.failed_stage({"OUTCOME_TESTS": "skipped", "OUTCOME_DIGEST": ""})[0], "SETUP")
        self.assertEqual(fa.failed_stage({})[0], "SETUP")

    def test_log_files_are_mapped_per_stage(self):
        self.assertEqual(fa.failed_stage({"OUTCOME_TESTS": "failure"})[2], "logs/unit_tests.log")
        self.assertEqual(fa.failed_stage({"OUTCOME_DIGEST": "failure"})[2], "logs/digest_run.log")
        self.assertIsNone(fa.failed_stage({})[2])


class SentMarkerAndLogTests(unittest.TestCase):
    def test_sent_time_only_for_todays_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "last_email_sent.json"
            marker.write_text(json.dumps({"date": "2026-10-19", "time": "09:02:11"}))
            self.assertEqual(fa.digest_sent_time("2026-10-19", marker), "09:02:11")
            self.assertIsNone(fa.digest_sent_time("2026-10-20", marker))
            marker.write_text("not json")
            self.assertIsNone(fa.digest_sent_time("2026-10-19", marker))
            self.assertIsNone(fa.digest_sent_time("2026-10-19", Path(tmp) / "missing.json"))

    def test_log_tail_keeps_the_last_lines_and_caps_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text("\n".join(f"line {i}" for i in range(100)) + "\n" + "x" * 500 + "\n")
            tail = fa.log_tail(log, max_lines=3).splitlines()
            self.assertEqual(tail[:2], ["line 98", "line 99"])
            self.assertEqual(len(tail[2]), 300)
            self.assertEqual(fa.log_tail(Path(tmp) / "missing.log"), "")
            self.assertEqual(fa.log_tail(None), "")


class BuildAlertTests(unittest.TestCase):
    ENV = {"RUN_URL": "https://github.com/o/r/actions/runs/1", "TRIGGER": "workflow_dispatch"}

    def test_failed_before_sending(self):
        env = dict(self.ENV, OUTCOME_DIGEST="failure")
        subject, body = fa.build_alert(env, NOW, None, "ERROR main: Trading 212 positions unavailable")
        self.assertIn("FAILED", subject)
        self.assertIn("Mon 19 Oct 2026", subject)
        self.assertIn("did not email today's digest", body)
        self.assertIn("main.py", body)
        self.assertIn("https://github.com/o/r/actions/runs/1", body)
        self.assertIn("Trading 212 positions unavailable", body)
        self.assertIn("at most one of these alerts per day", body)

    def test_emailed_but_snapshot_not_saved_warns_about_duplicates(self):
        subject, body = fa.build_alert(dict(self.ENV, OUTCOME_COMMIT="failure"), NOW, "09:02:11", "")
        self.assertIn("emailed, but saving the snapshot failed", subject)
        self.assertIn("WAS sent (at 09:02:11 London)", body)
        self.assertIn("duplicate", body)
        self.assertNotIn("log lines from the failed step", body)

    def test_failure_after_todays_email_already_went_out(self):
        subject, body = fa.build_alert(dict(self.ENV, OUTCOME_DIGEST="failure"), NOW, "09:02:11", "")
        self.assertIn("after today's email was sent", subject)
        self.assertIn("was sent at 09:02:11 London", body)


class SmtpTests(unittest.TestCase):
    def test_settings_parse_port_safely_and_split_recipients(self):
        s = fa.smtp_settings(dict(SMTP_ENV, SMTP_PORT="true"))
        self.assertEqual(s["port"], 587)  # a boolean-ish secret never crashes int() (GEMINI.md)
        self.assertEqual(s["recipients"], ["me@example.com", "partner@example.com"])
        self.assertEqual(fa.smtp_settings({})["host"], "smtp.gmail.com")

    def test_unconfigured_smtp_skips_without_connecting(self):
        with mock.patch.object(fa.smtplib, "SMTP") as smtp:
            self.assertFalse(fa.send_alert("s", "b", fa.smtp_settings({})))
        smtp.assert_not_called()

    def test_send_uses_starttls_login_and_quits(self):
        with mock.patch.object(fa.smtplib, "SMTP") as smtp:
            self.assertTrue(fa.send_alert("subject", "body", fa.smtp_settings(SMTP_ENV)))
        smtp.assert_called_once_with("smtp.example.com", 587, timeout=20)
        server = smtp.return_value
        server.starttls.assert_called_once()
        server.login.assert_called_once_with("bot@example.com", "app-password")
        sent = server.send_message.call_args[0][0]
        self.assertEqual(sent["To"], "me@example.com, partner@example.com")
        server.quit.assert_called_once()

    def test_smtp_errors_never_raise(self):
        with mock.patch.object(fa.smtplib, "SMTP") as smtp:
            smtp.return_value.login.side_effect = OSError("auth failed")
            self.assertFalse(fa.send_alert("s", "b", fa.smtp_settings(SMTP_ENV)))
            smtp.return_value.quit.assert_called_once()


class MainTests(unittest.TestCase):
    def run_main(self, send_ok):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            (root / "logs" / "digest_run.log").write_text("ERROR main: Live GBP/USD rate unavailable\n")
            output = root / "github_output"
            env = dict(SMTP_ENV, OUTCOME_DIGEST="failure", GITHUB_OUTPUT=str(output))
            with mock.patch.object(fa, "REPO_ROOT", root), \
                 mock.patch.object(fa, "ALERT_MARKER_DIR", root / ".failure-alert"), \
                 mock.patch.object(fa, "SENT_MARKER", root / "data" / "last_email_sent.json"), \
                 mock.patch.object(fa, "london_now", return_value=NOW), \
                 mock.patch.object(fa, "send_alert", return_value=send_ok) as send:
                code = fa.main(env)
            return (code, send.call_args[0][1],
                    output.read_text() if output.exists() else "",
                    (root / ".failure-alert" / "sent").exists())

    def test_successful_alert_sets_step_output_and_cache_marker(self):
        code, body, output, marker = self.run_main(send_ok=True)
        self.assertEqual(code, 0)
        self.assertIn("Live GBP/USD rate unavailable", body)
        self.assertEqual(output, "sent=true\n")
        self.assertTrue(marker)

    def test_failed_alert_leaves_no_marker_so_the_next_failure_retries(self):
        code, _, output, marker = self.run_main(send_ok=False)
        self.assertEqual(code, 0)
        self.assertEqual(output, "")
        self.assertFalse(marker)


if __name__ == "__main__":
    unittest.main()

import os
import re
import unittest

# GEMINI.md workflow invariants, checked as plain text so the suite needs no YAML dependency.

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKFLOW = os.path.join(_ROOT, ".github", "workflows", "daily_trading_digest.yml")


def _read_workflow():
    with open(_WORKFLOW, encoding="utf-8") as f:
        return f.read()


def _gitignored_paths():
    with open(os.path.join(_ROOT, ".gitignore"), encoding="utf-8") as f:
        return [line.strip().rstrip("/") for line in f if line.strip() and not line.startswith("#")]


class WorkflowInvariantTests(unittest.TestCase):
    def setUp(self):
        self.text = _read_workflow()
        self.lines = self.text.splitlines()

    def test_no_or_fallbacks_inside_expressions(self):
        self.assertEqual(re.findall(r"\$\{\{[^}]*\|\|[^}]*\}\}", self.text), [])

    def test_morning_retry_ticks_are_offset_from_the_hour(self):
        crons = re.findall(r"cron:\s*'([^']+)'", self.text)
        for tick in ("5 8", "20 8", "35 8", "50 8", "5 9", "20 9", "35 9", "50 9"):
            self.assertIn(f"{tick} * * *", crons)
        self.assertFalse([c for c in crons if c.split()[0] == "0"], "no cron tick may sit on minute :00")

    def test_gatekeeper_allows_late_runs_and_dedupes(self):
        self.assertIn('[ "$LONDON_HOUR" -lt 9 ]', self.text)
        self.assertNotRegex(self.text, r'LONDON_HOUR"?\s*=\s*"?0?9"')
        self.assertIn("origin/main:data/last_email_sent.json", self.text)

    def test_manual_and_external_dispatch_is_enabled(self):
        self.assertRegex(self.text, r"(?m)^\s+workflow_dispatch:")

    def test_git_add_never_includes_gitignored_files(self):
        ignored = [p for p in _gitignored_paths() if "*" not in p]
        add_lines = [line for line in self.lines if re.search(r"\bgit add\b", line)]
        self.assertTrue(add_lines)
        for line in add_lines:
            for path in ignored:
                self.assertNotIn(path, line.split("git add", 1)[1], f"gitignored {path!r} in: {line.strip()}")

    def test_inline_python_is_single_line(self):
        for line in self.lines:
            m = re.search(r"python3? -c ([\"'])", line)
            if m:
                self.assertIn(m.group(1), line[m.end():], f"multi-line python -c: {line.strip()}")

    def test_tee_pipelines_keep_the_real_exit_code(self):
        run_start = None
        for i, line in enumerate(self.lines):
            if re.match(r"\s+run: \|", line):
                run_start = i
            if "| tee" in line:
                self.assertIsNotNone(run_start)
                block = "\n".join(self.lines[run_start:i])
                self.assertIn("set -o pipefail", block, f"tee without pipefail at line {i + 1}")

    def test_failures_email_one_alert(self):
        self.assertIn("python3 scripts/failure_alert.py", self.text)
        self.assertRegex(self.text, r"if: failure\(\) && steps\.alert_seen\.outputs\.cache-hit != 'true'")
        self.assertIn("key: failure-alert-${{ steps.london.outputs.date }}", self.text)
        for step_id in ("tests", "digest", "commit"):
            self.assertRegex(self.text, rf"(?m)^\s+id: {step_id}$")
            self.assertIn(f"${{{{ steps.{step_id}.outcome }}}}", self.text)

    def test_unconfigured_repository_skips_instead_of_failing(self):
        # The template repo (and forks not set up yet) has no Trading 212 secrets: the gatekeeper must end the
        # run green with proceed=false, so GitHub never emails "Run failed" every day.
        gate = self.text.split("id: time_gate", 1)[1].split("- name:", 1)[0]
        self.assertIn("T212_KEY_ID: ${{ secrets.T212_KEY_ID }}", gate)
        self.assertIn("T212_KEY_ID_2: ${{ secrets.T212_KEY_ID_2 }}", gate)
        guard = gate.index('if [ -z "$T212_KEY_ID" ] && [ -z "$T212_KEY_ID_2" ]; then')
        self.assertLess(guard, gate.index("git fetch origin main"))
        self.assertIn('echo "proceed=false" >> $GITHUB_OUTPUT', gate[guard:guard + 400])


if __name__ == "__main__":
    unittest.main()

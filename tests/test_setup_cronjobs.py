import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts import setup_cronjobs as sc


class TestSetupCronjobs(unittest.TestCase):
    def test_parse_github_repo_handles_ssh_and_https(self) -> None:
        self.assertEqual(sc.parse_github_repo("git@github.com:owner/my-repo.git\n"), "owner/my-repo")
        self.assertEqual(sc.parse_github_repo("https://github.com/owner/my-repo.git"), "owner/my-repo")
        self.assertEqual(sc.parse_github_repo("https://github.com/owner/my-repo/"), "owner/my-repo")
        self.assertIsNone(sc.parse_github_repo("https://gitlab.com/owner/my-repo.git"))

    def test_load_env_file_and_resolve_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "# comment\n"
                "GITHUB_DISPATCH_TOKEN='ghp_test_token_123'\n"
                "export CRONJOB_API_KEY=\"cj_test_key_456\"\n",
                encoding="utf-8",
            )
            env: Dict[str, str] = {"GITHUB_DISPATCH_TOKEN": "already_set"}
            sc.load_env_file(env_file, env)
            self.assertEqual(env["GITHUB_DISPATCH_TOKEN"], "already_set")
            self.assertEqual(env["CRONJOB_API_KEY"], "cj_test_key_456")

            gh, cj = sc.resolve_credentials(env)
            self.assertEqual(gh, "already_set")
            self.assertEqual(cj, "cj_test_key_456")

        with self.assertRaises(sc.SetupError) as ctx:
            sc.resolve_credentials({})
        self.assertIn("GITHUB_DISPATCH_TOKEN", str(ctx.exception))
        self.assertIn("CRONJOB_API_KEY", str(ctx.exception))

    def test_build_job_specs_exact_schedules_and_redaction(self) -> None:
        specs = dict(sc.build_job_specs("acme/advisor", "secret_pat_xyz", ref="main", only="both"))
        self.assertEqual(set(specs.keys()), {"digest", "watch"})

        digest = specs["digest"]
        self.assertEqual(
            digest["url"],
            "https://api.github.com/repos/acme/advisor/actions/workflows/daily_trading_digest.yml/dispatches",
        )
        self.assertEqual(digest["requestMethod"], 1)  # POST
        self.assertEqual(digest["schedule"]["timezone"], "Europe/London")
        self.assertEqual(digest["schedule"]["hours"], [9])
        self.assertEqual(digest["schedule"]["minutes"], [0])
        self.assertEqual(digest["schedule"]["wdays"], [-1])
        self.assertEqual(digest["extendedData"]["body"], '{"ref":"main"}')

        watch = specs["watch"]
        self.assertEqual(
            watch["url"],
            "https://api.github.com/repos/acme/advisor/actions/workflows/intraday_watch.yml/dispatches",
        )
        self.assertEqual(watch["requestMethod"], 1)  # POST
        self.assertEqual(watch["schedule"]["timezone"], "Europe/London")
        self.assertEqual(watch["schedule"]["hours"], list(range(8, 21)))
        self.assertEqual(watch["schedule"]["minutes"], [0, 30])
        self.assertEqual(watch["schedule"]["wdays"], [1, 2, 3, 4, 5])
        hdrs = watch["extendedData"]["headers"]
        self.assertEqual(hdrs["Accept"], "application/vnd.github+json")
        self.assertEqual(hdrs["Authorization"], "Bearer secret_pat_xyz")
        self.assertEqual(hdrs["X-GitHub-Api-Version"], "2022-11-28")
        self.assertEqual(watch["extendedData"]["body"], '{"ref":"main"}')

        redacted = sc.redact_job_spec(watch)
        self.assertNotIn("secret_pat_xyz", json.dumps(redacted))
        self.assertEqual(redacted["extendedData"]["headers"]["Authorization"], "Bearer <REDACTED>")

    def test_verify_github_token_checks_read_and_write(self) -> None:
        calls: List[Tuple[str, str]] = []

        def fake_http(method: str, url: str, headers: Dict[str, str], body: Optional[bytes]) -> Tuple[int, str]:
            calls.append((method, url))
            if method == "GET":
                return 200, json.dumps({"workflows": [{"path": ".github/workflows/intraday_watch.yml"}]})
            return 204, ""

        paths = sc.verify_github_token("acme/advisor", "pat_ok", verify_dispatch=True, http=fake_http)
        self.assertEqual(paths, [".github/workflows/intraday_watch.yml"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0], "POST")

        def readonly_http(method: str, url: str, headers: Dict[str, str], body: Optional[bytes]) -> Tuple[int, str]:
            return (200, '{"workflows":[]}') if method == "GET" else (403, '{"message":"Resource not accessible"}')

        with self.assertRaises(sc.SetupError) as ctx:
            sc.verify_github_token("acme/advisor", "pat_ro", verify_dispatch=True, http=readonly_http)
        self.assertIn("Read and write", str(ctx.exception))
        self.assertNotIn("pat_ro", str(ctx.exception))

    def test_upsert_cronjobs_updates_existing_and_creates_missing(self) -> None:
        specs = sc.build_job_specs("acme/advisor", "pat_123", ref="main", only="both")
        calls: List[Tuple[str, str, Optional[str]]] = []
        sleeps: List[float] = []

        existing_payload = (
            '{"jobs":[{"jobId":777,"url":"https:\\/\\/api.github.com\\/repos\\/acme\\/advisor'
            '\\/actions\\/workflows\\/daily_trading_digest.yml\\/dispatches"}]}'
        )

        def fake_http(method: str, url: str, headers: Dict[str, str], body: Optional[bytes]) -> Tuple[int, str]:
            calls.append((method, url, body.decode("utf-8") if body else None))
            if method == "GET" and url.endswith("/jobs"):
                return 200, existing_payload
            if method == "PATCH" and url.endswith("/jobs/777"):
                return 200, "{}"
            if method == "PUT" and url.endswith("/jobs"):
                return 200, '{"jobId":888}'
            return 500, "unexpected"

        out = sc.upsert_cronjobs(specs, "cj_key", http=fake_http, sleep_fn=sleeps.append)
        self.assertEqual(
            [(r["kind"], r["action"], r["jobId"]) for r in out],
            [("digest", "updated", 777), ("watch", "created", 888)],
        )
        self.assertEqual(sleeps, [1.5])
        self.assertEqual([c[0] for c in calls], ["GET", "PATCH", "PUT"])


if __name__ == "__main__":
    unittest.main()

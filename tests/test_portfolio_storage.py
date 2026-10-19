import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Offline tests for portfolio_storage (trade detection, closed archive, read-only getters, deposit-adjusted
# TWR, benchmark index, recommendation ledger) and scripts/cleanup_trade_log.py. Every test runs against a
# temporary data directory (module paths are patched) and the VWRP.L benchmark fetch is mocked.

import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

import portfolio_storage as ps  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


class _TempDataDirMixin:
    """Points every portfolio_storage path at a fresh temp dir and mocks the benchmark download."""

    def setUp(self):
        super().setUp()
        self.maxDiff = None
        self.tmp = Path(tempfile.mkdtemp(prefix="t212-storage-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = self.tmp / "data"
        paths = {
            "DATA_DIR": self.data,
            "SNAPSHOTS_DIR": self.data / "snapshots",
            "HISTORY_FILE": self.data / "portfolio_history.json",
            "TRADE_LOG_FILE": self.data / "trade_log.json",
            "ENTRIES_FILE": self.data / "position_entries.json",
            "TRIMS_FILE": self.data / "position_trims.json",
            "CLOSED_FILE": self.data / "position_closed.json",
            "LEDGER_FILE": self.data / "recommendation_ledger.json",
        }
        for name, value in paths.items():
            patcher = mock.patch.object(ps, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        bench_patcher = mock.patch.object(ps, "_fetch_benchmark_closes", return_value=(None, None))
        self.bench = bench_patcher.start()
        self.addCleanup(bench_patcher.stop)

    # --- helpers -------------------------------------------------------------------------------------
    @staticmethod
    def _pos(ticker, yf_ticker, qty, price=100.0, fill=None):
        pos = {"ticker": ticker, "yf_ticker": yf_ticker, "quantity": qty, "currentPrice": price}
        if fill:
            pos["initialFillDate"] = fill
        return pos

    @staticmethod
    def _detect(day, positions_by_account):
        accounts = {
            acc: {"raw_positions": [dict(p) for p in positions]} for acc, positions in positions_by_account.items()
        }
        return ps.detect_and_record_trades(accounts, current_date_str=day)

    def _snapshot(self, day, positions_by_account, totals=None, cash=None, external_flows=None):
        """Persists a daily snapshot the way main.py does (it becomes the next run's prior holdings)."""
        accounts_data, account_shares = {}, {}
        for acc, positions in positions_by_account.items():
            analyses = [
                {
                    "ticker": p["yf_ticker"],
                    "quantity": p["quantity"],
                    "price": p.get("currentPrice", 0.0),
                    "currency": "USD",
                    "value_gbp": p["quantity"] * p.get("currentPrice", 0.0),
                }
                for p in positions
            ]
            total = (totals or {}).get(acc, sum(a["value_gbp"] for a in analyses))
            accounts_data[acc] = {"cash": dict((cash or {}).get(acc, {})), "analyses": analyses}
            account_shares[acc] = {"total_value_gbp": total, "invested_gbp": total, "share_of_total_pct": 0.0}
        consolidated = {
            "account_shares": account_shares,
            "total_net_worth_gbp": sum(s["total_value_gbp"] for s in account_shares.values()),
        }
        with mock.patch.object(ps, "_utc_now", return_value=datetime.fromisoformat(f"{day}T08:05:00+00:00")):
            path = ps.save_daily_snapshot(
                accounts_data, consolidated, {}, {}, 1.30, snapshot_date=day, external_flows=external_flows
            )
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))


class TradeDetectionTests(_TempDataDirMixin, unittest.TestCase):
    NVDA_FILL = "2026-09-10T14:30:12.000+03:00"

    def test_first_run_uses_initial_fill_date(self):
        new = self._detect("2026-10-01", {"Main": [
            self._pos("NVDA_US_EQ", "NVDA", 10, fill=self.NVDA_FILL),
            self._pos("VUAAl_EQ", "VUAA.L", 5),  # no initialFillDate: dated from the run
        ]})
        by_ticker = {t["ticker"]: t for t in new}
        self.assertEqual(set(by_ticker), {"NVDA", "VUAA.L"})
        self.assertEqual(by_ticker["NVDA"]["action"], "BUY (NEW ENTRY)")
        self.assertEqual(by_ticker["NVDA"]["date"], "2026-09-10")
        self.assertEqual(by_ticker["VUAA.L"]["date"], "2026-10-01")
        self.assertEqual(ps.get_position_entries()["Main"], {"NVDA": "2026-09-10", "VUAA.L": "2026-10-01"})

    def test_add_records_accumulation(self):
        day1 = {"Main": [self._pos("NVDA_US_EQ", "NVDA", 10, fill=self.NVDA_FILL)]}
        self._detect("2026-10-01", day1)
        self._snapshot("2026-10-01", day1)

        new = self._detect("2026-10-02", {"Main": [self._pos("NVDA_US_EQ", "NVDA", 15, price=110.0, fill=self.NVDA_FILL)]})
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]["action"], "BUY (ACCUMULATION)")
        self.assertEqual(new[0]["date"], "2026-10-02")
        self.assertAlmostEqual(new[0]["shares_added"], 5.0)
        self.assertAlmostEqual(new[0]["new_total_shares"], 15.0)
        self.assertEqual(ps.get_position_entries()["Main"]["NVDA"], "2026-10-02")  # an add re-arms the entry date

    def test_partial_sell_records_trim(self):
        day1 = {"Main": [self._pos("NVDA_US_EQ", "NVDA", 10, fill=self.NVDA_FILL)]}
        self._detect("2026-10-01", day1)
        self._snapshot("2026-10-01", day1)

        new = self._detect("2026-10-02", {"Main": [self._pos("NVDA_US_EQ", "NVDA", 6, fill=self.NVDA_FILL)]})
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]["action"], "SELL (TRIMMED)")
        self.assertAlmostEqual(new[0]["shares_sold"], 4.0)
        self.assertAlmostEqual(new[0]["remaining_shares"], 6.0)
        self.assertEqual(ps.get_position_trims()["Main"]["NVDA"], "2026-10-02")
        self.assertEqual(ps.get_position_entries()["Main"]["NVDA"], "2026-09-10")  # still held
        self.assertEqual(ps.get_closed_positions(), {})

    def test_full_close_records_sell_closed_and_archives_entry(self):
        day1 = {"Main": [
            self._pos("NVDA_US_EQ", "NVDA", 10, price=100.0, fill=self.NVDA_FILL),
            self._pos("MSFT_US_EQ", "MSFT", 3, price=400.0),
        ]}
        self._detect("2026-10-01", day1)
        self._snapshot("2026-10-01", day1)

        new = self._detect("2026-10-02", {"Main": [self._pos("MSFT_US_EQ", "MSFT", 3, price=405.0)]})
        self.assertEqual(len(new), 1)  # MSFT unchanged: no event
        closed = new[0]
        self.assertEqual((closed["ticker"], closed["action"]), ("NVDA", "SELL (CLOSED)"))
        self.assertAlmostEqual(closed["shares_sold"], 10.0)
        self.assertEqual(closed["remaining_shares"], 0.0)
        self.assertEqual(closed["price"], 100.0)  # last known price, from the prior snapshot
        self.assertEqual(closed["price_source"], "prior_snapshot")

        self.assertNotIn("NVDA", ps.get_position_entries()["Main"])
        expected = {"closed_date": "2026-10-02", "entry_date": "2026-09-10", "shares_sold": 10.0}
        self.assertEqual(ps.get_closed_positions()["Main"]["NVDA"], expected)
        self.assertEqual(self._read(ps.CLOSED_FILE)["Main"]["NVDA"], expected)

    def test_idempotent_rerun_creates_no_duplicates(self):
        day1 = {"Main": [
            self._pos("NVDA_US_EQ", "NVDA", 10, fill=self.NVDA_FILL),
            self._pos("MSFT_US_EQ", "MSFT", 3),
            self._pos("AAPL_US_EQ", "AAPL", 2),
        ]}
        self.assertEqual(len(self._detect("2026-10-01", day1)), 3)
        self._snapshot("2026-10-01", day1)
        day2 = {"Main": [self._pos("NVDA_US_EQ", "NVDA", 12), self._pos("MSFT_US_EQ", "MSFT", 1)]}  # add, trim, close
        self.assertEqual(len(self._detect("2026-10-02", day2)), 3)
        log_after_first_pass = ps.TRADE_LOG_FILE.read_bytes()

        self.assertEqual(self._detect("2026-10-02", day2), [])  # same-day re-run
        self._snapshot("2026-10-02", day2)
        self.assertEqual(self._detect("2026-10-02", day2), [])  # re-run after today's snapshot exists
        self.assertEqual(self._detect("2026-10-01", day1), [])  # replaying an older day is also deduped globally
        self.assertEqual(ps.TRADE_LOG_FILE.read_bytes(), log_after_first_pass)
        keys = [(t["account"], t["ticker"], t["date"], t["action"]) for t in self._read(ps.TRADE_LOG_FILE)]
        self.assertEqual(len(keys), len(set(keys)))

    def test_close_then_rebuy(self):
        day1 = {"Main": [self._pos("NVDA_US_EQ", "NVDA", 10, fill=self.NVDA_FILL)]}
        self._detect("2026-10-01", day1)
        self._snapshot("2026-10-01", day1)
        closed = self._detect("2026-10-02", {"Main": []})  # genuine empty account: everything was sold
        self.assertEqual([t["action"] for t in closed], ["SELL (CLOSED)"])
        self._snapshot("2026-10-02", {"Main": []})

        rebuy = self._detect("2026-10-03", {"Main": [self._pos("NVDA_US_EQ", "NVDA", 4, fill="2026-10-03T09:00:00Z")]})
        self.assertEqual(len(rebuy), 1)
        self.assertEqual((rebuy[0]["action"], rebuy[0]["date"]), ("BUY (NEW ENTRY)", "2026-10-03"))
        self.assertEqual(ps.get_position_entries()["Main"]["NVDA"], "2026-10-03")
        self.assertEqual(ps.get_closed_positions()["Main"]["NVDA"]["closed_date"], "2026-10-02")
        actions = [(t["date"], t["action"]) for t in self._read(ps.TRADE_LOG_FILE)]
        self.assertEqual(
            actions,
            [("2026-09-10", "BUY (NEW ENTRY)"), ("2026-10-02", "SELL (CLOSED)"), ("2026-10-03", "BUY (NEW ENTRY)")],
        )

    def test_rebuy_ignores_a_fill_date_from_before_the_close(self):
        day1 = {"Main": [self._pos("NVDA_US_EQ", "NVDA", 10, fill=self.NVDA_FILL)]}
        self._detect("2026-10-01", day1)
        self._snapshot("2026-10-01", day1)
        self._detect("2026-10-02", {"Main": []})
        self._snapshot("2026-10-02", {"Main": []})
        rebuy = self._detect("2026-10-03", {"Main": [self._pos("NVDA_US_EQ", "NVDA", 4, fill=self.NVDA_FILL)]})
        self.assertEqual(rebuy[0]["date"], "2026-10-03")  # a stale pre-close fill date is not trusted

    def test_missing_position_data_is_never_treated_as_closed(self):
        day1 = {"Main": [self._pos("NVDA_US_EQ", "NVDA", 10)], "Secondary": [self._pos("AAPL_US_EQ", "AAPL", 5)]}
        self._detect("2026-10-01", day1)
        self._snapshot("2026-10-01", day1)
        entries_before = ps.ENTRIES_FILE.read_bytes()

        with self.assertLogs("storage", level="WARNING"):
            new = ps.detect_and_record_trades({"Main": {"raw_positions": None}}, current_date_str="2026-10-02")
        self.assertEqual(new, [])  # Main failed (skipped) and Secondary is absent: neither is treated as sold out
        self.assertEqual(ps.ENTRIES_FILE.read_bytes(), entries_before)
        self.assertFalse(ps.CLOSED_FILE.exists())

    def test_canonical_keys_prevent_phantom_trades(self):
        ps.SNAPSHOTS_DIR.mkdir(parents=True)
        (ps.SNAPSHOTS_DIR / "2026-10-01.json").write_text(json.dumps({
            "date": "2026-10-01",
            "accounts": {"Secondary": {"positions": [{"ticker": "ASMLa_EQ", "quantity": 2.0, "price": 1500.0}]}},
        }), encoding="utf-8")
        ps.ENTRIES_FILE.write_text(json.dumps({"Secondary": {"ASMLa_EQ": "2026-09-15", "ASML.AS": "2026-09-18"}}))
        ps.TRADE_LOG_FILE.write_text(json.dumps([
            {"account": "Secondary", "ticker": "ASMLa_EQ", "date": "2026-10-02", "action": "SELL (TRIMMED)"},
        ]))
        self.assertEqual(ps.get_position_entries()["Secondary"], {"ASML.AS": "2026-09-18"})  # latest date wins

        unchanged = {"Secondary": [self._pos("ASMLa_EQ", "ASML.AS", 2.0)]}
        self.assertEqual(self._detect("2026-10-02", unchanged), [])  # no NEW ENTRY / CLOSED pair
        self.assertEqual(self._read(ps.ENTRIES_FILE), {"Secondary": {"ASML.AS": "2026-09-18"}})  # stale key migrated

        trimmed = {"Secondary": [self._pos("ASMLa_EQ", "ASML.AS", 1.0)]}
        self.assertEqual(self._detect("2026-10-02", trimmed), [])  # same canonical event already logged


class ReadOnlyGetterTests(_TempDataDirMixin, unittest.TestCase):
    def _call_read_paths(self):
        accounts = {"Main": {"cash": {"recent_transactions": [], "currency": "GBP"}}}
        return [
            ps.get_position_entries(),
            ps.get_position_trims(),
            ps.get_closed_positions(),
            ps.load_latest_snapshot(),
            ps.load_latest_snapshot(before_date="2026-10-02"),
            ps.load_portfolio_history(),
            ps.compute_external_flows(accounts, current_date_str="2026-10-02", gbp_rate=1.3),
            ps.compute_day_over_day_delta(1000.0, {"Main": 1000.0}, current_date_str="2026-10-02"),
        ]

    def test_getters_never_create_files_or_directories(self):
        self._call_read_paths()
        self.assertFalse(self.data.exists())

    def test_getters_never_modify_existing_files(self):
        ps.SNAPSHOTS_DIR.mkdir(parents=True)
        ps.ENTRIES_FILE.write_text(json.dumps({"Secondary": {"ASMLa_EQ": "2026-09-15"}}))
        ps.TRIMS_FILE.write_text(json.dumps({"Main": {"NVDA": "2026-10-01"}}))
        ps.TRADE_LOG_FILE.write_text(json.dumps([
            {"account": "Main", "ticker": "TSM", "date": "2026-10-07", "action": "SELL (TRIMMED)"},
        ]))
        ps.CLOSED_FILE.write_text(json.dumps({"Main": {"MSFT": {"closed_date": "2026-10-01", "shares_sold": 1.0}}}))
        ps.HISTORY_FILE.write_text(json.dumps([{"date": "2026-10-01", "combined_total": 1000.0}]))
        (ps.SNAPSHOTS_DIR / "2026-10-01.json").write_text(json.dumps({
            "date": "2026-10-01",
            "timestamp_utc": "2026-10-01T08:05:00+00:00",
            "combined": {"total_value_gbp": 1000.0},
            "accounts": {"Main": {"total_value_gbp": 1000.0, "positions": []}},
        }))
        before = {p: p.read_bytes() for p in self.data.rglob("*") if p.is_file()}

        results = self._call_read_paths()
        self.assertEqual(results[0], {"Secondary": {"ASML.AS": "2026-09-15"}})  # canonicalised in memory only
        self.assertEqual(results[1]["Main"], {"NVDA": "2026-10-01", "TSM": "2026-10-07"})  # trade-log trims merged

        after = {p: p.read_bytes() for p in self.data.rglob("*") if p.is_file()}
        self.assertEqual(after, before)


class PerformanceTrackingTests(_TempDataDirMixin, unittest.TestCase):
    DAY1_POSITIONS = {"Main": [{"ticker": "VWRPl_EQ", "yf_ticker": "VWRP.L", "quantity": 10.0, "currentPrice": 100.0}]}

    def test_twr_factor_with_deposit(self):
        self.assertAlmostEqual(ps.compute_twr_factor(1000.0, 1600.0, 500.0), 1600.0 / 1500.0)
        self.assertAlmostEqual(ps.compute_twr_factor(1000.0, 900.0, -200.0), 900.0 / 800.0)  # withdrawal
        self.assertAlmostEqual(ps.compute_twr_factor(1000.0, 1100.0), 1.1)  # no flow
        self.assertIsNone(ps.compute_twr_factor(0.0, 100.0, 0.0))  # undefined without a starting value

    def test_snapshot_chain_records_flows_twr_and_benchmark(self):
        snap1 = self._snapshot("2026-10-01", self.DAY1_POSITIONS, totals={"Main": 1000.0})
        self.assertEqual(snap1["performance"]["twr_index_total"], 100.0)
        self.assertEqual(snap1["performance"]["twr_index_main"], 100.0)
        self.assertIsNone(snap1["performance"]["benchmark_vwrp_index"])  # benchmark unavailable -> None

        txs = [
            {"type": "DEPOSIT", "amount": 500.0, "reference": "DEP-1", "dateTime": "2026-10-01T15:00:00.000Z"},
            {"type": "DEPOSIT", "amount": 500.0, "reference": "DEP-1", "dateTime": "2026-10-01T15:00:00.000Z"},  # repeat
            {"type": "INTEREST", "amount": 1.25, "reference": "INT-1", "dateTime": "2026-10-01T16:00:00.000Z"},
            {"type": "DEPOSIT", "amount": 900.0, "reference": "DEP-0", "dateTime": "2026-09-30T10:00:00.000Z"},  # pre-snapshot
        ]
        cash2 = {"Main": {"recent_transactions": txs, "recent_transactions_ok": True, "currency": "GBP"}}
        accounts2 = {"Main": {"cash": cash2["Main"]}}
        flows = ps.compute_external_flows(accounts2, current_date_str="2026-10-02", gbp_rate=1.3)
        self.assertEqual(flows["by_account"], {"Main": 500.0})
        self.assertEqual([f["reference"] for f in flows["items"]["Main"]], ["DEP-1"])

        dod = ps.compute_day_over_day_delta(
            1600.0, {"Main": 1600.0}, current_date_str="2026-10-02", external_flows_by_account=flows["by_account"]
        )
        self.assertAlmostEqual(dod["combined_delta_gbp"], 100.0)  # 1600 - 1000 - 500 deposit
        self.assertAlmostEqual(dod["combined_delta_pct"], 100.0 / 1500.0 * 100.0)
        self.assertAlmostEqual(dod["accounts"]["Main"]["delta_gbp"], 100.0)

        self.bench.return_value = (100.0, 103.0)
        snap2 = self._snapshot("2026-10-02", self.DAY1_POSITIONS, totals={"Main": 1600.0}, cash=cash2, external_flows=flows)
        perf2 = snap2["performance"]
        self.assertAlmostEqual(perf2["twr_index_main"], 100.0 * 1600.0 / 1500.0, places=4)
        self.assertAlmostEqual(perf2["twr_index_total"], 100.0 * 1600.0 / 1500.0, places=4)
        self.assertEqual(perf2["external_flows_gbp"], 500.0)
        self.assertEqual(perf2["benchmark_vwrp_index"], 103.0)
        self.assertEqual(snap2["accounts"]["Main"]["external_flows_gbp"], 500.0)
        self.assertEqual(snap2["combined"]["external_flows_gbp"], 500.0)
        self.bench.assert_called_with("2026-10-01", "2026-10-02")  # benchmark based at the first tracked day

        history = ps.load_portfolio_history()
        self.assertEqual([h["date"] for h in history], ["2026-10-01", "2026-10-02"])
        for key in ("external_flows_gbp", "twr_index_total", "twr_index_main", "benchmark_vwrp_index"):
            self.assertIn(key, history[-1])
        self.assertEqual(history[-1]["external_flows_gbp"], 500.0)

        # Day 3: the same deposit is still in the broker's recent list but must not be counted again.
        cash3 = {"Main": dict(cash2["Main"])}
        flows3 = ps.compute_external_flows({"Main": {"cash": cash3["Main"]}}, current_date_str="2026-10-03", gbp_rate=1.3)
        self.assertEqual(flows3["by_account"], {"Main": 0.0})
        snap3 = self._snapshot("2026-10-03", self.DAY1_POSITIONS, totals={"Main": 1700.0}, cash=cash3, external_flows=flows3)
        self.assertAlmostEqual(snap3["performance"]["twr_index_main"], 100.0 * (1600.0 / 1500.0) * (1700.0 / 1600.0), places=4)

    def test_withdrawal_is_a_negative_flow(self):
        self._snapshot("2026-10-01", self.DAY1_POSITIONS, totals={"Main": 1000.0})
        txs = [{"type": "WITHDRAW", "amount": -200.0, "reference": "WD-1", "dateTime": "2026-10-01T12:00:00Z"}]
        flows = ps.compute_external_flows(
            {"Main": {"cash": {"recent_transactions": txs, "currency": "GBP"}}}, current_date_str="2026-10-02"
        )
        self.assertEqual(flows["by_account"], {"Main": -200.0})

    def test_no_flows_attributed_on_the_first_tracked_day(self):
        txs = [{"type": "DEPOSIT", "amount": 500.0, "reference": "DEP-1", "dateTime": "2026-10-01T12:00:00Z"}]
        flows = ps.compute_external_flows(
            {"Main": {"cash": {"recent_transactions": txs, "currency": "GBP"}}}, current_date_str="2026-10-02"
        )
        self.assertEqual(flows["by_account"], {"Main": 0.0})


class RecommendationLedgerTests(_TempDataDirMixin, unittest.TestCase):
    RECORD = {
        "date": "2026-10-02",
        "model_version": "2026.10.18-review-fixes",
        "account": "Main",
        "action": "BUY (NEW)",
        "ticker": "VWRP.L",
        "amount_gbp": 1000.0,
        "price": 120.0,
        "currency": "GBP",
        "reason": "test",
    }

    def test_append_dedupes_same_day_records(self):
        self.assertEqual(ps.append_recommendation_records([self.RECORD]), 1)
        self.assertEqual(ps.append_recommendation_records([self.RECORD]), 0)
        self.assertEqual(self._read(ps.LEDGER_FILE), [self.RECORD])
        self.assertEqual(ps.append_recommendation_records([]), 0)

    def test_unreadable_ledger_is_never_overwritten(self):
        self.data.mkdir(parents=True)
        ps.LEDGER_FILE.write_text("{corrupt", encoding="utf-8")
        with self.assertLogs("storage", level="ERROR"):
            self.assertEqual(ps.append_recommendation_records([self.RECORD]), 0)
        self.assertEqual(ps.LEDGER_FILE.read_text(encoding="utf-8"), "{corrupt")


def _load_cleanup_module():
    spec = importlib.util.spec_from_file_location("cleanup_trade_log", REPO_ROOT / "scripts" / "cleanup_trade_log.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CleanupScriptTests(unittest.TestCase):
    LOG = [
        {"account": "Secondary", "ticker": "ASMLa_EQ", "date": "2026-09-15", "action": "BUY (NEW ENTRY)",
         "shares_added": 1.25, "new_total_shares": 1.25, "price": 1400.0},
        {"account": "Secondary", "ticker": "ASML.AS", "date": "2026-09-15", "action": "BUY (NEW ENTRY)",
         "shares_added": 1.25, "new_total_shares": 1.25, "price": 1400.5},  # canonical twin
        {"account": "Main", "ticker": "NVDA", "date": "2026-10-05", "action": "SELL (TRIMMED)",
         "shares_sold": 2.0, "remaining_shares": 5.0, "price": 1.0},
        {"account": "Main", "ticker": "NVDA", "date": "2026-10-05", "action": "SELL (TRIMMED)",
         "shares_sold": 2.0000000000000004, "remaining_shares": 5.0, "price": 1.1},  # exact repeat (float noise)
        {"account": "Main", "ticker": "NVDA", "date": "2026-10-05", "action": "SELL (TRIMMED)",
         "shares_sold": 4.0, "remaining_shares": 3.0, "price": 1.2},  # different shares: kept
    ]

    @classmethod
    def setUpClass(cls):
        cls.cleanup = _load_cleanup_module()

    def test_dedupe_keeps_first_occurrence_and_canonicalises(self):
        cleaned, stats = self.cleanup.dedupe_trade_log([dict(r) for r in self.LOG])
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(stats["exact_duplicates"], 1)
        self.assertEqual(stats["canonical_duplicates"], 1)
        self.assertEqual(stats["tickers_canonicalised"], 1)
        self.assertEqual((cleaned[0]["ticker"], cleaned[0]["price"]), ("ASML.AS", 1400.0))  # first record kept
        self.assertEqual([r["shares_sold"] for r in cleaned[1:]], [2.0, 4.0])
        again, stats_again = self.cleanup.dedupe_trade_log(cleaned)
        self.assertEqual(again, cleaned)
        self.assertEqual(stats_again["exact_duplicates"] + stats_again["canonical_duplicates"], 0)

    def test_date_map_migration_keeps_latest_date(self):
        raw = {"Secondary": {"ASMLa_EQ": "2026-09-15", "ASML.AS": "2026-09-18", "TSM": "2026-09-15"}}
        migrated, changes, stale = self.cleanup.migrate_date_map(raw)
        self.assertEqual(migrated, {"Secondary": {"ASML.AS": "2026-09-18", "TSM": "2026-09-15"}})
        self.assertEqual((stale, len(changes)), (1, 1))
        self.assertEqual(self.cleanup.migrate_date_map(migrated), (migrated, [], 0))

    def test_main_rewrites_once_then_is_a_no_op(self):
        tmp = Path(tempfile.mkdtemp(prefix="t212-cleanup-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "trade_log.json").write_text(json.dumps(self.LOG, indent=2))
        (tmp / "position_entries.json").write_text(json.dumps({"Secondary": {"ASMLa_EQ": "2026-09-15"}}, indent=2))
        (tmp / "position_trims.json").write_text(json.dumps({"Main": {"NVDA": "2026-10-05"}}, indent=2))
        trims_before = (tmp / "position_trims.json").read_bytes()

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.cleanup.main(["--data-dir", str(tmp), "--dry-run"]), 0)
        self.assertEqual(json.loads((tmp / "trade_log.json").read_text()), self.LOG)  # dry run wrote nothing

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.cleanup.main(["--data-dir", str(tmp)]), 0)
        self.assertEqual(len(json.loads((tmp / "trade_log.json").read_text())), 3)
        self.assertEqual(json.loads((tmp / "position_entries.json").read_text()), {"Secondary": {"ASML.AS": "2026-09-15"}})
        self.assertEqual((tmp / "position_trims.json").read_bytes(), trims_before)  # unchanged file not rewritten

        snapshot = {p.name: p.read_bytes() for p in tmp.iterdir()}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(self.cleanup.main(["--data-dir", str(tmp)]), 0)
        self.assertEqual({p.name: p.read_bytes() for p in tmp.iterdir()}, snapshot)
        self.assertIn("none (already clean)", out.getvalue())


if __name__ == "__main__":
    unittest.main()

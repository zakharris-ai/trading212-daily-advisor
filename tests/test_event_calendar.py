"""Offline tests for the event radar (scheduled macro events, holdings' results dates, circuit-breaker check)."""

import os
import sys
import unittest
from datetime import date
from unittest import mock

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config  # noqa: E402
import event_calendar as ec  # noqa: E402
import mailer  # noqa: E402
import strategy_engine as se  # noqa: E402
from research_agent import DeepResearchAgent  # noqa: E402

MON_2_NOV = date(2026, 11, 2)
ALLOC_55_20 = {"targets": {"core": 0.40, "satellite": 0.55, "defensive": 0.05}, "rebalance_policy": "active",
               "risk_off_satellite_pct": 0.20}


def _pos(ticker, value, date_str=None, ts=None, estimate=None, name=None):
    research = {}
    if date_str:
        research = {"earnings_date_str": date_str, "earnings_ts_utc": ts, "earnings_is_estimate": estimate}
    return {"ticker": ticker, "name": name or ticker, "search_symbol": ticker.split(".")[0], "quantity": 10.0,
            "value_gbp": value, "research": research}


def _results(sells_main=(), buys_main=()):
    return {
        "Main": {
            "analyses": [
                _pos("PLTR", 8000.0, "Nov 02, 2026", "2026-11-02T21:00:00+00:00", False, "Palantir"),
                _pos("SMGB.L", 15000.0),
                _pos("CRWD", 3000.0, "Dec 01, 2026", "2026-12-01T21:00:00+00:00", True, "CrowdStrike"),
            ],
            "rebalancing_plan": {"sell_orders": list(sells_main), "buy_orders": list(buys_main)},
        },
        "Second": {
            "analyses": [_pos("PLTR", 7000.0, "Nov 02, 2026", "2026-11-02T21:00:00+00:00", False, "Palantir")],
            "rebalancing_plan": {"sell_orders": [], "buy_orders": []},
        },
    }


class MacroCalendarTests(unittest.TestCase):
    def test_window_and_uk_times(self):
        evs = {e["source"] + "|" + e["date"]: e for e in ec.upcoming_macro_events(MON_2_NOV, 14)}
        self.assertIn("one_off|2026-11-03", evs)                       # US midterms
        self.assertEqual(evs["boe|2026-11-05"]["time_uk"], "12:00")
        self.assertEqual(evs["us_jobs|2026-11-06"]["time_uk"], "13:30")  # 08:30 ET in winter
        self.assertEqual(evs["us_cpi|2026-11-10"]["title"], "US CPI inflation (October)")
        self.assertNotIn("uk_cpi|2026-11-18", evs)                     # outside 14 days
        self.assertTrue(all(not e["action_needed"] for e in evs.values()))
        later = {e["source"] + "|" + e["date"]: e for e in ec.upcoming_macro_events(MON_2_NOV, 40)}
        self.assertEqual(later["fomc|2026-12-09"]["time_uk"], "19:00")  # 14:00 ET
        self.assertIn("one_off|2026-11-26", later)                      # Thanksgiving

    def test_us_times_follow_both_daylight_saving_switches(self):
        # 2027: US clocks change 14 Mar, UK clocks 28 Mar -> 14:00 ET is 18:00 UK on 17 Mar
        evs = {e["source"] + "|" + e["date"]: e for e in ec.upcoming_macro_events(date(2027, 3, 1), 30)}
        self.assertEqual(evs["us_cpi|2027-03-10"]["time_uk"], "13:30")
        self.assertEqual(evs["fomc|2027-03-17"]["time_uk"], "18:00")

    def test_tax_year_end_has_a_longer_lead_and_needs_action(self):
        evs = [e for e in ec.upcoming_macro_events(date(2027, 3, 1), 14) if e["kind"] == "tax"]
        self.assertEqual(len(evs), 1)
        self.assertTrue(evs[0]["action_needed"])
        self.assertIn("ISA", evs[0]["what_to_do"])

    def test_schedule_dates_are_valid_and_sorted(self):
        for key, dates in ec.MACRO_SCHEDULE.items():
            parsed = [ec._parse_date(d) for d in dates]
            self.assertTrue(all(parsed), key)
            self.assertEqual(parsed, sorted(parsed), key)
            self.assertIn(key, ec.MACRO_TEMPLATES)


class EarningsTests(unittest.TestCase):
    def test_holders_are_grouped_and_hold_is_the_default(self):
        items = ec.holdings_earnings_events(_results(), MON_2_NOV, 14, household_basis_gbp=200000.0)
        pltr = [i for i in items if i["root"] == "PLTR"]
        self.assertEqual(len(pltr), 1)
        it = pltr[0]
        self.assertEqual(it["time_uk"], "21:00")
        self.assertEqual(it["when_label"], "after the US close")
        self.assertIn("Main £8,000", it["holders"])
        self.assertIn("Second £7,000", it["holders"])
        self.assertEqual(it["household_pct"], 7.5)
        self.assertEqual(it["impact"], "high")
        self.assertFalse(it["action_needed"])
        self.assertTrue(it["what_to_do"].startswith("Hold."))
        self.assertFalse([i for i in items if i["root"] == "CRWD"])    # 1 Dec is outside 14 days
        crwd = [i for i in ec.holdings_earnings_events(_results(), MON_2_NOV, 30) if i["root"] == "CRWD"]
        self.assertIn("estimate", crwd[0]["when_label"])

    def test_a_due_sell_is_placed_before_the_results_and_the_order_is_stamped(self):
        sell = {"ticker": "PLTR", "search_symbol": "PLTR", "action": "TRIM", "estimated_gbp": 2000.0, "shares": 5}
        results = _results(sells_main=[sell])
        radar = ec.build_event_radar(results, tech_regime=None, as_of=MON_2_NOV, horizon_days=14,
                                     household_basis_gbp=200000.0)
        pltr = [i for i in radar["items"] if i.get("root") == "PLTR"][0]
        self.assertTrue(pltr["action_needed"])
        self.assertIn("before the results", pltr["what_to_do"])
        self.assertIn("results due Today 21:00 UK: place this before them", sell["event_note"])
        self.assertIn("need", radar["headline"])
        tldr = se.generate_tldr_summary(results)["Main"]
        self.assertIn("⏰ results due Today 21:00 UK", tldr["sells_to_place"][0]["instruction"])

    def test_bellwether_note(self):
        res = {"Main": {"analyses": [_pos("NVDA", 5000.0, "Nov 17, 2026", "2026-11-17T21:00:00+00:00", False, "NVIDIA")],
                        "rebalancing_plan": {}}}
        it = ec.holdings_earnings_events(res, date(2026, 11, 10), 14, household_basis_gbp=500000.0)[0]
        self.assertEqual(it["impact"], "high")
        self.assertIn("semiconductor", it["why"])

    def test_outside_holdings_results_use_the_injected_fetcher(self):
        calls = []

        def fake_fetch(t):
            calls.append(t)
            return {"earnings_date_str": "Nov 05, 2026", "earnings_ts_utc": "2026-11-05T21:00:00+00:00",
                    "earnings_is_estimate": False}

        rows = [{"ticker": "XYZ", "account": "Main", "vested_gbp": 30000.0},
                {"ticker": "XYZ", "account": "Main", "vested_gbp": 0.0},
                {"ticker": "ABC", "account": "Main", "vested_gbp": 0.0, "unvested_gbp": 9000.0}]
        radar = ec.build_event_radar({}, as_of=MON_2_NOV, horizon_days=14, household_basis_gbp=200000.0,
                                     outside_rows=rows, earnings_fetcher=fake_fetch)
        self.assertEqual(calls, ["XYZ"])  # unvested-only rows are not looked up
        it = [i for i in radar["items"] if i.get("source") == "earnings_outside"][0]
        self.assertEqual(it["impact"], "high")  # 15% of the household
        self.assertIn("outside T212", it["holders"])
        self.assertFalse(it["action_needed"])
        self.assertEqual(ec.outside_earnings_events(rows, MON_2_NOV, 14, 200000.0, lambda t: {}), [])

    def test_research_timing_only_attaches_a_matching_timestamp(self):
        info = {"earningsTimestampStart": 1793653200, "isEarningsDateEstimate": False}  # 2026-11-02 21:00 UTC
        self.assertEqual(DeepResearchAgent._earnings_timing(info, "Nov 02, 2026"), ("2026-11-02T21:00:00+00:00", False))
        stale = {"earningsTimestamp": 1788123600, "isEarningsDateEstimate": True}     # 2026-08-30: last quarter
        self.assertEqual(DeepResearchAgent._earnings_timing(stale, "Dec 01, 2026"), (None, True))
        self.assertEqual(DeepResearchAgent._earnings_timing({}, None), (None, None))
        self.assertEqual(DeepResearchAgent._earnings_timing({"earningsTimestamp": "bad"}, "Nov 02, 2026"), (None, None))


class CircuitBreakerTests(unittest.TestCase):
    def test_trip_level_is_the_mean_of_the_last_n_minus_one_month_ends(self):
        idx = pd.bdate_range("2025-01-01", "2026-11-02")
        closes = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)
        r = se.tech_regime_from_closes(closes, as_of=MON_2_NOV, sma_months=10, proxy="TEST")
        me = closes[closes.index < "2026-11-01"].groupby(closes[closes.index < "2026-11-01"].index.to_period("M")).last()
        self.assertAlmostEqual(r["trip_level"], round(float(me.iloc[-9:].mean()), 4), places=3)
        self.assertEqual(r["latest_close_date"], "2026-11-02")
        self.assertAlmostEqual(r["trip_distance_pct"], round((r["trip_level"] / r["latest_close"] - 1) * 100, 1))
        self.assertLess(r["trip_distance_pct"], 0)

    def test_event_text_names_the_check_date_level_and_action(self):
        regime = {"regime": "RISK_ON", "proxy": "CNX1.L", "trip_level": 120329.0, "latest_close": 122120.0,
                  "trip_distance_pct": -1.5}
        with mock.patch.object(config, "ALLOCATION_CONFIG", dict(ALLOC_55_20), create=True):
            ev = ec.circuit_breaker_event(MON_2_NOV, regime)
        self.assertEqual(ev["date"], "2026-11-30")
        self.assertEqual(ev["impact"], "high")                 # within 3% of tripping
        self.assertIn("120,329", ev["what_to_do"])
        self.assertIn("Tue 1 Dec email", ev["what_to_do"])
        self.assertIn("cut satellites to 20%", ev["what_to_do"])
        self.assertIn("It is close", ev["why"])
        self.assertIsNone(ec.circuit_breaker_event(MON_2_NOV, {"regime": "UNKNOWN"}))


class AssemblyAndRenderingTests(unittest.TestCase):
    def test_order_timing_lines_by_exchange(self):
        results = _results(sells_main=[{"ticker": "SMGB.L", "search_symbol": "SMGB", "action": "TRIM"}],
                           buys_main=[{"ticker": "V", "search_symbol": "V"}])
        lines = ec.order_timing_lines(results, MON_2_NOV)
        self.assertTrue(any(l.startswith("London-listed (SMGB): place between 08:15 and 16:15 UK") for l in lines))
        self.assertTrue(any(l.startswith("US-listed (V): place between 14:45 and 20:45 UK") for l in lines))

    def test_radar_survives_garbage_and_sorts_by_date(self):
        radar = ec.build_event_radar({"A": None, "B": {"analyses": [None, {"research": "bad"}], "rebalancing_plan": []}},
                                     tech_regime="bad", as_of=MON_2_NOV, horizon_days=14)
        dates = [i["date"] for i in radar["items"]]
        self.assertEqual(dates, sorted(dates))
        self.assertTrue(radar["headline"])
        self.assertTrue(ec.radar_text_lines(radar)[0].startswith("📅 EVENT RADAR"))

    def test_email_card_renders_and_escapes(self):
        radar = ec.build_event_radar(_results(), tech_regime=None, as_of=MON_2_NOV, horizon_days=14,
                                     household_basis_gbp=200000.0)
        radar["items"].append({"date": "2026-11-04", "day_label": "Wed 4 Nov", "title": "<script>x</script>",
                               "impact": "low", "what_to_do": "a & b", "kind": "data"})
        html = mailer.build_event_radar_html({"event_radar": radar})
        self.assertIn("Event radar", html)
        self.assertIn("Palantir (PLTR) results", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>x", html)
        self.assertEqual(mailer.build_event_radar_html({}), "")


if __name__ == "__main__":
    unittest.main()

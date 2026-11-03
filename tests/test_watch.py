"""Offline tests for the intraday watcher (watch.py) and its persisted state (action_state.py)."""

import os
import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import action_state  # noqa: E402
import config  # noqa: E402
import watch  # noqa: E402

LON = config.LONDON_TZ


def at(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=LON)


TUE_0907 = at(2026, 11, 3, 9, 7)
ACTIONS = {
    "date": "2026-11-02",
    "accounts": {
        "Main": {"sells": [{"ticker": "SMGB.L", "search_symbol": "SMGB", "name": "Semis fund", "shares": 33.0,
                            "estimated_gbp": 2370.0, "qty_before": 217.0, "order_reason": "glide_rebalance"},
                           {"ticker": "PLTR", "search_symbol": "PLTR", "name": "Palantir", "shares": 10.0,
                            "estimated_gbp": 1300.0, "qty_before": 60.0, "order_reason": "glide_rebalance"}],
                 "buys": []},
    },
}


def live_with(smgb_qty=217.0, pltr_qty=60.0, smgb_px=70.5, pltr_px=169.0):
    return {"Main": {"positions": [{"ticker": "SMGB.L", "qty": smgb_qty, "price": smgb_px, "ppl": -1200.0},
                                   {"ticker": "PLTR", "qty": pltr_qty, "price": pltr_px, "ppl": 2000.0}],
                     "total": 100000.0, "free": 0.0}}


class WindowAndMarketTests(unittest.TestCase):
    def test_watch_window(self):
        self.assertTrue(watch.in_watch_window(at(2026, 11, 3, 8, 0)))
        self.assertFalse(watch.in_watch_window(at(2026, 11, 3, 7, 59)))
        self.assertFalse(watch.in_watch_window(at(2026, 11, 3, 21, 0)))
        self.assertFalse(watch.in_watch_window(at(2026, 11, 7, 12, 0)))  # Saturday

    def test_market_status_by_exchange_and_holiday(self):
        self.assertTrue(watch.market_status("SMGB.L", TUE_0907)[0])
        self.assertFalse(watch.market_status("SMGB.L", at(2026, 11, 3, 16, 20))[0])  # after 16:15
        self.assertFalse(watch.market_status("PLTR", TUE_0907)[0])                   # US opens 14:30 UK
        is_open, label, window = watch.market_status("PLTR", at(2026, 11, 3, 15, 0))
        self.assertTrue(is_open)
        self.assertEqual((label, window), ("US", "between 14:45 and 20:45 UK"))
        self.assertFalse(watch.market_status("SMGB.L", at(2026, 12, 28, 10, 0))[0])  # UK bank holiday (Boxing Day sub)
        self.assertFalse(watch.market_status("PLTR", at(2026, 11, 26, 15, 0))[0])    # Thanksgiving


class ReminderTests(unittest.TestCase):
    def test_unplaced_sells_are_reminded_when_their_market_is_open(self):
        alerts = watch.unplaced_sell_alerts(ACTIONS, live_with(), TUE_0907, None)
        self.assertEqual(len(alerts), 1)
        a = alerts[0]
        self.assertEqual(a["key"], "remind|2026-11-02|Main")
        self.assertEqual(a["level"], "action")
        text = " ".join(a["lines"])
        self.assertIn("sell 33 Semis fund (SMGB) ≈ £2,370: place it now", text)
        self.assertIn("PLTR", text)
        self.assertIn("between 14:45 and 20:45 UK", text)  # US line says when, not now

    def test_partial_and_full_placement(self):
        part = watch.unplaced_sell_alerts(ACTIONS, live_with(smgb_qty=207.0), TUE_0907, None)
        self.assertIn("sell 23 Semis fund", part[0]["lines"][0])
        done = watch.unplaced_sell_alerts(ACTIONS, live_with(smgb_qty=184.0, pltr_qty=50.0), TUE_0907, None)
        self.assertEqual(done, [])

    def test_no_reminder_when_closed_stale_or_just_sent(self):
        self.assertEqual(watch.unplaced_sell_alerts(ACTIONS, live_with(smgb_qty=184.0), TUE_0907, None), [])  # only US left, closed
        self.assertEqual(watch.unplaced_sell_alerts(ACTIONS, live_with(), at(2026, 11, 9, 9, 7), None), [])   # > 3 days old
        fresh = dict(ACTIONS, date="2026-11-03")
        self.assertEqual(watch.unplaced_sell_alerts(fresh, live_with(), TUE_0907, at(2026, 11, 3, 8, 30)), [])  # 37 min ago
        self.assertEqual(len(watch.unplaced_sell_alerts(fresh, live_with(), TUE_0907, at(2026, 11, 3, 7, 0))), 1)


class BreakerTests(unittest.TestCase):
    ON = {"regime": "RISK_ON", "proxy": "CNX1.L", "trip_level": 120329.0, "latest_close": 120130.0,
          "trip_distance_pct": 0.2, "sma_months": 10}

    def test_below_the_line_is_a_heads_up_not_an_action(self):
        a = watch.breaker_alerts(self.ON, TUE_0907, "cut satellites to 20% of the household")
        self.assertEqual([x["key"] for x in a], ["breaker_below|2026-11-03"])
        self.assertEqual(a[0]["level"], "warning")
        self.assertIn("Mon 30 Nov", a[0]["lines"][0])
        above = dict(self.ON, latest_close=122000.0, trip_distance_pct=-1.4)
        self.assertEqual(watch.breaker_alerts(above, TUE_0907, ""), [])

    def test_month_end_day_and_tripped(self):
        a = watch.breaker_alerts(dict(self.ON, trip_distance_pct=-1.0), at(2026, 11, 30, 14, 37), "cut")
        self.assertEqual(a[0]["key"], "breaker_decides|2026-11-30")
        off = watch.breaker_alerts({"regime": "RISK_OFF", "as_of_month_end": "2026-11-30", "pct_vs_sma": -2.1},
                                   at(2026, 12, 1, 8, 7), "cut satellites to 20% of the household")
        self.assertEqual(off[0]["level"], "action")
        self.assertEqual(off[0]["key"], "breaker_tripped|2026-11-30")
        self.assertIn("cut satellites to 20%", off[0]["lines"][1])
        self.assertEqual(watch.breaker_alerts({"regime": "UNKNOWN"}, TUE_0907, ""), [])


class MoveAndHouseholdTests(unittest.TestCase):
    def test_big_moves_with_fund_threshold_and_unit_reconciliation(self):
        live = {"Main": {"positions": [{"ticker": "MRVL", "qty": 50, "price": 154.0},     # -10.2%
                                       {"ticker": "AAPL", "qty": 20, "price": 322.0},     # -4.8%
                                       {"ticker": "SMGB.L", "qty": 200, "price": 6650.0}, # pence vs pounds, -7.2%
                                       {"ticker": "VWRP.L", "qty": 100, "price": 137.0}]}} # -2%
        prev = {"MRVL": 171.4, "AAPL": 338.3, "SMGB.L": 71.7, "VWRP.L": 139.8}
        is_fund = lambda t: t.endswith(".L")
        alerts = watch.move_alerts(live, prev, {"MRVL"}, {"SMGB": "Semis fund"}, TUE_0907, is_fund)
        keys = sorted(a["key"] for a in alerts)
        self.assertEqual(keys, ["move|MRVL|2026-11-03", "move|SMGB|2026-11-03"])
        mrvl = [a for a in alerts if "MRVL" in a["key"]][0]
        self.assertEqual(mrvl["title"], "MRVL -10.2% today")
        self.assertIn("already on your sell list", mrvl["lines"][1])
        smgb = [a for a in alerts if "SMGB" in a["key"]][0]
        self.assertIn("no sell signal", smgb["lines"][1])

    def test_household_alert_uses_pnl_so_cash_moves_do_not_count(self):
        snap = {"combined_total": 200000.0, "total_ppl": 9000.0, "updated_at": "2026-11-02T18:11:51"}
        live = {"Main": {"positions": [{"ppl": 2000.0}], "total": 150000.0}}   # -7,000 = -3.5%
        a = watch.household_alert(live, snap, TUE_0907)
        self.assertEqual(a[0]["key"], "household|2026-11-03")
        self.assertIn("-3.5%", a[0]["title"])
        self.assertEqual(watch.household_alert({"Main": {"positions": [{"ppl": 4000.0}]}}, snap, TUE_0907), [])


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p = mock.patch.object(action_state, "DATA_DIR", __import__("pathlib").Path(self.tmp))
        self.p.start()
        action_state._save("latest_actions.json", ACTIONS)

    def tearDown(self):
        self.p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, sender, now=TUE_0907, live=None):
        with mock.patch.object(watch, "london_now", return_value=now), \
             mock.patch.object(watch, "fetch_live_accounts", return_value=live or live_with()), \
             mock.patch.object(watch, "fetch_previous_closes", return_value={}), \
             mock.patch("strategy_engine.compute_tech_regime", return_value={"regime": "UNKNOWN"}), \
             mock.patch("mailer.send_email_digest", side_effect=sender), \
             mock.patch.dict(os.environ, {"GITHUB_TOKEN": "", "GITHUB_REPOSITORY": ""}):
            return watch.run()

    def test_sends_once_then_dedupes_and_escapes(self):
        sent = []
        self._run(lambda html, subject_prefix: sent.append((subject_prefix, html)) or True)
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0][0].startswith("🚨 Action needed: Main: trims from the latest plan not placed yet"))
        self.assertIn("DO NOW", sent[0][1])
        state = action_state.load_alert_state(TUE_0907.date())
        self.assertIn("remind|2026-11-02|Main", state["sent"])
        self._run(lambda html, subject_prefix: sent.append((subject_prefix, html)) or True)
        self.assertEqual(len(sent), 1)  # nothing new the second time
        # A failed send is not remembered, so the next run retries
        action_state.save_alert_state({"date": "2026-11-03", "sent": []})
        self._run(lambda html, subject_prefix: False)
        self.assertEqual(action_state.load_alert_state(TUE_0907.date())["sent"], [])

    def test_outside_window_does_nothing(self):
        called = []
        self._run(lambda html, subject_prefix: called.append(1) or True, now=at(2026, 11, 3, 22, 0))
        self.assertEqual(called, [])

    def test_render_escapes(self):
        html = watch.render_alert_html([{"level": "warning", "title": "<b>x</b>", "lines": ["a & b"]}], TUE_0907)
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", html)
        self.assertIn("a &amp; b", html)


class ActionStateTests(unittest.TestCase):
    def test_latest_actions_and_glide_plan(self):
        results = {"Main": {
            "analyses": [{"ticker": "SMGB.L", "quantity": 217.0}, {"ticker": "SMGB.L", "quantity": 5.0, "is_earmarked": True}],
            "rebalancing_plan": {
                "sell_orders": [{"ticker": "SMGB.L", "shares": 33.0, "estimated_gbp": 2370.0, "order_reason": "glide_rebalance"},
                                {"ticker": "X", "shares": 1, "is_pending": True}],
                "buy_orders": [],
                "rebalance_glide": {"trigger": "glide_path", "is_buy_day": True,
                                    "proposals": [{"ticker": "SMGB.L", "shares": 33.0, "estimated_gbp": 2370.0}]},
            }}}
        la = action_state.build_latest_actions(results, "2026-11-02")
        self.assertEqual(len(la["accounts"]["Main"]["sells"]), 1)                 # queued orders excluded
        self.assertEqual(la["accounts"]["Main"]["sells"][0]["qty_before"], 217.0)  # earmarked line excluded
        gp = action_state.build_glide_plan(results, "2026-11-02")
        self.assertEqual((gp["month"], gp["buy_day"]), ("2026-11", "2026-11-02"))
        results["Main"]["rebalancing_plan"]["rebalance_glide"]["is_buy_day"] = False
        self.assertIsNone(action_state.build_glide_plan(results, "2026-11-03"))

    def test_alert_state_resets_each_day(self):
        tmp = tempfile.mkdtemp()
        try:
            with mock.patch.object(action_state, "DATA_DIR", __import__("pathlib").Path(tmp)):
                action_state.save_alert_state({"date": "2026-11-02", "sent": ["a"]})
                self.assertEqual(action_state.load_alert_state(date(2026, 11, 3)), {"date": "2026-11-03", "sent": []})
                self.assertEqual(action_state.load_alert_state(date(2026, 11, 2))["sent"], ["a"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

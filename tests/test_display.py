import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Offline rendering tests for the Phase 1 household-allocation display: the email digest (mailer.py), the web
# dashboard (dashboard_generator.py) and the console / plain-text helpers (main.py). Fixtures follow the engine's
# data contract (consolidated_view household_mix / big_tech / tracking_performance, per-account
# household_allocation, buy-order bucket / allocation_reason). Every test blocks DNS and sockets (and fails if a
# renderer even tries to connect), and nothing is written to disk.

import copy  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import traceback  # noqa: E402
import unittest  # noqa: E402
from typing import Any, Dict  # noqa: E402
from unittest import mock  # noqa: E402

import dashboard_generator as dg  # noqa: E402

TRACKING_TEXT = (
    "Since tracking began (19 Oct 2026): your portfolio +3.2% vs VWRP +2.1% "
    "(time-weighted, net of deposits and withdrawals)."
)
EXPLAINER = (
    "New money goes to whichever part of the mix is furthest below target. Nothing profitable is sold to rebalance."
)
BIG_TECH_LINE = "41% of household wealth (cap 30%) — no new big-tech buys"
CORE_REASON = "Core is 22% vs a 65% target, the furthest below target."
DEFENSIVE_REASON = "Defensive is 3% vs a 15% target & the cash <reserve> stays intact."
MAIN_CADENCE = "Off-cycle buy allowed: £3,000 deployable is at least the £2,000 threshold."
MAIN_NOTES = [
    "Satellites are 75% vs a 20% target: no new single-stock buys.",
    "US big tech is 41% vs the 30% cap: no buys of big-tech stocks or tech-heavy funds.",
]
SECOND_CADENCE = "Next monthly buy day is 2 Nov 2026; only sells and risk alerts run today."
MACRO = {"macro_score": 55, "risk_climate": "🟡 NEUTRAL", "crash_defcon": "🟡 DEFCON 3 — ELEVATED"}


def _html_has(html: str, text: str) -> bool:
    """True when text appears literally or JSON-escaped (strings embedded in the dashboard's <script> payloads)."""
    return text in html or json.dumps(text)[1:-1] in html


def _section(html: str, start: str, end: str) -> str:
    """Substring of html from the start marker up to the end marker (asserts both exist, in order)."""
    i = html.index(start)
    return html[i:html.index(end, i)]


def _import_or_skip(module_name: str):
    """Imports mailer / main. Skips (not fails) only when the import breaks inside ANOTHER module, e.g.
    strategy_engine while it is being edited concurrently; a failure inside the module itself still fails."""
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        frames = traceback.extract_tb(e.__traceback__)
        origin = str(getattr(e, "filename", "") or (frames[-1].filename if frames else ""))
        if os.path.basename(origin) == f"{module_name}.py":
            raise
        raise unittest.SkipTest(f"{module_name} import unavailable during concurrent edits: {e!r}")


# ---------------------------------------------------------------------------
# Fixtures (shapes follow the engine data contract)
# ---------------------------------------------------------------------------
def _big_tech_breakdown():
    rows = [
        {"account": "Main", "ticker": "GOOGL", "source": "outside", "value_gbp": 20000.0,
         "lookthrough_weight": 1.0, "exposure_gbp": 20000.0},
        {"account": "Main", "ticker": "NVDA", "source": "direct", "value_gbp": 30000.0,
         "lookthrough_weight": 1.0, "exposure_gbp": 30000.0},
        {"account": "Second", "ticker": "VUAA.L", "source": "fund", "value_gbp": 10000.0,
         "lookthrough_weight": 0.35, "exposure_gbp": 3500.0},
    ]
    # Seven smaller synthetic rows so the dashboard has more than 8 to choose from (already sorted, as contracted)
    rows += [
        {"account": "Second", "ticker": f"FIXT{i}", "source": "direct", "value_gbp": 3000.0 - 300.0 * i,
         "lookthrough_weight": 1.0, "exposure_gbp": 3000.0 - 300.0 * i}
        for i in range(7)
    ]
    return rows


def _consolidated() -> Dict[str, Any]:
    return {
        "total_net_worth_gbp": 198200.0,
        "total_invested_gbp": 191380.0,
        "total_liquid_cash_gbp": 6820.0,
        "total_blocked_cash_gbp": 0.0,
        "total_ppl_gbp": 1000.0,
        "day_over_day": {"has_prior": False},
        "account_shares": {
            "Main": {"total_value_gbp": 120000.0, "share_of_total_pct": 60.5},
            "Second": {"total_value_gbp": 78200.0, "share_of_total_pct": 39.5},
        },
        "thematic_breakdown": [],
        "household_mix": {
            "basis_gbp": 200000.0,
            "band_pp": 5.0,
            "buckets": [
                {"bucket": "core", "label": "Core global index", "target_pct": 65.0,
                 "current_gbp": 56000.0, "current_pct": 21.9, "post_trade_gbp": 60000.0, "post_trade_pct": 23.4,
                 "gap_gbp": 110400.0, "status": "under", "out_of_band": True},
                {"bucket": "satellite", "label": "Satellites (single stocks & thematic funds)", "target_pct": 20.0,
                 "current_gbp": 192000.0, "current_pct": 75.0, "post_trade_gbp": 186600.0, "post_trade_pct": 72.9,
                 "gap_gbp": -140800.0, "status": "over", "out_of_band": True},
                {"bucket": "defensive", "label": "Defensive (cash, money-market, gold, short gilts)",
                 "target_pct": 15.0, "current_gbp": 8000.0, "current_pct": 3.1, "post_trade_gbp": 9400.0,
                 "post_trade_pct": 3.7, "gap_gbp": 30400.0, "status": "under", "out_of_band": True},
            ],
            "outside_included_gbp": 20000.0,
            "excluded_cash_gbp": 0.0,
            "post_trade_available": True,
            "summary": "Core 22% (target 65%) · Satellites 75% (20%) · Defensive 3% (15%)",
        },
        "big_tech": {
            "cap_pct": 30.0,
            "exposure_gbp": 120000.0,
            "exposure_pct": 41.0,
            "post_trade_exposure_gbp": 119000.0,
            "post_trade_exposure_pct": 46.5,
            "basis_gbp": 200000.0,
            "over_cap": True,
            "tickers": ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "AVGO", "TSLA"],
            "breakdown": _big_tech_breakdown(),
            "note": "Fund look-through weights are approximations, reviewed yearly.",
        },
        "tracking_performance": {
            "available": True, "start_date": "2026-10-19", "end_date": "2026-10-23",
            "portfolio_return_pct": 3.2, "benchmark_return_pct": 2.1, "difference_pp": 1.1, "days": 4,
            "text": TRACKING_TEXT,
        },
    }


def _core_buy() -> Dict[str, Any]:
    return {
        "ticker": "VWRP.L", "search_symbol": "VWRP", "name": "Vanguard FTSE All-World", "action": "BUY",
        "is_new": False, "shares": 30.0, "price": 130.0, "currency": "GBP", "estimated_gbp": 3900.0,
        "expected_5y_cagr_pct": 7.0, "plain_english_why": "Core index top-up.",
        "app_instruction": "Open Trading 212 ➔ Search 'VWRP' ➔ Tap Buy ➔ Enter £3,900",
        "bucket": "core", "allocation_reason": CORE_REASON,
    }


def _defensive_buy() -> Dict[str, Any]:
    return {
        "ticker": "SGLN.L", "search_symbol": "SGLN", "name": "iShares Physical Gold", "action": "BUY",
        "is_new": True, "shares": 20.0, "price": 56.0, "currency": "GBP", "estimated_gbp": 1120.0,
        "plain_english_why": "Defensive sleeve top-up.",
        "app_instruction": "Open Trading 212 ➔ Search 'SGLN' ➔ Tap Buy ➔ Enter £1,120",
        "bucket": "defensive", "allocation_reason": DEFENSIVE_REASON,
    }


def _holding() -> Dict[str, Any]:
    return {
        "ticker": "VWRP.L", "search_symbol": "VWRP", "name": "Vanguard FTSE All-World", "price": 130.0,
        "currency": "GBP", "avg_price": 120.0, "quantity": 100.0, "value_gbp": 13000.0, "unrealized_pnl": 1000.0,
        "portfolio_weight_pct": 10.8, "signal": "🟢 HOLD (CORE FOUNDATION)", "plan_status": "HOLD",
        "research": {"expected_5y_cagr_pct": 7.0},
    }


def _base_plan(**extra: Any) -> Dict[str, Any]:
    plan = {
        "available_cash_gbp": 0.0, "blocked_cash_gbp": 0.0, "buffer_reserve_gbp": 500.0,
        "reserve_floor_gbp": 500.0, "reserve_funded_gbp": 0.0, "reserve_shortfall_gbp": 500.0,
        "net_budget_to_deploy_gbp": 0.0, "total_deployable_gbp": 0.0, "cash_from_sells_gbp": 0.0,
        "sell_orders": [], "pending_sell_orders": [], "buy_orders": [], "hold_items": [], "table_items": [],
        "thematic_breakdown": [], "final_cash_balance_gbp": 0.0,
    }
    plan.update(extra)
    return plan


def _results() -> Dict[str, Dict[str, Any]]:
    holding = _holding()
    main_plan = _base_plan(
        available_cash_gbp=6520.0, buffer_reserve_gbp=1500.0, reserve_floor_gbp=1500.0,
        reserve_funded_gbp=1500.0, reserve_shortfall_gbp=0.0, net_budget_to_deploy_gbp=3000.0,
        total_deployable_gbp=6520.0, final_cash_balance_gbp=1500.0,
        buy_orders=[_core_buy(), _defensive_buy()], table_items=[holding],
        household_allocation={
            "buys_allowed": True, "cadence": "monthly", "is_buy_day": False, "next_buy_day": "2026-11-02",
            "cadence_note": MAIN_CADENCE, "deployable_gbp": 3000.0,
            "by_bucket_gbp": {"core": 3900.0, "satellite": 0.0, "defensive": 1120.0},
            "notes": list(MAIN_NOTES),
        },
    )
    second_plan = _base_plan(
        available_cash_gbp=300.0,
        household_allocation={
            "buys_allowed": False, "cadence": "monthly", "is_buy_day": False, "next_buy_day": "2026-11-02",
            "cadence_note": SECOND_CADENCE, "deployable_gbp": 0.0,
            "by_bucket_gbp": {"core": 250.0, "satellite": 0.0, "defensive": 50.0},
            "notes": [],
        },
    )
    return {
        "Main": {"cash": {"free": 6520.0, "free_gbp": 6520.0, "total": 120000.0, "ppl": 900.0},
                "analyses": [holding], "rebalancing_plan": main_plan},
        "Second": {"cash": {"free": 300.0, "free_gbp": 300.0, "total": 78200.0, "ppl": 100.0},
                 "analyses": [], "rebalancing_plan": second_plan},
    }


def _without_phase1(results: Dict[str, Dict[str, Any]], cv: Dict[str, Any]):
    """The same fixtures as an engine that predates Phase 1 would produce them (none of the new keys)."""
    results, cv = copy.deepcopy(results), copy.deepcopy(cv)
    for key in ("household_mix", "big_tech", "tracking_performance"):
        cv.pop(key, None)
    for data in results.values():
        plan = data["rebalancing_plan"]
        plan.pop("household_allocation", None)
        for order in plan.get("buy_orders", []):
            order.pop("bucket", None)
            order.pop("allocation_reason", None)
    return results, cv


def _malformed(results: Dict[str, Dict[str, Any]], cv: Dict[str, Any]):
    """Garbage in every Phase 1 slot: wrong types, NaN, None, unknown enums."""
    results, cv = copy.deepcopy(results), copy.deepcopy(cv)
    cv["household_mix"] = {
        "basis_gbp": "lots", "band_pp": float("nan"), "post_trade_available": "maybe",
        "buckets": [None, 5, "core", {"bucket": "core"}, {"bucket": "mystery", "current_pct": "x"},
                    {"label": "Odd", "current_pct": float("inf"), "target_pct": None, "status": 7}],
    }
    cv["big_tech"] = {"exposure_pct": "abc", "exposure_gbp": None, "breakdown": "nope", "over_cap": "yes"}
    cv["tracking_performance"] = ["not", "a", "dict"]
    main_plan = results["Main"]["rebalancing_plan"]
    main_plan["household_allocation"] = {
        "notes": [None, {"a": 1}, 5, "  "], "by_bucket_gbp": {"core": "NaN", "defensive": None},
        "cadence_note": None, "buys_allowed": "sometimes",
    }
    main_plan["buy_orders"][0]["bucket"] = 7
    main_plan["buy_orders"][0]["allocation_reason"] = None
    main_plan["buy_orders"][1]["bucket"] = {"bucket": "core"}
    results["Second"]["rebalancing_plan"]["household_allocation"] = "monthly"
    return results, cv


class _OfflineMixin:
    """Blocks DNS and outbound sockets for each test and fails if a renderer attempted a connection."""

    def setUp(self):
        super().setUp()
        self._net_calls = []
        for target, attr in ((socket.socket, "connect"), (socket, "getaddrinfo"), (socket, "create_connection")):
            patcher = mock.patch.object(target, attr, side_effect=self._blocked)
            patcher.start()
            self.addCleanup(patcher.stop)
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def _blocked(self, *args, **kwargs):
        self._net_calls.append(args)
        raise OSError("network access is blocked in offline display tests")

    def tearDown(self):
        self.assertEqual(self._net_calls, [], "a renderer attempted a network connection")
        super().tearDown()


# ---------------------------------------------------------------------------
# Email digest (mailer.py)
# ---------------------------------------------------------------------------
class DigestPhase1Tests(_OfflineMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mailer = _import_or_skip("mailer")

    def _render(self, results=None, cv=None):
        return self.mailer.generate_html_digest(
            results if results is not None else _results(), macro_regime=MACRO,
            consolidated_view=cv if cv is not None else _consolidated(),
        )

    def test_household_mix_card_renders_near_the_top(self):
        html = self._render()
        card = _section(html, "<!-- 🧭 HOUSEHOLD MIX", "<!-- Consolidated Portfolio Dashboard -->")
        self.assertIn("Household mix: all accounts + vested outside shares", card)
        self.assertIn("<th style=\"padding: 8px; text-align: right;\">After today's plan</th>", card)
        for label in ("Core global index", "Satellites (single stocks &amp; thematic funds)",
                      "Defensive (cash, money-market, gold, short gilts)"):
            self.assertIn(label, card)
        for value in (">21.9%<", ">65%<", ">23.4%<", ">75%<", ">20%<", ">72.9%<", ">3.1%<", ">15%<", ">3.7%<"):
            self.assertIn(value, card)
        # Status chips: under target = amber, over target = red
        self.assertEqual(len(re.findall(r"color: #b45309;[^>]*>Under target</span>", card)), 2)
        self.assertEqual(len(re.findall(r"color: #b91c1c;[^>]*>Over target</span>", card)), 1)
        self.assertIn(
            "Basis £200,000: Trading 212 investable money (earmarked funds excluded) + £20,000 vested shares held "
            "outside Trading 212 · band ±5pp", card,
        )
        self.assertIn(f"US big tech:</b> <span style=\"color: #b91c1c; font-weight: 700;\">{BIG_TECH_LINE}</span>", card)
        self.assertIn("Fund look-through weights are approximations, reviewed yearly.", card)
        self.assertIn(TRACKING_TEXT, card)
        self.assertIn(EXPLAINER, card)
        self.assertIn("<table role=\"presentation\"", card)  # table-friendly for email clients
        # After the TL;DR, before the consolidated overview and the per-account detail
        self.assertLess(html.index("What You Need To Do Today</h2>"), html.index("<!-- 🧭 HOUSEHOLD MIX"))
        self.assertLess(html.index("<!-- 🧭 HOUSEHOLD MIX"), html.index("Combined Portfolio Overview"))
        self.assertLess(html.index("Combined Portfolio Overview"), html.index("'s Account Actions"))

    def test_buy_cards_show_bucket_badge_reason_and_info_only_cagr(self):
        html = self._render()
        self.assertIn(">CORE</span>", html)
        self.assertIn(">DEFENSIVE</span>", html)
        self.assertNotIn(">SATELLITE</span>", html)  # no satellite buy in the fixture
        self.assertIn(f"🧭 <b>Allocation:</b> {CORE_REASON}</div>", html)
        self.assertIn("Defensive is 3% vs a 15% target &amp; the cash &lt;reserve&gt; stays intact.", html)
        self.assertNotIn("<reserve>", html)  # engine text is escaped
        tldr = _section(html, "<!-- ⚡ 30-SECOND TL;DR", "<!-- 🧭 HOUSEHOLD MIX")
        self.assertIn("<b>BUY Vanguard FTSE All-World (VWRP)</b> <span", tldr)
        self.assertIn(">CORE</span> ➔ Enter <b>£3,900</b>", tldr)
        self.assertIn(">DEFENSIVE</span> ➔ Enter <b>£1,120</b>", tldr)
        self.assertIn("Fwd 5Y CAGR (model estimate, info only): +7%", html)  # buy card
        self.assertIn("Fwd 5Y CAGR (model estimate, info only):</span>", html)  # research dossier
        self.assertNotIn("Fwd 5Y CAGR (model):", html)

    def test_cadence_note_and_notes_per_account(self):
        html = self._render()
        main = _section(html, "👤 Main's Account Actions", "<!-- 2-Minute Action Cards -->")
        self.assertNotIn("New money today", main)  # the block sits under the action heading, not above it
        main_actions = _section(html, "👤 Main's Account Actions", "<!-- Thematic Concentration Dashboard -->")
        self.assertIn(f"🗓️ <b>New money today:</b> {MAIN_CADENCE}", main_actions)
        self.assertIn("🧭 <b>New money split:</b> Core £3,900 · Defensive £1,120", main_actions)
        for note in MAIN_NOTES:
            self.assertIn(f"<li style='margin-bottom: 2px;'>{note}</li>", main_actions)
        second_actions = _section(html, "👤 Second's Account Actions", "<!-- Thematic Concentration Dashboard -->")
        self.assertIn(f"🗓️ <b>New money today:</b> {SECOND_CADENCE}", second_actions)
        self.assertNotIn("New money split", second_actions)  # buys not allowed today: no hypothetical split
        tldr = _section(html, "<!-- ⚡ 30-SECOND TL;DR", "<!-- 🧭 HOUSEHOLD MIX")
        self.assertIn(f"🗓️ <b>New money today:</b> {MAIN_CADENCE}", tldr)
        self.assertIn(f"🗓️ <b>New money today:</b> {SECOND_CADENCE}", tldr)

    def test_in_band_under_cap_and_no_post_trade_wording(self):
        cv = _consolidated()
        for bucket in cv["household_mix"]["buckets"]:
            bucket["status"], bucket["out_of_band"] = "in_band", False
        cv["household_mix"]["post_trade_available"] = False
        cv["big_tech"].update({"exposure_pct": 24.2, "over_cap": False})
        html = self._render(cv=cv)
        card = _section(html, "<!-- 🧭 HOUSEHOLD MIX", "<!-- Consolidated Portfolio Dashboard -->")
        self.assertEqual(len(re.findall(r"color: #047857;[^>]*>In band</span>", card)), 3)
        self.assertNotIn("Under target", card)
        self.assertIn("24% of household wealth (cap 30%)</span>", card)
        self.assertNotIn("no new big-tech buys", card)
        self.assertEqual(card.count("white-space: nowrap;\">—</td>"), 3)  # after-plan unknown, never faked
        self.assertIn("After-plan figures are unavailable today (no plan applied).", card)

    def test_partial_data_renders_only_what_exists(self):
        cv = _consolidated()
        del cv["household_mix"], cv["big_tech"]
        card = _section(self._render(cv=cv), "<!-- 🧭 HOUSEHOLD MIX", "<!-- Consolidated Portfolio Dashboard -->")
        self.assertIn(TRACKING_TEXT, card)
        self.assertNotIn("<table", card)
        self.assertNotIn("US big tech", card)
        self.assertNotIn(EXPLAINER, card)  # the explainer belongs to the mix table

    def test_plain_text_alternative_carries_the_household_mix(self):
        text = self.mailer.html_to_plain_text(self._render())
        self.assertIn("Household mix: all accounts + vested outside shares", text)
        self.assertIn("Satellites (single stocks & thematic funds)", text)
        self.assertIn(BIG_TECH_LINE, text)
        self.assertIn(TRACKING_TEXT, text)
        self.assertNotIn("<span", text)

    def test_without_phase1_fields_renders_without_new_sections(self):
        results, cv = _without_phase1(_results(), _consolidated())
        for consolidated in (cv, None):  # None: the digest falls back to per-account copies (none here)
            html = self._render(results=results, cv=consolidated) if consolidated is not None else (
                self.mailer.generate_html_digest(results, macro_regime=MACRO, consolidated_view=None)
            )
            self.assertIn("What You Need To Do Today", html)
            self.assertNotIn("HOUSEHOLD MIX", html)
            self.assertNotIn("Household mix:", html)
            self.assertNotIn("New money today", html)
            self.assertNotIn("Allocation:</b>", html)
            for badge in (">CORE</span>", ">SATELLITE</span>", ">DEFENSIVE</span>"):
                self.assertNotIn(badge, html)

    def test_malformed_phase1_fields_never_crash(self):
        results, cv = _malformed(_results(), _consolidated())
        html = self._render(results=results, cv=cv)
        self.assertIn("What You Need To Do Today", html)
        card = _section(html, "<!-- 🧭 HOUSEHOLD MIX", "<!-- Consolidated Portfolio Dashboard -->")
        self.assertIn("Core global index", card)  # label falls back to the bucket's default name
        self.assertIn(">N/A<", card)  # missing numbers read N/A, never an invented value
        self.assertNotRegex(card, r"(?i)\b(nan|inf)\b")  # NaN / inf never leak into the text
        self.assertNotIn("US big tech", card)  # no usable exposure figure
        self.assertNotIn(">CORE</span>", html)  # unknown / non-string buckets show no badge


# ---------------------------------------------------------------------------
# Dashboard (dashboard_generator.py)
# ---------------------------------------------------------------------------
class DashboardPhase1Tests(_OfflineMixin, unittest.TestCase):
    def _render(self, results=None, cv=None):
        return dg.generate_dashboard_html(
            results_by_account=results if results is not None else _results(),
            consolidated_view=cv if cv is not None else _consolidated(),
            macro_regime=MACRO, gbp_rate=1.30, history=None, watchlist_scan=None,
        )

    def test_household_panel_renders(self):
        html = self._render()
        panel = _section(html, "<!-- Household Mix (strategic allocation", "<!-- Account View Selector Tabs -->")
        self.assertIn("🧭 Household Mix — all accounts + vested outside shares, as one household", panel)
        self.assertIn("<th class=\"mix-num\">After today's plan</th>", panel)
        self.assertIn("Satellites (single stocks &amp; thematic funds)", panel)
        self.assertEqual(panel.count('<span class="chip chip-under">Under target</span>'), 2)
        self.assertEqual(panel.count('<span class="chip chip-over">Over target</span>'), 1)
        for value in ("<b>21.9%</b>", ">65%<", ">23.4%<", "<b>75%</b>", ">72.9%<"):
            self.assertIn(value, panel)
        self.assertIn(f'<span style="color: var(--rose);">{BIG_TECH_LINE}</span>', panel)
        self.assertIn("After today&#x27;s plan: 46.5%", panel)
        self.assertIn(TRACKING_TEXT, panel)
        self.assertIn(EXPLAINER, panel)
        # Big-tech breakdown: top 8 rows by exposure only
        for ticker in ("GOOGL", "NVDA", "VUAA.L", "FIXT0", "FIXT4"):
            self.assertIn(f"<td><b>{ticker}</b><div class=\"mix-sub-cell\">", panel)
        for ticker in ("FIXT5", "FIXT6"):
            self.assertNotIn(ticker, panel)
        self.assertIn("<b>VUAA.L</b><div class=\"mix-sub-cell\">Second</div>", panel)
        self.assertIn("<td>Fund look-through</td><td class=\"mix-num\">£10,000</td><td class=\"mix-num\">35%</td>", panel)
        self.assertIn("<td class=\"mix-num\"><b>£3,500</b></td>", panel)
        self.assertIn("<td>Vested, outside T212</td>", panel)
        self.assertIn("Top 8 of 10 holdings shown · Fund look-through weights are approximations, reviewed yearly.", panel)
        # Per-account buy cadence
        self.assertIn("🗓️ New money today, by account", panel)
        self.assertIn(f"<b>👤 Main:</b> {dg._html.escape(MAIN_CADENCE)}", panel)
        self.assertIn("New money split: Core £3,900 · Defensive £1,120", panel)
        self.assertIn(f"<b>👤 Second:</b> {dg._html.escape(SECOND_CADENCE)}", panel)
        # Household-wide panel sits above the account tabs, below the KPI row
        self.assertLess(html.index("id=\"kpiReserve\""), html.index("<!-- Household Mix (strategic allocation"))

    def test_buy_cards_carry_bucket_badges_in_the_json_payload(self):
        html = self._render()
        payload = html[html.index("const ACTION_CARDS = "):html.index("const ACTION_BADGES = ")]
        self.assertTrue(_html_has(payload, '<span class="bucket-badge bucket-core">CORE</span>'))
        self.assertTrue(_html_has(payload, '<span class="bucket-badge bucket-defensive">DEFENSIVE</span>'))
        self.assertTrue(_html_has(payload, f"<b>Allocation:</b> {CORE_REASON}</div>"))
        self.assertTrue(_html_has(payload, "the cash &lt;reserve&gt; stays intact."))
        cards = json.loads(payload[len("const ACTION_CARDS = "):].strip().rstrip(";"))
        self.assertIn("bucket-badge bucket-core", "".join(cards["Main"]))
        self.assertIn("bucket-badge bucket-core", "".join(cards["ALL"]))

    def test_blended_cagr_is_labelled_model_estimate_info_only(self):
        html = self._render()
        self.assertIn("Est. 5Y CAGR (blended hist./model estimate, info only):", html)
        self.assertNotIn("Est. 5Y CAGR (blended hist./model):", html)

    def test_engine_strings_cannot_break_out_of_html_or_script(self):
        results, cv = _results(), _consolidated()
        evil = "</script><script>alert(1)</script>"
        results["Main"]["rebalancing_plan"]["buy_orders"][0]["allocation_reason"] = evil
        results["Main"]["rebalancing_plan"]["household_allocation"]["notes"].append(evil)
        cv["big_tech"]["breakdown"][0]["ticker"] = evil
        cv["household_mix"]["buckets"][0]["label"] = evil
        cv["tracking_performance"]["text"] = evil
        html = self._render(results=results, cv=cv)
        self.assertEqual(html.count("</script>"), 1)  # only the dashboard's own script block closes
        self.assertNotIn("<script>alert(1)", html)
        self.assertIn("&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;", html)

    def test_without_phase1_fields_renders_without_the_panel(self):
        results, cv = _without_phase1(_results(), _consolidated())
        html = self._render(results=results, cv=cv)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertNotIn("<!-- Household Mix (strategic allocation", html)
        self.assertFalse(_html_has(html, 'class="bucket-badge'))
        self.assertFalse(_html_has(html, "<b>Allocation:</b>"))
        html_minimal = dg.generate_dashboard_html({}, {}, {}, None)
        self.assertNotIn("<!-- Household Mix (strategic allocation", html_minimal)

    def test_malformed_phase1_fields_never_crash(self):
        results, cv = _malformed(_results(), _consolidated())
        html = self._render(results=results, cv=cv)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertFalse(_html_has(html, 'class="bucket-badge'))
        self.assertNotIn("US big tech", html)


class SharedViewHelperTests(unittest.TestCase):
    """dashboard_generator view helpers shared by the email, the dashboard and the console."""

    def test_missing_inputs_give_empty_views(self):
        for cv in (None, {}, {"household_mix": []}, {"household_mix": {"buckets": "x"}}, "garbage"):
            self.assertEqual(dg._household_mix_view(cv), {})
            self.assertEqual(dg._big_tech_view(cv), {})
            self.assertEqual(dg._tracking_text(cv), "")
        for plan in (None, {}, {"household_allocation": None}, {"household_allocation": {"notes": []}}):
            self.assertEqual(dg._allocation_view(plan), {})
        self.assertEqual(dg._bucket_label(None), "")
        self.assertEqual(dg._bucket_label("Satellites"), "SATELLITE")
        self.assertEqual(dg._bucket_label(" Core "), "CORE")

    def test_status_is_derived_when_the_engine_omits_it(self):
        cv = {"household_mix": {"band_pp": 5.0, "buckets": [
            {"bucket": "core", "current_pct": 50.0, "target_pct": 65.0},
            {"bucket": "satellite", "current_pct": 22.0, "target_pct": 20.0},
            {"bucket": "defensive", "current_pct": 28.0, "target_pct": 15.0, "out_of_band": True},
        ]}}
        rows = dg._household_mix_view(cv)["rows"]
        self.assertEqual([r["status"] for r in rows], ["under", "in_band", "over"])
        self.assertEqual([r["label"] for r in rows], [dg._BUCKET_LABELS[k] for k in dg._BUCKET_ORDER])

    def test_big_tech_line_from_amounts_and_cap(self):
        view = dg._big_tech_view({"big_tech": {"exposure_gbp": 20000.0, "basis_gbp": 200000.0, "cap_pct": 30.0}})
        self.assertEqual(view["line"], "10% of household wealth (cap 30%)")
        self.assertFalse(view["over_cap"])
        over = dg._big_tech_view({"big_tech": {"exposure_pct": 31.0, "cap_pct": 30.0}})
        self.assertEqual(over["line"], "31% of household wealth (cap 30%) — no new big-tech buys")

    def test_allocation_view_cadence_fallback_and_split(self):
        waiting = dg._allocation_view({"household_allocation": {
            "buys_allowed": False, "next_buy_day": "2026-11-02", "by_bucket_gbp": {"core": 900.0},
        }})
        self.assertEqual(waiting["cadence"], "New buys wait for the next monthly buy day (2026-11-02).")
        self.assertEqual(waiting["split_text"], "")
        buying = dg._allocation_view({"household_allocation": {
            "buys_allowed": True, "notes": "single note", "by_bucket_gbp": {"satellite": 400.0, "core": 0.2},
        }})
        self.assertEqual(buying["notes"], ["single note"])
        self.assertEqual(buying["split_text"], "Satellites £400")


# ---------------------------------------------------------------------------
# Console / plain-text summary helpers (main.py)
# ---------------------------------------------------------------------------
class ConsoleSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = logging.getLogger()
        handlers_before, level_before = list(root.handlers), root.level
        try:
            cls.main = _import_or_skip("main")
        finally:
            # main.py calls logging.basicConfig() at import: undo it so later test modules stay as quiet as before
            for handler in list(root.handlers):
                if handler not in handlers_before:
                    root.removeHandler(handler)
            root.setLevel(level_before)

    def test_household_mix_block(self):
        text = "\n".join(self.main._household_mix_console_lines(_consolidated()))
        self.assertIn("🧭 HOUSEHOLD MIX", text)
        self.assertIn(
            "  • Core global index: now 21.9% | target 65% | after plan 23.4%  ⚠️ OUT OF BAND (under target)", text
        )
        self.assertIn(
            "  • Satellites (single stocks & thematic funds): now 75% | target 20% | after plan 72.9%"
            "  ⚠️ OUT OF BAND (over target)", text,
        )
        self.assertIn(f"  🏢 US big tech: {BIG_TECH_LINE} | after plan 46.5%", text)
        self.assertIn(f"  📈 {TRACKING_TEXT}", text)
        self.assertIn(f"  💡 {EXPLAINER}", text)
        self.assertIn("band ±5pp", text)

    def test_household_mix_block_is_empty_without_data(self):
        for cv in (None, {}, _without_phase1(_results(), _consolidated())[1]):
            self.assertEqual(self.main._household_mix_console_lines(cv), [])
        cv = _consolidated()
        for bucket in cv["household_mix"]["buckets"]:
            bucket["status"] = "in_band"
        cv["household_mix"]["post_trade_available"] = False
        text = "\n".join(self.main._household_mix_console_lines(cv))
        self.assertIn("after plan n/a  ✅ within band", text)
        self.assertNotIn("OUT OF BAND", text)

    def test_bucket_tag_and_allocation_lines(self):
        self.assertEqual(self.main._bucket_tag({"bucket": "core"}), "[CORE] ")
        self.assertEqual(self.main._bucket_tag({"bucket": "satellite"}), "[SATELLITE] ")
        self.assertEqual(self.main._bucket_tag({"bucket": "defensive"}), "[DEFENSIVE] ")
        for order in (None, {}, {"bucket": None}, {"bucket": 7}, {"bucket": "mystery"}):
            self.assertEqual(self.main._bucket_tag(order), "")
        main_plan = _results()["Main"]["rebalancing_plan"]
        self.assertEqual(self.main._allocation_console_lines(main_plan), [
            f"  🗓️ New money today: {MAIN_CADENCE}",
            "  🧭 New money split: Core £3,900 · Defensive £1,120",
            f"     • {MAIN_NOTES[0]}",
            f"     • {MAIN_NOTES[1]}",
        ])
        self.assertEqual(self.main._allocation_console_lines({}), [])
        self.assertEqual(self.main._allocation_console_lines(None), [])


if __name__ == "__main__":
    unittest.main()

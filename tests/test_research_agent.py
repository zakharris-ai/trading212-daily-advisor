import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Offline tests for research_agent (C2/C3 contracts, quality checklist, ETF detection, live-FX handling) plus
# offline smoke tests for dashboard_generator and mailer rendering. yfinance is always mocked: no network.

import json
import logging
import py_compile
import unittest
from typing import Any, Dict, Optional
from unittest import mock

import pandas as pd

import research_agent
from research_agent import DeepResearchAgent

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Fake yfinance objects
# ---------------------------------------------------------------------------
class FakeTicker:
    """Minimal stand-in for yfinance.Ticker used by research_agent."""

    def __init__(
        self,
        symbol: str,
        info: Optional[Dict[str, Any]] = None,
        last_price: Optional[float] = None,
        history: Optional[pd.DataFrame] = None,
        raise_info: bool = False,
        raise_fast_info: bool = False,
        balance_sheet: Optional[pd.DataFrame] = None,
        financials: Optional[pd.DataFrame] = None,
        calendar: Any = None,
    ):
        self.ticker = symbol
        self._info = info if info is not None else {}
        self._last_price = last_price
        self._history = history
        self._raise_info = raise_info
        self._raise_fast_info = raise_fast_info
        self.balance_sheet = balance_sheet
        self.financials = financials
        self.calendar = calendar
        self.earnings_history = None
        self.eps_revisions = None
        self.insider_purchases = None
        self.institutional_holders = None
        self.options = ()
        self.news = []

    @property
    def info(self) -> Dict[str, Any]:
        if self._raise_info:
            raise RuntimeError("simulated Yahoo quoteSummary failure")
        return self._info

    @property
    def fast_info(self) -> Dict[str, Any]:
        if self._raise_fast_info:
            raise RuntimeError("simulated fast_info failure")
        return {"lastPrice": self._last_price, "marketCap": None}

    def history(self, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        if self._history is None:
            return pd.DataFrame()
        return self._history

    def option_chain(self, expiry: str):  # pragma: no cover - options are empty in these fakes
        raise RuntimeError("no options")


def _ticker_factory(mapping: Dict[str, FakeTicker], fail_unknown: bool = True):
    """Returns a yf.Ticker replacement serving FakeTickers by symbol; unknown symbols raise (simulated outage)."""

    def _factory(symbol: str, *args, **kwargs):
        if symbol in mapping:
            return mapping[symbol]
        if fail_unknown:
            raise RuntimeError(f"simulated network failure for {symbol}")
        return FakeTicker(symbol)

    return _factory


def _rising_history(rows: int = 41, start: float = 100.0) -> pd.DataFrame:
    """Steadily rising closes on flat volume: zero distribution days."""
    closes = [start + i for i in range(rows)]
    return pd.DataFrame({"Close": closes, "Volume": [1_000_000] * rows})


def _healthy_stock_info(**overrides: Any) -> Dict[str, Any]:
    info = {
        "symbol": "TEST",
        "shortName": "Test Corp",
        "sector": "Technology",
        "industry": "Software",
        "currency": "USD",
        "financialCurrency": "USD",
        "marketCap": 500e9,
        "forwardPE": 25.0,
        "trailingPE": 30.0,
        "pegRatio": 1.4,
        "revenueGrowth": 0.15,
        "earningsGrowth": 0.18,
        "operatingMargins": 0.30,
        "grossMargins": 0.60,
        "returnOnEquity": 0.25,
        "currentRatio": 1.5,
        "debtToEquity": 50.0,
        "freeCashflow": 15e9,
        "operatingCashflow": 20e9,
        "netIncomeToCommon": 12e9,
        "totalRevenue": 60e9,
        "totalCash": 30e9,
        "totalDebt": 10e9,
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 30,
    }
    info.update(overrides)
    return info


def _all_string_values(obj: Any):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _all_string_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_string_values(v)
    elif isinstance(obj, str):
        yield obj


def _html_has(html: str, text: str) -> bool:
    """True when text appears literally or JSON-escaped (strings embedded in the dashboard's <script> payloads)."""
    return text in html or json.dumps(text)[1:-1] in html


class _QuietLogsMixin:
    def setUp(self):  # noqa: D401 - unittest hook
        research_agent._FX_RATE_CACHE.clear()
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)
        research_agent._FX_RATE_CACHE.clear()


# ---------------------------------------------------------------------------
# C2: deep-dive fail-closed contract
# ---------------------------------------------------------------------------
class DeepDiveFailClosedTests(_QuietLogsMixin, unittest.TestCase):
    def _assert_fail_closed(self, research: Dict[str, Any]) -> None:
        self.assertIn("data_ok", research)
        self.assertIn("expected_5y_cagr_capped", research)
        self.assertIs(research["data_ok"], False)
        self.assertEqual(research["fundamental_score"], 50)
        self.assertEqual(research["quality_grade"], "N/A")
        self.assertEqual(research["street_consensus"], "N/A")
        self.assertIsNone(research["expected_5y_cagr_pct"])
        self.assertIsNone(research["expected_5y_cagr_capped"])
        strings = list(_all_string_values(research))
        self.assertNotIn("A-", strings)
        self.assertFalse(any("Moderate Buy" in s for s in strings))
        self.assertNotEqual(research["fundamental_score"], 70)

    def test_ticker_construction_exception_is_fail_closed(self):
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=RuntimeError("network down")):
            research = DeepResearchAgent().conduct_deep_dive("NVDA")
        self._assert_fail_closed(research)
        self.assertIn("network down", str(research.get("data_error")))

    def test_info_exception_is_fail_closed(self):
        fake = FakeTicker("MSFT", raise_info=True)
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory({"MSFT": fake})):
            research = DeepResearchAgent().conduct_deep_dive("MSFT")
        self._assert_fail_closed(research)

    def test_empty_fundamentals_is_fail_closed_without_exception(self):
        fake = FakeTicker("ZZZZ", info={"trailingPegRatio": None}, last_price=10.0)
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory({"ZZZZ": fake})):
            research = DeepResearchAgent().conduct_deep_dive("ZZZZ")
        self._assert_fail_closed(research)

    def test_calendar_exception_does_not_abort_deep_dive(self):
        class ExplodingCalendarTicker(FakeTicker):
            @property
            def calendar(self):  # type: ignore[override]
                raise RuntimeError("calendar endpoint down")

            @calendar.setter
            def calendar(self, value):
                pass

        fake = ExplodingCalendarTicker("TEST", info=_healthy_stock_info(), last_price=100.0)
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory({"TEST": fake})):
            research = DeepResearchAgent().conduct_deep_dive("TEST")
        self.assertIs(research["data_ok"], True)
        self.assertIsNone(research["days_to_earnings"])
        self.assertIn("Unknown", research["earnings_catalyst_status"])

    def test_successful_deep_dive_sets_data_ok_true(self):
        fake = FakeTicker("TEST", info=_healthy_stock_info(), last_price=100.0)
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory({"TEST": fake})):
            research = DeepResearchAgent().conduct_deep_dive("TEST")
        self.assertIs(research["data_ok"], True)
        self.assertIn(research["expected_5y_cagr_capped"], (None, "cap", "floor"))
        self.assertIsNotNone(research["expected_5y_cagr_pct"])
        self.assertEqual(research["street_consensus"], "Buy")
        self.assertIsNotNone(research["fcf_yield_pct"])  # same currency: FX identity, FCF yield available

    def test_missing_recommendation_is_na_not_buy(self):
        info = _healthy_stock_info()
        info.pop("recommendationKey")
        fake = FakeTicker("TEST", info=info, last_price=100.0)
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory({"TEST": fake})):
            research = DeepResearchAgent().conduct_deep_dive("TEST")
        self.assertIs(research["data_ok"], True)
        self.assertEqual(research["street_consensus"], "N/A")


class CompoundingModelTests(_QuietLogsMixin, unittest.TestCase):
    BASE = {
        "ticker": "TESTCO",
        "roic": 15.0,
        "operating_margin_pct": 20.0,
        "gross_margin_pct": 40.0,
        "revenue_growth_yoy": 10.0,
        "earnings_growth_yoy": 10.0,
        "forward_pe": 20.0,
        "fcf_yield_pct": 3.0,
        "rule_of_40": 30.0,
    }

    def test_zero_eps_momentum_is_preserved(self):
        agent = DeepResearchAgent()
        r_zero = agent._compute_intrinsic_5y_compounding(dict(self.BASE, eps_revision_momentum_pct=0.0))
        r_neutral = agent._compute_intrinsic_5y_compounding(dict(self.BASE, eps_revision_momentum_pct=50.0))
        r_missing = agent._compute_intrinsic_5y_compounding(dict(self.BASE, eps_revision_momentum_pct=None))
        # 0% momentum (all downgrades) applies the -2.0pt penalty instead of being coerced to neutral 50
        self.assertAlmostEqual(r_neutral["expected_5y_cagr_pct"] - r_zero["expected_5y_cagr_pct"], 2.0, places=1)
        # A genuinely missing value still falls back to neutral 50
        self.assertEqual(r_missing["expected_5y_cagr_pct"], r_neutral["expected_5y_cagr_pct"])

    def test_cap_and_floor_flags(self):
        agent = DeepResearchAgent()
        hot = dict(self.BASE, roic=60.0, revenue_growth_yoy=60.0, earnings_growth_yoy=60.0, forward_pe=5.0,
                   fcf_yield_pct=12.0, eps_revision_momentum_pct=90.0)
        cold = dict(self.BASE, roic=1.0, revenue_growth_yoy=-20.0, earnings_growth_yoy=-30.0, forward_pe=400.0,
                    fcf_yield_pct=-1.0, eps_revision_momentum_pct=10.0, operating_margin_pct=2.0, gross_margin_pct=10.0)
        r_hot = agent._compute_intrinsic_5y_compounding(hot)
        r_cold = agent._compute_intrinsic_5y_compounding(cold)
        r_mid = agent._compute_intrinsic_5y_compounding(dict(self.BASE, eps_revision_momentum_pct=50.0))
        self.assertEqual(r_hot["expected_5y_cagr_pct"], 28.5)
        self.assertEqual(r_hot["expected_5y_cagr_capped"], "cap")
        self.assertEqual(r_cold["expected_5y_cagr_pct"], 5.5)
        self.assertEqual(r_cold["expected_5y_cagr_capped"], "floor")
        self.assertIsNone(r_mid["expected_5y_cagr_capped"])


# ---------------------------------------------------------------------------
# C3: macro regime never defaults to risk-on
# ---------------------------------------------------------------------------
def _macro_mapping(prices: Dict[str, Optional[float]], with_histories: bool = True) -> Dict[str, FakeTicker]:
    mapping = {sym: FakeTicker(sym, last_price=px) for sym, px in prices.items()}
    if with_histories:
        for sym in ("SPY", "QQQ", "SMH", "XLK", "XLI", "XLF", "XLV", "XLE"):
            existing = mapping.get(sym)
            mapping[sym] = FakeTicker(sym, last_price=existing._last_price if existing else None,
                                      history=_rising_history())
    return mapping


NEUTRALISH_PRICES = {
    "^VIX": 21.0, "^VIX3M": 22.0, "^TNX": 4.2, "^IRX": 4.0, "DX-Y.NYB": 103.0,
    "HYG": 72.0, "LQD": 100.0, "HG=F": 3.9, "GC=F": 3000.0, "RSP": 168.0, "SPY": 600.0,
    "TIP": 113.0, "IEF": 100.0, "CL=F": 80.0, "^GSPC": 6000.0, "^NDX": 21000.0,
    "SPHB": 99.0, "SPLV": 45.0, "SMH": 450.0,
}


class MacroRegimeTests(_QuietLogsMixin, unittest.TestCase):
    def test_all_fail_macro_is_neutral_not_risk_on(self):
        logging.disable(logging.NOTSET)
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=RuntimeError("offline")):
            with self.assertLogs("research_agent", level="WARNING") as logs:
                regime = DeepResearchAgent().get_macro_regime()
        self.assertIs(regime["macro_data_ok"], False)
        self.assertIsInstance(regime["macro_missing"], list)
        self.assertIn("VIX", regime["macro_missing"])
        self.assertIn("SPY trend", regime["macro_missing"])
        self.assertEqual(regime["macro_score"], 50)
        self.assertEqual(regime["risk_climate"], "⚪ MACRO DATA UNAVAILABLE (NEUTRAL)")
        self.assertNotIn("RISK-ON", regime["risk_climate"])
        self.assertIn("DEFCON 3", regime["crash_defcon"])
        self.assertNotIn("DEFCON 5", regime["crash_defcon"])
        self.assertFalse(any(r.levelno < logging.WARNING for r in logs.records))

    def test_healthy_macro_is_data_ok(self):
        with mock.patch.object(research_agent.yf, "Ticker",
                               side_effect=_ticker_factory(_macro_mapping(NEUTRALISH_PRICES))):
            regime = DeepResearchAgent().get_macro_regime()
        self.assertIs(regime["macro_data_ok"], True)
        self.assertEqual(regime["macro_missing"], [])
        # 50 + 4 (SPHB/SPLV 2.2x) with every other component inside its neutral band
        self.assertEqual(regime["macro_score"], 54)
        self.assertIn("BALANCED", regime["risk_climate"])
        self.assertAlmostEqual(regime["us3m_yield"], 4.0)
        self.assertIn("10Y–3M", regime["yield_curve_status"])

    def test_missing_non_key_input_scores_neutral(self):
        prices = dict(NEUTRALISH_PRICES)
        prices.pop("SPHB")
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory(_macro_mapping(prices))):
            regime = DeepResearchAgent().get_macro_regime()
        self.assertIs(regime["macro_data_ok"], True)
        self.assertIn("High-beta / low-vol (SPHB/SPLV)", regime["macro_missing"])
        self.assertIn("high_beta_ratio", regime["macro_missing_keys"])
        self.assertEqual(regime["macro_score"], 50)  # no +4 from the old 2.15x placeholder

    def test_missing_vix_forces_neutral(self):
        prices = dict(NEUTRALISH_PRICES)
        prices.pop("^VIX")
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory(_macro_mapping(prices))):
            regime = DeepResearchAgent().get_macro_regime()
        self.assertIs(regime["macro_data_ok"], False)
        self.assertEqual(regime["macro_missing"], ["VIX"])
        self.assertEqual(regime["macro_score"], 50)
        self.assertIn("UNAVAILABLE", regime["risk_climate"])

    def test_missing_spy_trend_forces_neutral(self):
        mapping = _macro_mapping(NEUTRALISH_PRICES)
        mapping["SPY"] = FakeTicker("SPY", last_price=600.0, history=_rising_history(rows=5))
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory(mapping)):
            regime = DeepResearchAgent().get_macro_regime()
        self.assertIs(regime["macro_data_ok"], False)
        self.assertIn("SPY trend", regime["macro_missing"])
        self.assertNotIn("RISK-ON", regime["risk_climate"])


# ---------------------------------------------------------------------------
# ETF detection
# ---------------------------------------------------------------------------
class EtfDetectionTests(_QuietLogsMixin, unittest.TestCase):
    def test_new_sister_lines_are_known_etfs(self):
        for t in ("VUAG.L", "CSPX.L", "CNX1.L", "VWRA.L", "VUAA.L", "SGLN.L", "CSH2.L"):
            self.assertTrue(research_agent._is_known_etf_ticker(t), t)
        self.assertFalse(research_agent._is_known_etf_ticker("NVDA"))

    def test_vuag_deep_dive_returns_etf_profile(self):
        fake = FakeTicker("VUAG.L", info={}, last_price=90.0)
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory({"VUAG.L": fake})):
            research = DeepResearchAgent().conduct_deep_dive("VUAG.L")
        self.assertIs(research.get("is_etf"), True)
        self.assertIs(research["data_ok"], True)
        self.assertIsNone(research["expected_5y_cagr_capped"])
        self.assertEqual(research["company_name"], "Vanguard S&P 500 UCITS ETF")
        self.assertEqual(research["macro_cluster"], "Diversified Index Core")

    def test_quote_type_etf_routes_to_etf_profile(self):
        fake = FakeTicker("ABCD", info={"quoteType": "ETF", "shortName": "Some Index ETF"}, last_price=50.0)
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory({"ABCD": fake})):
            research = DeepResearchAgent().conduct_deep_dive("ABCD")
        self.assertIs(research.get("is_etf"), True)
        self.assertIs(research["data_ok"], True)
        self.assertEqual(research["company_name"], "Some Index ETF")

    def test_etf_profile_failure_is_fail_closed(self):
        agent = DeepResearchAgent()
        with mock.patch.object(agent, "_build_etf_profile", side_effect=RuntimeError("profile boom")):
            research = agent.conduct_deep_dive("VUAG.L")
        self.assertIs(research.get("is_etf"), True)
        self.assertIs(research["data_ok"], False)
        self.assertEqual(research["quality_grade"], "N/A")
        self.assertEqual(research["street_consensus"], "N/A")
        self.assertIsNone(research["expected_5y_cagr_pct"])


# ---------------------------------------------------------------------------
# Quality Checklist (0-9) missing-data points
# ---------------------------------------------------------------------------
class QualityChecklistTests(_QuietLogsMixin, unittest.TestCase):
    def test_missing_current_ratio_point_is_not_awarded(self):
        agent = DeepResearchAgent()
        info = _healthy_stock_info()
        full_score, _, _ = agent._compute_piotroski_score(info)
        info_missing = dict(info)
        info_missing.pop("currentRatio")
        missing_score, label, checks = agent._compute_piotroski_score(info_missing)
        self.assertEqual(full_score, 9)
        self.assertEqual(missing_score, 8)
        self.assertTrue(label.startswith("8/9"))
        self.assertTrue(any("Current ratio not reported" in c for c in checks))

    def test_missing_debt_to_equity_needs_cash_above_debt(self):
        agent = DeepResearchAgent()
        base = _healthy_stock_info()
        base.pop("debtToEquity")
        cash_rich, _, _ = agent._compute_piotroski_score(dict(base, totalCash=30e9, totalDebt=10e9))
        indebted, _, _ = agent._compute_piotroski_score(dict(base, totalCash=5e9, totalDebt=10e9))
        unknown_debt = dict(base)
        unknown_debt.pop("totalDebt")
        unverifiable, _, _ = agent._compute_piotroski_score(unknown_debt)
        self.assertEqual(cash_rich, 9)
        self.assertEqual(indebted, 8)
        self.assertEqual(unverifiable, 8)


# ---------------------------------------------------------------------------
# Live FX divisor
# ---------------------------------------------------------------------------
class FxDivisorTests(_QuietLogsMixin, unittest.TestCase):
    def test_same_currency_is_identity_without_network(self):
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=AssertionError("no network expected")):
            self.assertEqual(research_agent._get_fin_fx_divisor({"currency": "USD", "financialCurrency": "USD"}), 1.0)
            self.assertEqual(research_agent._get_fin_fx_divisor({"currency": "GBp", "financialCurrency": "GBP"}), 1.0)

    def test_missing_currency_is_none(self):
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=AssertionError("no network expected")):
            self.assertIsNone(research_agent._get_fin_fx_divisor({"currency": "USD"}))

    def test_live_rate_is_used_and_cached(self):
        fx = FakeTicker("USDTWD=X", last_price=32.5)
        factory = mock.Mock(side_effect=_ticker_factory({"USDTWD=X": fx}))
        info = {"currency": "USD", "financialCurrency": "TWD"}
        with mock.patch.object(research_agent.yf, "Ticker", factory):
            self.assertEqual(research_agent._get_fin_fx_divisor(info), 32.5)
            self.assertEqual(research_agent._get_fin_fx_divisor(info), 32.5)
        self.assertEqual(factory.call_count, 1)

    def test_unknown_pair_returns_none_and_fx_metrics_are_none(self):
        self.assertIsNone(
            self._divisor_with_mapping({}, {"currency": "USD", "financialCurrency": "XYZ"})
        )
        balance_sheet = pd.DataFrame(
            {"2025": [100e9, 20e9, 40e9, 30e9]},
            index=["Total Assets", "Working Capital", "Retained Earnings", "Total Liabilities Net Minority Interest"],
        )
        financials = pd.DataFrame({"2025": [25e9, 60e9]}, index=["EBIT", "Total Revenue"])
        info = _healthy_stock_info(financialCurrency="XYZ")
        stock = FakeTicker("TEST", info=info, last_price=100.0, balance_sheet=balance_sheet, financials=financials)
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory({"TEST": stock})):
            research = DeepResearchAgent().conduct_deep_dive("TEST")
        self.assertIs(research["data_ok"], True)
        self.assertIsNone(research["fcf_yield_pct"])
        self.assertIsNone(research["free_cashflow_b"])
        self.assertIsNone(research["altman_z"])
        self.assertIn("FX", research["altman_label"])

        # Same statements with a matching currency do produce a Z-score (X4 available)
        research_agent._FX_RATE_CACHE.clear()
        stock_ok = FakeTicker("TEST", info=_healthy_stock_info(), last_price=100.0,
                              balance_sheet=balance_sheet, financials=financials)
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory({"TEST": stock_ok})):
            research_ok = DeepResearchAgent().conduct_deep_dive("TEST")
        self.assertIsNotNone(research_ok["altman_z"])

    @staticmethod
    def _divisor_with_mapping(mapping: Dict[str, FakeTicker], info: Dict[str, Any]) -> Optional[float]:
        with mock.patch.object(research_agent.yf, "Ticker", side_effect=_ticker_factory(mapping)):
            return research_agent._get_fin_fx_divisor(info)


# ---------------------------------------------------------------------------
# Compile + offline render smoke tests for the display modules
# ---------------------------------------------------------------------------
def _risk_profile(new_shape: bool) -> Dict[str, Any]:
    rp = {
        "current_cagr_5y_pct": 12.3, "post_cagr_5y_pct": 12.9,
        "current_sortino_5y": 1.21, "post_sortino_5y": 1.18,
        "current_beta_5y": 1.05, "post_beta_5y": 1.10,
        "current_stress_2022_pct": -24.0, "post_stress_2022_pct": -26.5,
        "current_stress_2022_gbp": -12000.0, "post_stress_2022_gbp": -13250.0,
        "current_shield_pct": 13.0, "post_shield_pct": 11.0,
        "crash_readiness_grade": "🟡 BUILDING CRASH SHIELD BUFFER",
        "projected_5y_conservative_gbp": 70000.0, "projected_5y_conservative_cagr_pct": 8.5,
        "projected_5y_target_gbp": 90000.0, "projected_5y_target_cagr_pct": 12.0,
    }
    if new_shape:
        rp.update({
            "current_defensive_pct": 13.0, "post_defensive_pct": 3.0,
            "current_defensive_gbp": 6500.0, "post_defensive_gbp": 1500.0,
            "crash_delta_gbp": -1250.0,
            "projected_5y_bear_gbp": 52000.0, "projected_5y_bear_cagr_pct": 0.8,
            "projected_5y_base_gbp": 71000.0, "projected_5y_base_cagr_pct": 7.2,
            "projected_5y_bull_gbp": 93000.0, "projected_5y_bull_cagr_pct": 13.1,
        })
    return rp


def _analysis(ticker: str, new_shape: bool, **extra: Any) -> Dict[str, Any]:
    a = {
        "ticker": ticker, "search_symbol": ticker.replace(".L", ""), "name": f"{ticker} Holding",
        "price": 100.0, "currency": "USD", "avg_price": 90.0, "quantity": 10.0, "value_gbp": 5000.0,
        "unrealized_pnl": 500.0, "portfolio_weight_pct": 10.0, "signal": "🟢 HOLD (CORE FOUNDATION)",
        "action_short": "HOLD", "plan_status": "HOLD", "plan_badge_color": "#6366f1",
        "plan_summary": "Hold.", "conviction_score": 60, "piotroski_score": 7, "altman_z": 4.2,
        "risk_reward_ratio": 2.5, "cagr_5y_pct": 0.0, "sortino_5y": 0.0, "crash_2022_ret_pct": 0.0,
        "is_weekly_stage_2": True, "weekly_trend_label": "Weekly Stage 2 Confirmed",
        "minervini_stage": "Stage 2", "trailing_stop": 95.0, "chandelier_stop": 95.0,
        "research": {"quality_grade": "B", "piotroski_label": "7/9 — Strong Institutional Grade",
                     "expected_5y_cagr_pct": 28.5, "expected_5y_cagr_capped": "cap"},
    }
    if new_shape:
        a.update({"data_ok": True, "valuation_ok": True, "stop_breached": False})
    a.update(extra)
    return a


def _results(new_shape: bool) -> Dict[str, Dict[str, Any]]:
    main_plan = {
        "available_cash_gbp": 0.0, "blocked_cash_gbp": 0.0, "buffer_reserve_gbp": 1500.0,
        "net_budget_to_deploy_gbp": 0.0, "total_deployable_gbp": 0.0, "cash_from_sells_gbp": 0.0,
        "sell_orders": [], "pending_sell_orders": [], "buy_orders": [], "hold_items": [],
        "thematic_breakdown": [], "final_cash_balance_gbp": 0.0,
        "portfolio_5y_risk_profile": _risk_profile(new_shape),
    }
    secondary_buy = {
        "ticker": "NVDA", "search_symbol": "NVDA", "name": "NVIDIA", "is_new": True, "shares": 5.0,
        "price": 180.0, "currency": "USD", "estimated_gbp": 1600.0, "risk_reward_ratio": None,
        "cagr_5y_pct": 60.0, "sortino_5y": 1.5, "crash_2022_ret_pct": -46.3, "piotroski_score": 8,
        "altman_z": 30.0, "expected_5y_cagr_pct": 28.5, "plain_english_why": "Test buy.",
        "app_instruction": "Open Trading 212 ➔ Search 'NVDA' ➔ Tap Buy ➔ Enter £1,600",
        "research": {"expected_5y_cagr_capped": "cap"},
    }
    secondary_sell = {
        "ticker": "QQQ3.L", "search_symbol": "QQQ3", "name": "Nasdaq 3x", "action": "🔴 SELL (LEVERAGE POLICY — 0% CAP)",
        "is_pending": False, "shares": 10.0, "price": 200.0, "currency": "GBP", "estimated_gbp": 2000.0,
        "plain_english_why": "Leverage policy.", "app_instruction": "Sell all.",
    }
    secondary_plan = {
        "available_cash_gbp": 5500.0, "blocked_cash_gbp": 0.0, "buffer_reserve_gbp": 500.0,
        "net_budget_to_deploy_gbp": 3000.0, "total_deployable_gbp": 5500.0, "cash_from_sells_gbp": 2000.0,
        "sell_orders": [secondary_sell], "pending_sell_orders": [], "buy_orders": [secondary_buy], "hold_items": [],
        "thematic_breakdown": [{"cluster": "AI Hardware & Semiconductors", "value_gbp": 15000.0, "weight_pct": 27.5,
                                "status_tag": "🔴 Saturated", "badge_bg": "#ef4444"}],
        "final_cash_balance_gbp": 2000.0, "portfolio_5y_risk_profile": _risk_profile(new_shape),
    }
    main_analyses = [
        _analysis("CSH2.L", new_shape, signal="🔒 HOLD (EARMARKED — EXCLUDED FROM PORTFOLIO MATHS)"),
        _analysis("VUAA.L", new_shape, stop_breached=True if new_shape else None),
    ]
    secondary_analyses = [_analysis("QQQ3.L", new_shape, signal="🔴 SELL (LEVERAGE POLICY — 0% CAP)",
                               action_short="SELL", plan_status="SELL", is_weekly_stage_2=False)]
    if new_shape:
        main_plan.update({
            "reserve_floor_gbp": 1500.0, "reserve_funded_gbp": 0.0, "reserve_shortfall_gbp": 1500.0,
            "data_degraded_reason": "Weekly trend data unavailable for VUAA.L",
            "earmarked_gbp": 20000.0, "earmarked_tickers": ["CSH2.L"],
        })
        secondary_plan.update({
            "reserve_floor_gbp": 500.0, "reserve_funded_gbp": 500.0, "reserve_shortfall_gbp": 0.0,
            "data_degraded_reason": None, "earmarked_gbp": 0.0, "earmarked_tickers": [],
        })
    main_plan["table_items"] = main_analyses
    secondary_plan["table_items"] = secondary_analyses
    main_plan["hold_items"] = list(main_analyses)
    return {
        "Main": {"cash": {"free": 0.0, "free_gbp": 0.0, "total": 50000.0, "ppl": 100.0},
                "analyses": main_analyses, "rebalancing_plan": main_plan},
        "Secondary": {"cash": {"free": 5500.0, "free_gbp": 5500.0, "total": 55000.0, "ppl": 900.0},
                 "analyses": secondary_analyses, "rebalancing_plan": secondary_plan},
    }


def _consolidated(new_shape: bool) -> Dict[str, Any]:
    cv = {
        "total_net_worth_gbp": 105000.0, "total_invested_gbp": 99500.0, "total_liquid_cash_gbp": 5500.0,
        "total_blocked_cash_gbp": 0.0, "total_ppl_gbp": 1000.0,
        "day_over_day": {"has_prior": True, "combined_delta_gbp": 0.0, "combined_delta_pct": 0.0},
        "account_shares": {
            "Main": {"total_value_gbp": 50000.0, "share_of_total_pct": 47.6},
            "Secondary": {"total_value_gbp": 55000.0, "share_of_total_pct": 52.4},
        },
        "thematic_breakdown": [
            {"cluster": "AI Hardware & Semiconductors", "value_gbp": 30000.0, "weight_pct": 28.6,
             "status_tag": "x", "badge_bg": "#ef4444"},
            {"cluster": "Diversified Index Core", "value_gbp": 20000.0, "weight_pct": 19.1,
             "status_tag": "x", "badge_bg": "#10b981"},
        ],
        "portfolio_5y_risk_profile": _risk_profile(new_shape),
    }
    if new_shape:
        cv.update({"outside_holdings": [{"ticker": "ACME", "name": "Acme Corp", "vested_gbp": 40000.0, "unvested_gbp": 60000.0}],
                   "outside_vested_gbp_total": 40000.0, "outside_unvested_gbp_total": 60000.0})
    return cv


def _macro(new_shape: bool) -> Dict[str, Any]:
    m = {"vix": 15.0, "macro_score": 75, "risk_climate": "🟢 RISK-ON / HIGH-BETA FAVORABLE",
         "crash_defcon": "🟢 DEFCON 5 — LOW CRASH RISK", "us10y_yield": 4.5, "yield_curve_spread": 0.7,
         "vix_term_structure": 0.85, "crude_oil_price": 80.0}
    if new_shape:
        m = {"vix": 15.0, "macro_score": 50, "risk_climate": "⚪ MACRO DATA UNAVAILABLE (NEUTRAL)",
             "crash_defcon": "⚪ DEFCON 3 — MACRO DATA UNAVAILABLE (NEUTRAL STANCE)", "macro_data_ok": False,
             "macro_missing": ["VIX", "SPY trend"], "macro_missing_keys": ["vix", "vix_term_structure", "spy_dist_days"]}
    return m


class DisplayModuleSmokeTests(_QuietLogsMixin, unittest.TestCase):
    def test_py_compile_all_three_files(self):
        for name in ("research_agent.py", "dashboard_generator.py", "mailer.py"):
            py_compile.compile(os.path.join(REPO_ROOT, name), doraise=True)

    def test_dashboard_renders_old_shape(self):
        import dashboard_generator
        html = dashboard_generator.generate_dashboard_html(
            results_by_account=_results(False), consolidated_view=_consolidated(False),
            macro_regime=_macro(False), gbp_rate=1.27, history=None,
        )
        self.assertIn("<!DOCTYPE html>", html)
        self.assertNotIn("ALL's Account", html)
        self.assertIn("Quality Checklist (0-9)", html)
        self.assertNotIn("Piotroski", html)
        self.assertNotIn("100% Ring-fenced safe to withdraw", html)
        self.assertIn("UNFUNDED", html)  # Main: £0 free cash vs £1,500 floor (derived from legacy keys)
        self.assertNotIn("fully preserved and liquid", html)
        self.assertNotIn("Piotroski 8–9/9", html)
        self.assertNotIn("SCREENED</span>", html)  # no watchlist scan supplied -> nothing is "screened"

    def test_dashboard_renders_new_shape(self):
        import dashboard_generator
        scan = [
            {"ticker": "NVDA", "data_ok": True, "piotroski_score": 8, "altman_z": 30.2, "is_weekly_stage_2": True,
             "weekly_trend_label": "Weekly Stage 2 Confirmed", "risk_reward_ratio": 2.4, "research": {"is_etf": False}},
            {"ticker": "MSFT", "data_ok": False, "piotroski_score": None, "research": {}},
            {"ticker": "SGLN.L", "data_ok": True, "piotroski_score": None, "is_weekly_stage_2": True,
             "risk_reward_ratio": 6.0, "research": {"is_etf": True}},
        ]
        history = [{"date": f"2026-10-{d:02d}", "combined_total": 100000.0 + d, "accounts": {"Main": 1.0, "Secondary": 2.0}}
                   for d in range(1, 6)]
        html = dashboard_generator.generate_dashboard_html(
            results_by_account=_results(True), consolidated_view=_consolidated(True),
            macro_regime=_macro(True), gbp_rate=1.27, history=history, watchlist_scan=scan,
        )
        self.assertNotIn("ALL's Account", html)
        self.assertIn("UNFUNDED — £1,500 short", html)
        self.assertNotIn("100% Ring-fenced safe to withdraw", html)
        self.assertNotIn("fully preserved and liquid", html)
        self.assertIn("DATA QUALITY WARNING", html)
        self.assertIn("Weekly trend data unavailable for VUAA.L", html)
        self.assertIn("(Trailing 5 Sessions)", html)
        self.assertIn("24% Maximum Cluster Cap", html)
        self.assertIn("plan adds £1,250 of 2022-style stress", html)
        self.assertIn("True defensive", html)
        self.assertIn("Bear £52,000", html)
        self.assertIn("Ring-fenced (excluded from portfolio maths): CSH2.L £20,000", html)
        self.assertIn("Holdings outside Trading 212 (ACME): vested £40,000 (counted in concentration caps)"
                      " + unvested £60,000 (not counted)", html)
        self.assertTrue(_html_has(html, "🔴 SELL (LEVERAGE POLICY — 0% CAP)"))
        self.assertTrue(_html_has(html, "🔒 HOLD (EARMARKED — EXCLUDED FROM PORTFOLIO MATHS)"))
        self.assertIn("badge-slate", html)  # neutral regime badge, not always green
        self.assertEqual(html.count("SCREENED</span>"), 2)  # NVDA + SGLN.L only (data_ok True)
        self.assertIn("Model version:", html)

    def test_mailer_renders_both_shapes(self):
        try:
            import mailer
        except Exception as e:  # strategy_engine is edited concurrently by other engineers
            if "mailer.py" in getattr(e, "filename", "") or "mailer" in str(e):
                raise
            self.skipTest(f"strategy_engine import unavailable during concurrent edits: {e}")
        for new_shape in (False, True):
            html = mailer.generate_html_digest(_results(new_shape), macro_regime=_macro(new_shape),
                                               consolidated_view=_consolidated(new_shape))
            self.assertIn("Quality Checklist (0-9)", html)
            self.assertNotIn("Piotroski", html)
            self.assertNotIn("Chandelier Stop", html)
            self.assertIn("Reference stop (not an order)", html)
            self.assertNotIn("SAFE TO WITHDRAW", html)
            self.assertNotIn("100% ring-fenced", html)
            self.assertIn("Model version:", html)
            self.assertIn("10Y–3M", html)
            self.assertIn("+28% (model cap)", html)
            text = mailer.html_to_plain_text(html)
            self.assertIn("Model version:", text)
            self.assertNotIn("<div", text)
            if new_shape:
                self.assertIn("UNFUNDED", html)
                self.assertIn("UNFUNDED", text)
                self.assertIn("DATA QUALITY WARNING", html)
                self.assertIn("⚠️ below stop", html)
                self.assertIn("Bear £52,000", html)
                self.assertIn("True defensive", html)
                self.assertIn("Ring-fenced (excluded from portfolio maths): CSH2.L £20,000", html)
                self.assertIn("Holdings outside Trading 212 (ACME)", html)


if __name__ == "__main__":
    unittest.main()

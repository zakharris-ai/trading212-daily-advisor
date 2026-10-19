import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Offline tests for strategy_engine: FX/GBP basis, fail-closed data handling, enforced risk limits,
# earmarked funds, outside holdings, post-trade simulation and narrative honesty. yfinance is fully mocked.

import math  # noqa: E402
import py_compile  # noqa: E402
import unittest  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from unittest import mock  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import strategy_engine as se  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_REBALANCE_CFG = {
    "min_cash_reserve_gbp": 1500.0,
    "min_cash_buffer_pct": 0.05,
    "max_position_pct": 0.15,
    "min_order_gbp": 300.0,
    "max_new_ideas_to_buy": 2,
    "target_crash_shield_pct": 0.15,
    "max_leveraged_etf_pct": 0.08,
    "max_cluster_pct": 0.24,
    "cap_applies_to_broad_etfs": False,
}

# Legacy allocation used by the pre-household tests: every pound of new money goes through stock selection, buys are
# daily and the household caps are effectively off, so those tests keep exercising the stock-selection pipeline.
LEGACY_ALLOCATION_CFG = {
    "targets": {"core": 0.0, "satellite": 1.0, "defensive": 0.0},
    "band_pp": 5.0,
    "satellite_max_single_pct": 1.0,
    "defensive_split": {"gold": 0.5, "short_gilts": 0.5},
    "core_buy_ticker": "VWRP.L",
    "gold_buy_ticker": "SGLN.L",
    "short_gilt_buy_ticker": "IGLS.L",
    "big_tech_cap_pct": 10.0,
    "buy_cadence": "daily",
    "off_cycle_min_deploy_gbp": 2000.0,
}

# The owner-approved household defaults (config.ALLOCATION_CONFIG as shipped), used by the household tests
HOUSEHOLD_ALLOCATION_CFG = {
    "targets": {"core": 0.65, "satellite": 0.20, "defensive": 0.15},
    "band_pp": 5.0,
    "satellite_max_single_pct": 0.05,
    "defensive_split": {"gold": 0.5, "short_gilts": 0.5},
    "core_buy_ticker": "VWRP.L",
    "gold_buy_ticker": "SGLN.L",
    "short_gilt_buy_ticker": "IGLS.L",
    "big_tech_cap_pct": 0.30,
    "buy_cadence": "monthly",
    "off_cycle_min_deploy_gbp": 2000.0,
}
BIG_TECH = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "AVGO", "TSLA"}
BIG_TECH_LOOKTHROUGH = {"VWRP": 0.22, "VUAA": 0.35, "VUAG": 0.35, "CNDX": 0.50, "CNX1": 0.50, "QQQ3": 1.50}


# ---------------------------------------------------------------------------
# Fakes & builders
# ---------------------------------------------------------------------------
class FakeTicker:
    """Minimal yfinance.Ticker stand-in (no network)."""

    def __init__(self, daily=None, weekly=None, fast_info=None, info=None, fail=False):
        self._daily = daily
        self._weekly = weekly
        self.fast_info = fast_info if fast_info is not None else {}
        self.info = info if info is not None else {}
        self._fail = fail

    def history(self, period="1y", interval="1d"):
        if self._fail:
            raise RuntimeError("history unavailable (offline test)")
        frame = self._weekly if interval == "1wk" else self._daily
        return frame.copy() if frame is not None else pd.DataFrame()


def ticker_factory(mapping):
    """yf.Ticker replacement: known symbols return their fake, anything else behaves like a network failure."""
    def _factory(symbol, *args, **kwargs):
        return mapping.get(symbol, FakeTicker(fail=True))
    return _factory


def weekly_frame(values, start="2021-10-04", tz=None):
    idx = pd.date_range(start=start, periods=len(values), freq="W-MON", tz=tz)
    return pd.DataFrame({"Close": np.asarray(values, dtype=float)}, index=idx)


def alternating_path(n, up, down, start=100.0):
    vals = [start]
    for i in range(1, n):
        vals.append(vals[-1] * (1.0 + (up if i % 2 == 1 else -down)))
    return vals


def daily_frame(closes, start="2025-10-01"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start=start, periods=len(closes))
    vol = [1300.0 if (i > 0 and closes[i] > closes[i - 1]) else 1000.0 for i in range(len(closes))]
    return pd.DataFrame(
        {"Open": closes, "High": closes * 1.005, "Low": closes * 0.995, "Close": closes, "Volume": vol},
        index=idx,
    )


def make_analysis(ticker, value_gbp, price_gbp=100.0, **overrides):
    """Analysis dict shaped like analyze_position output (only the keys the plan / risk maths use)."""
    cluster = overrides.pop("macro_cluster", "Enterprise Cloud & Software")
    d = {
        "ticker": ticker,
        "search_symbol": ticker,
        "name": ticker,
        "price": price_gbp,
        "price_gbp": price_gbp,
        "currency": "GBP",
        "quantity": (value_gbp / price_gbp) if price_gbp > 0 else 0.0,
        "value_gbp": value_gbp,
        "action_short": "HOLD",
        "signal": "🟡 HOLD (DO NOTHING)",
        "reason": "",
        "plain_english_why": "",
        "cagr_5y_pct": 10.0,
        "expected_5y_cagr_pct": 10.0,
        "sortino_5y": 1.0,
        "beta_5y": 1.0,
        "corr_5y": 0.7,
        "crash_2022_ret_pct": -15.0,
        "max_dd_5y_pct": -25.0,
        "macro_cluster": cluster,
        "research": {"macro_cluster": cluster},
        "data_ok": True,
        "price_data_ok": True,
        "research_data_ok": True,
        "valuation_ok": True,
        "stop_breached": False,
        "is_weekly_stage_2": True,
        "minervini_pass": 4,
        "conviction_score": 60,
        "rsi": 50.0,
        "is_extended": False,
        "is_recently_trimmed": False,
        "days_held": 120,
        "trailing_stop": price_gbp * 0.9,
        "target_price": price_gbp * 1.2,
        "risk_reward_ratio": None,
        "vcp_pct": 2.0,
        "is_crash_shield": False,
        "ema_200": price_gbp * 0.8,
        "unrealized_pnl": 10.0,
    }
    d.update(overrides)
    return d


WEEKLY_UPTREND = {
    "is_weekly_stage_2": True, "label": "🟢 Weekly Stage 2 Confirmed Bull (+3.0% vs 30w EMA)", "badge_color": "#10b981",
    "ema_10w": 0.0, "ema_30w": 0.0, "ema_50w": 0.0, "pct_vs_30w": 3.0, "cagr_5y_pct": 12.0, "vol_5y_pct": 15.0,
    "sortino_5y": 1.3, "sharpe_5y": 0.9, "max_dd_5y_pct": -20.0, "calmar_5y": 0.6, "crash_2022_ret_pct": -8.0,
    "beta_5y": 0.9, "corr_5y": 0.9, "crash_role_label": "📈 5Y Secular Growth", "crash_role_color": "#64748b",
    "fx_basis": "GBP", "data_ok": True,
}
WEEKLY_DOWNTREND = dict(WEEKLY_UPTREND, is_weekly_stage_2=False, label="🔴 Weekly Stage 4 / Below 30w EMA (-9.0%)", pct_vs_30w=-9.0)
MINERVINI_BULL = {
    "pass_count": 5, "stage_label": "Stage 2 Confirmed Bull (5/5)", "badge_bg": "#10b981", "is_stage_2": True,
    "ema_50": 0.0, "sma_150": 0.0, "sma_200": 0.0, "low_52w": 0.0, "high_52w": 0.0,
    "c1": True, "c2": True, "c3": True, "c4": True, "c5": True,
}
MINERVINI_BEAR = dict(MINERVINI_BULL, pass_count=1, stage_label="Stage 4 Distribution (1/5)", badge_bg="#ef4444", is_stage_2=False)
VOLUME_ACCUM = {
    "vol_ratio": 1.0, "up_down_vol_ratio": 1.3, "obv_20_slope": 1.0,
    "label": "🟢 Institutional Accumulation (U/D: 1.3x)", "badge_color": "#10b981", "is_accumulation": True,
}


class EngineTestCase(unittest.TestCase):
    """Isolates every test: no network, empty caches, deterministic config limits / exclusions / earmarks / outside holdings."""

    allocation_cfg = LEGACY_ALLOCATION_CFG

    def setUp(self):
        se.reset_runtime_caches()
        self._patches = [
            mock.patch.object(se.yf, "Ticker", side_effect=RuntimeError("network disabled in offline tests")),
            mock.patch.object(se, "_fetch_json", side_effect=RuntimeError("network disabled in offline tests")),
            mock.patch.dict(se.REBALANCE_CONFIG, BASE_REBALANCE_CFG),
            mock.patch.object(config, "EARMARKED_TICKERS", set(), create=True),
            mock.patch.object(config, "OUTSIDE_HOLDINGS", {}, create=True),
            mock.patch.object(se, "EXCLUDED_TICKERS", {"COIN"}),
            mock.patch.object(config, "ALLOCATION_CONFIG", dict(self.allocation_cfg), create=True),
            mock.patch.object(config, "BIG_TECH_TICKERS", set(BIG_TECH), create=True),
            mock.patch.object(config, "BIG_TECH_LOOKTHROUGH", dict(BIG_TECH_LOOKTHROUGH), create=True),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        se.reset_runtime_caches()

    def use_tickers(self, mapping):
        p = mock.patch.object(se.yf, "Ticker", side_effect=ticker_factory(mapping))
        p.start()
        self._patches.append(p)

    def plan(self, analyses, cash=0.0, opps=None, **kwargs):
        kwargs.setdefault("min_cash_reserve_gbp", 1500.0)
        return se.build_rebalancing_plan(
            analyses=analyses,
            available_cash_gbp=cash,
            new_opportunities=opps or [],
            gbp_rate=1.25,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Helpers, FX & currency conversion
# ---------------------------------------------------------------------------
class TestNumericHelpers(EngineTestCase):
    def test_num_preserves_genuine_zero(self):
        self.assertEqual(se._num(0.0, 5.0), 0.0)
        self.assertEqual(se._num(0, -18.0), 0.0)
        self.assertEqual(se._num("0", 1.0), 0.0)

    def test_num_defaults_for_missing_or_non_finite(self):
        self.assertEqual(se._num(None, 7.0), 7.0)
        self.assertEqual(se._num(float("nan"), 7.0), 7.0)
        self.assertEqual(se._num(float("inf"), 7.0), 7.0)
        self.assertEqual(se._num("abc", 7.0), 7.0)
        self.assertIsNone(se._num(None, None))

    def test_first_num_keeps_zero(self):
        self.assertEqual(se._first_num(None, 0.0, 3.0), 0.0)
        self.assertEqual(se._first_num(None, float("nan"), default=9.0), 9.0)

    def test_fractional_share_formatting(self):
        self.assertEqual(se._fmt_shares(1234.5), "1,234.5")
        self.assertEqual(se._fmt_shares(0.01234), "0.0123")
        self.assertEqual(se._fmt_shares(10.0), "10")
        self.assertEqual(se._fmt_shares(0.5), "0.5")

    def test_forward_cagr_integer_with_model_cap_flag(self):
        self.assertEqual(se._fmt_fwd_cagr(28.4, "cap"), "+28% (model cap)")
        self.assertEqual(se._fmt_fwd_cagr(12.2, None), "+12%")
        self.assertEqual(se._fmt_fwd_cagr(None), "n/a")


class TestToGbp(EngineTestCase):
    def test_gbp_pence_and_usd(self):
        self.assertAlmostEqual(se.to_gbp(100.0, "GBP", 1.35), 100.0)
        self.assertAlmostEqual(se.to_gbp(100.0, "GBp", 1.35), 1.0)
        self.assertAlmostEqual(se.to_gbp(250.0, "GBX", 1.35), 2.5)
        self.assertAlmostEqual(se.to_gbp(250.0, "PENCE", 1.35), 2.5)
        self.assertAlmostEqual(se.to_gbp(135.0, "USD", 1.35), 100.0)

    def test_eur_and_chf_use_live_fx(self):
        with mock.patch.object(se, "get_fx_to_gbp", return_value=0.85) as fx:
            self.assertAlmostEqual(se.to_gbp(100.0, "EUR", 1.35), 85.0)
            fx.assert_called_once_with("EUR")
        with mock.patch.object(se, "get_fx_to_gbp", return_value=0.9) as fx:
            self.assertAlmostEqual(se.to_gbp(10.0, "CHF", 1.35), 9.0)
            fx.assert_called_once_with("CHF")

    def test_unknown_or_unconvertible_currency_is_nan_never_raw_price(self):
        with mock.patch.object(se, "get_fx_to_gbp", side_effect=RuntimeError("no rate")):
            self.assertTrue(math.isnan(se.to_gbp(100.0, "XYZ", 1.35)))
        with mock.patch.object(se, "get_fx_to_gbp") as fx:
            for ccy in (None, "", "UNKNOWN", "N/A"):
                self.assertTrue(math.isnan(se.to_gbp(100.0, ccy, 1.35)), ccy)
            fx.assert_not_called()

    def test_non_positive_price_is_zero(self):
        self.assertEqual(se.to_gbp(0.0, "USD", 1.35), 0.0)
        self.assertEqual(se.to_gbp(None, "GBP", 1.35), 0.0)

    def test_usd_without_any_live_rate_is_nan(self):
        with mock.patch.object(se, "get_gbp_usd_rate", side_effect=RuntimeError("down")):
            self.assertTrue(math.isnan(se.to_gbp(100.0, "USD", None)))


class TestLiveFxRates(EngineTestCase):
    def test_gbp_usd_raises_instead_of_silent_default(self):
        with self.assertRaises(RuntimeError):
            se.get_gbp_usd_rate()
        self.assertIsNone(se._CACHED_GBP_RATE)

    def test_gbp_usd_live_value_is_cached(self):
        fake = FakeTicker(fast_info={"lastPrice": 1.27})
        factory = mock.Mock(side_effect=ticker_factory({"GBPUSD=X": fake}))
        with mock.patch.object(se.yf, "Ticker", factory):
            self.assertAlmostEqual(se.get_gbp_usd_rate(), 1.27)
            self.assertAlmostEqual(se.get_gbp_usd_rate(), 1.27)
        self.assertEqual(factory.call_count, 1)

    def test_fx_to_gbp_live_cached_and_fail_closed(self):
        factory = mock.Mock(side_effect=ticker_factory({"EURGBP=X": FakeTicker(fast_info={"lastPrice": 0.86})}))
        with mock.patch.object(se.yf, "Ticker", factory):
            self.assertAlmostEqual(se.get_fx_to_gbp("EUR"), 0.86)
            self.assertAlmostEqual(se.get_fx_to_gbp("eur"), 0.86)
            self.assertEqual(factory.call_count, 1)
            with self.assertRaises(RuntimeError):
                se.get_fx_to_gbp("SEK")
            with self.assertRaises(RuntimeError):
                se.get_fx_to_gbp("SEK")  # cached failure, still raises
        self.assertEqual(se.get_fx_to_gbp("GBP"), 1.0)
        self.assertEqual(se.get_fx_to_gbp("GBp"), 0.01)
        se._CACHED_GBP_RATE = 1.25
        self.assertAlmostEqual(se.get_fx_to_gbp("USD"), 0.8)

    # --- Fallback chain: Yahoo -> ECB (Frankfurter) -> ExchangeRate-API, never a static rate ---

    @staticmethod
    def fx_sources(frankfurter=None, er_api=None):
        """Fake _fetch_json routing by host; an Exception value is raised, anything else is returned."""
        def fake(url, timeout=10.0):
            payload = frankfurter if "frankfurter" in url else er_api if "er-api" in url else None
            if payload is None:
                raise RuntimeError(f"unexpected URL {url}")
            if isinstance(payload, Exception):
                raise payload
            return payload
        return mock.Mock(side_effect=fake)

    def test_gbp_usd_falls_back_to_ecb_reference_rate_and_caches_it(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fetch = self.fx_sources(frankfurter={"amount": 1.0, "base": "GBP", "date": today, "rates": {"USD": 1.3331}})
        with mock.patch.object(se, "_fetch_json", fetch):
            self.assertAlmostEqual(se.get_gbp_usd_rate(), 1.3331)
            self.assertAlmostEqual(se.get_gbp_usd_rate(), 1.3331)
        self.assertEqual(fetch.call_count, 1)
        self.assertIn("base=GBP&symbols=USD", fetch.call_args[0][0])

    def test_gbp_usd_uses_second_reference_source_when_ecb_fails(self):
        now_unix = int(datetime.now(timezone.utc).timestamp())
        fetch = self.fx_sources(
            frankfurter=RuntimeError("HTTP 503"),
            er_api={"result": "success", "base_code": "GBP", "time_last_update_unix": now_unix, "rates": {"USD": 1.334, "EUR": 1.18}},
        )
        with mock.patch.object(se, "_fetch_json", fetch):
            self.assertAlmostEqual(se.get_gbp_usd_rate(), 1.334)
        self.assertEqual(fetch.call_count, 2)

    def test_gbp_usd_rejects_stale_and_implausible_reference_rates_then_fails_closed(self):
        stale = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        fetch = self.fx_sources(
            frankfurter={"date": stale, "rates": {"USD": 1.30}},
            er_api={"time_last_update_unix": int(datetime.now(timezone.utc).timestamp()), "rates": {"USD": 13.3}},
        )
        with mock.patch.object(se, "_fetch_json", fetch):
            with self.assertRaises(RuntimeError):
                se.get_gbp_usd_rate()
        self.assertIsNone(se._CACHED_GBP_RATE)

    def test_live_yahoo_quote_never_touches_reference_sources(self):
        fetch = self.fx_sources(frankfurter={"date": "2000-01-01", "rates": {"USD": 9.9}})
        self.use_tickers({"GBPUSD=X": FakeTicker(fast_info={"lastPrice": 1.27})})
        with mock.patch.object(se, "_fetch_json", fetch):
            self.assertAlmostEqual(se.get_gbp_usd_rate(), 1.27)
        fetch.assert_not_called()

    def test_fx_to_gbp_falls_back_to_reference_rate(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fetch = self.fx_sources(frankfurter={"base": "EUR", "date": today, "rates": {"GBP": 0.84205}})
        with mock.patch.object(se, "_fetch_json", fetch):
            self.assertAlmostEqual(se.get_fx_to_gbp("EUR"), 0.84205)
            self.assertAlmostEqual(se.get_fx_to_gbp("EUR"), 0.84205)  # cached
        self.assertEqual(fetch.call_count, 1)
        self.assertIn("base=EUR&symbols=GBP", fetch.call_args[0][0])


# ---------------------------------------------------------------------------
# Weekly trend: GBP basis, fallback, 2022 clamp
# ---------------------------------------------------------------------------
class TestWeeklyTrendGbpBasis(EngineTestCase):
    def setUp(self):
        super().setUp()
        n = 260
        rng = np.random.RandomState(7)
        self.fx = 1.25 + 0.08 * np.sin(np.arange(n) / 6.0)               # GBPUSD (USD per GBP)
        self.spy_usd = 400.0 * np.cumprod(1.0 + rng.normal(0.002, 0.02, n))
        self.spy_gbp = self.spy_usd / self.fx

    def _mapping(self, asset_ticker, with_fx=True):
        mapping = {
            "SPY": FakeTicker(weekly=weekly_frame(self.spy_usd, tz="America/New_York")),
            "ASSET": asset_ticker,
        }
        if with_fx:
            mapping["GBPUSD=X"] = FakeTicker(weekly=weekly_frame(self.fx, tz="Europe/London"))
        return mapping

    def test_metrics_use_gbp_series_and_spy_in_gbp(self):
        # A GBP-quoted fund that is exactly "SPY converted into GBP" must show beta 1.0 / corr 1.0 vs SPY-in-GBP
        asset = FakeTicker(weekly=weekly_frame(self.spy_gbp, tz="Europe/London"), fast_info={"currency": "GBP"})
        self.use_tickers(self._mapping(asset))
        res = se.evaluate_weekly_trend(asset, yf_ticker="ASSET")
        self.assertTrue(res["data_ok"])
        self.assertEqual(res["fx_basis"], "GBP")
        self.assertAlmostEqual(res["beta_5y"], 1.0, places=2)
        self.assertAlmostEqual(res["corr_5y"], 1.0, places=2)

    def test_usd_asset_is_converted_before_cagr(self):
        # Flat in USD while GBP weakens -> positive CAGR in GBP
        n = 260
        fx_falling = np.linspace(1.40, 1.10, n)
        flat_usd = np.full(n, 100.0)
        asset = FakeTicker(weekly=weekly_frame(flat_usd), fast_info={"currency": "USD"})
        self.use_tickers({
            "SPY": FakeTicker(weekly=weekly_frame(self.spy_usd)),
            "GBPUSD=X": FakeTicker(weekly=weekly_frame(fx_falling)),
        })
        res = se.evaluate_weekly_trend(asset, yf_ticker="FLAT")
        self.assertEqual(res["fx_basis"], "GBP")
        self.assertGreater(res["cagr_5y_pct"], 3.0)

    def test_native_fallback_when_fx_history_unavailable(self):
        asset = FakeTicker(weekly=weekly_frame(self.spy_gbp), fast_info={"currency": "GBP"})
        self.use_tickers(self._mapping(asset, with_fx=False))
        res = se.evaluate_weekly_trend(asset, yf_ticker="ASSET")
        self.assertTrue(res["data_ok"])
        self.assertEqual(res["fx_basis"], "native")
        self.assertLess(res["corr_5y"], 0.999)  # measured against USD SPY: no longer a perfect match

    def test_weekly_fallback_is_fail_closed(self):
        res = se.evaluate_weekly_trend(FakeTicker(fail=True), yf_ticker="BROKEN")
        self.assertFalse(res["is_weekly_stage_2"])
        self.assertEqual(res["label"], "⚪ Weekly data unavailable")
        self.assertFalse(res["data_ok"])

    def test_2022_beta_estimate_is_clamped_to_plus_2pct(self):
        n = 260
        spy = np.concatenate([np.full(13, 400.0), np.linspace(400.0, 300.0, 52), np.linspace(300.0, 420.0, n - 65)])
        fx = np.full(n, 1.25)
        start_idx = 70  # asset listed in 2023 (no 2022 bars)
        spy_gbp = spy / fx
        inverse = 1000.0 / spy_gbp[start_idx:]  # strongly negative beta
        asset = FakeTicker(weekly=weekly_frame(inverse, start=str(weekly_frame(spy).index[start_idx].date())), fast_info={"currency": "GBP"})
        self.use_tickers({"SPY": FakeTicker(weekly=weekly_frame(spy)), "GBPUSD=X": FakeTicker(weekly=weekly_frame(fx))})
        res = se.evaluate_weekly_trend(asset, yf_ticker="NEWFUND")
        self.assertLess(res["beta_5y"], 0.0)
        self.assertLessEqual(res["crash_2022_ret_pct"], 2.0)


# ---------------------------------------------------------------------------
# Instrument classification
# ---------------------------------------------------------------------------
class TestClassification(EngineTestCase):
    def test_sector_etfs_are_not_core(self):
        self.assertFalse(se.is_core_asset("SMGB.L", "VanEck Semiconductor UCITS ETF", "AI Hardware & Semiconductors"))
        self.assertFalse(se.is_core_asset("DFND.SW", "iShares Global Aerospace & Defence UCITS ETF", "Aerospace & Defense"))

    def test_broad_index_gold_and_mmf_stay_core(self):
        for t in ("VWRP.L", "VUAA.L", "CNDX.L", "SGLN.L", "CSH2.L", "VUAG.L"):
            self.assertTrue(se.is_core_asset(t), t)
        self.assertFalse(se.is_core_asset("QQQ3.L", "WisdomTree NASDAQ 100 3x Daily Leveraged"))

    def test_leveraged_detection(self):
        for t in ("QQQ3.L", "3LUS.L", "LQQ3.L", "3USL.L", "TQQQ"):
            self.assertTrue(se.is_leveraged_etf(t), t)
        self.assertFalse(se.is_leveraged_etf("QQQ"))
        self.assertTrue(se.is_broad_index_etf("QQQ"))
        self.assertFalse(se.is_broad_index_etf("QQQ3.L"))


# ---------------------------------------------------------------------------
# Risk profile & consolidated view
# ---------------------------------------------------------------------------
class TestRiskProfile(EngineTestCase):
    METRIC_KEYS = ("cagr_5y_pct", "sortino_5y", "beta_5y", "stress_2022_pct", "stress_2022_gbp",
                   "shield_gbp", "shield_pct", "lev_3x_pct", "defensive_gbp", "defensive_pct")

    def assert_post_equals_current(self, rp):
        for k in self.METRIC_KEYS:
            self.assertEqual(rp[f"current_{k}"], rp[f"post_{k}"], k)
        self.assertEqual(rp["crash_delta_gbp"], 0.0)
        self.assertEqual(rp["crash_capital_saved_gbp"], 0.0)

    def _accounts(self):
        return {
            "Main": {
                "cash": {"free_gbp": 0.0, "blocked_gbp": 0.0, "total": 80000.0},
                "analyses": [
                    make_analysis("VWRP.L", 50000.0, macro_cluster="Diversified Index Core", crash_2022_ret_pct=-6.3, beta_5y=0.72),
                    make_analysis("NVDA", 20000.0, macro_cluster="AI Hardware & Semiconductors", crash_2022_ret_pct=-39.7, beta_5y=2.06),
                    make_analysis("CSH2.L", 10000.0, macro_cluster="Uncorrelated Crash Hedge & Gold", crash_2022_ret_pct=0.0, beta_5y=0.0, corr_5y=0.0),
                ],
            },
            "Secondary": {
                "cash": {"free_gbp": 5000.0, "blocked_gbp": 0.0, "total": 45000.0},
                "analyses": [
                    make_analysis("VWRP.L", 30000.0, macro_cluster="Diversified Index Core", crash_2022_ret_pct=-6.3, beta_5y=0.72),
                    make_analysis("NVDA", 10000.0, macro_cluster="AI Hardware & Semiconductors", crash_2022_ret_pct=-39.7, beta_5y=2.06),
                ],
            },
        }

    def test_no_orders_post_equals_current_single_account(self):
        book = [make_analysis("MSFT", 60000.0), make_analysis("SGLN.L", 10000.0, macro_cluster="Uncorrelated Crash Hedge & Gold")]
        rp = se.compute_portfolio_5y_risk_profile(book, 30000.0, 0.0, 100000.0, sell_orders=[], buy_orders=[])
        self.assert_post_equals_current(rp)

    def test_no_orders_post_equals_current_with_ticker_in_two_accounts(self):
        view = se.build_consolidated_portfolio_view(
            self._accounts(), gbp_rate=1.25,
            account_plans={"Main": {"sell_orders": [], "buy_orders": []}, "Secondary": {"sell_orders": [], "buy_orders": []}},
        )
        self.assert_post_equals_current(view["risk_profile_5y"])
        self.assertEqual(view["holdings"]["NVDA"]["total_value_gbp"], 30000.0)

    def test_consolidated_sell_only_touches_the_selling_accounts_copy(self):
        accounts = self._accounts()
        sell = {"ticker": "NVDA", "estimated_gbp": 10000.0, "is_pending": False}   # no account tag: gets stamped
        view = se.build_consolidated_portfolio_view(
            accounts, gbp_rate=1.25, account_plans={"Secondary": {"sell_orders": [sell], "buy_orders": []}},
        )
        rp = view["risk_profile_5y"]
        self.assertEqual(sell["account"], "Secondary")
        # Free cash rises by exactly Secondary's £10k NVDA; Main's £20k NVDA copy stays in the book
        self.assertEqual(rp["post_defensive_gbp"] - rp["current_defensive_gbp"], 10000.0)
        self.assertGreater(rp["crash_delta_gbp"], 0.0)
        self.assertEqual(rp["crash_capital_saved_gbp"], rp["crash_delta_gbp"])

    def test_crash_delta_is_signed(self):
        book = [make_analysis("VWRP.L", 50000.0, crash_2022_ret_pct=-5.0, macro_cluster="Diversified Index Core")]
        sells = [{"ticker": "VWRP.L", "estimated_gbp": 20000.0}]
        buys = [{"ticker": "NVDA", "estimated_gbp": 20000.0, "crash_2022_ret_pct": -40.0, "beta_5y": 2.0}]
        rp = se.compute_portfolio_5y_risk_profile(book, 0.0, 0.0, 50000.0, sell_orders=sells, buy_orders=buys)
        self.assertLess(rp["crash_delta_gbp"], 0.0)
        self.assertEqual(rp["crash_capital_saved_gbp"], 0.0)

    def test_new_buy_missing_metrics_uses_conservative_defaults(self):
        rp = se.compute_portfolio_5y_risk_profile([], 10000.0, 0.0, 10000.0, sell_orders=[], buy_orders=[{"ticker": "NEWCO", "estimated_gbp": 5000.0}])
        self.assertAlmostEqual(rp["post_stress_2022_pct"], round(0.5 * 1.8 + 0.5 * -25.0, 1))
        self.assertAlmostEqual(rp["post_beta_5y"], 0.6)

    def test_real_zero_beta_and_crash_are_preserved(self):
        book = [make_analysis("CSH2.L", 10000.0, beta_5y=0.0, crash_2022_ret_pct=0.0, macro_cluster="Uncorrelated Crash Hedge & Gold")]
        rp = se.compute_portfolio_5y_risk_profile(book, 0.0, 0.0, 10000.0)
        self.assertEqual(rp["current_beta_5y"], 0.0)
        self.assertEqual(rp["current_stress_2022_pct"], 0.0)

    def test_defensive_metric_excludes_earmarked(self):
        book = [
            make_analysis("CSH2.L", 20000.0, macro_cluster="Uncorrelated Crash Hedge & Gold"),
            make_analysis("SGLN.L", 5000.0, macro_cluster="Uncorrelated Crash Hedge & Gold"),
            make_analysis("VWRP.L", 75000.0, macro_cluster="Diversified Index Core"),
        ]
        rp_plain = se.compute_portfolio_5y_risk_profile(book, 0.0, 0.0, 100000.0)
        self.assertEqual(rp_plain["current_defensive_gbp"], 25000.0)
        self.assertEqual(rp_plain["current_defensive_pct"], 25.0)
        with mock.patch.object(config, "EARMARKED_TICKERS", {"CSH2.L"}):
            rp = se.compute_portfolio_5y_risk_profile(book, 0.0, 0.0, 100000.0)
        self.assertEqual(rp["current_defensive_gbp"], 5000.0)
        self.assertEqual(rp["current_defensive_pct"], 6.2)  # 5,000 / 80,000 = 6.25%
        self.assertEqual(rp["earmarked_gbp"], 20000.0)
        self.assertEqual(rp["investable_total_gbp"], 80000.0)

    def test_blocked_cash_is_not_shield(self):
        rp = se.compute_portfolio_5y_risk_profile([make_analysis("MSFT", 85000.0)], 10000.0, 5000.0, 100000.0)
        self.assertEqual(rp["current_shield_gbp"], 10000.0)
        self.assertEqual(rp["current_defensive_gbp"], 10000.0)

    def test_shield_status_uses_config_target_and_leverage_cap(self):
        def status(shield_val, lev_val):
            book = [make_analysis("MSFT", 100000.0 - shield_val - lev_val)]
            if shield_val:
                book.append(make_analysis("SGLN.L", shield_val, macro_cluster="Uncorrelated Crash Hedge & Gold"))
            if lev_val:
                book.append(make_analysis("QQQ3.L", lev_val, macro_cluster="High-Beta Technology"))
            return se.compute_portfolio_5y_risk_profile(book, 0.0, 0.0, 100000.0)["shield_status"]

        self.assertIn("ACTIVE", status(20000.0, 0.0))
        self.assertIn("BUILDING", status(10000.0, 0.0))
        self.assertIn("HIGH BETA", status(5000.0, 0.0))
        self.assertIn("BUILDING", status(20000.0, 10000.0))  # 10% leverage breaches the 8% cap

    def test_projections_bear_base_bull_and_aliases(self):
        rp = se.compute_portfolio_5y_risk_profile([], 10000.0, 0.0, 10000.0)
        expected = round(10000.0 * (1.035 ** 5), 0)
        for scen in ("bear", "base", "bull"):
            self.assertEqual(rp[f"projected_5y_{scen}_gbp"], expected)
            self.assertEqual(rp[f"projected_5y_{scen}_cagr_pct"], 3.5)
        book = [make_analysis("MSFT", 50000.0), make_analysis("SGLN.L", 25000.0, macro_cluster="Uncorrelated Crash Hedge & Gold")]
        rp = se.compute_portfolio_5y_risk_profile(book, 25000.0, 0.0, 100000.0)
        base_rate = 0.5 * 0.06 + 0.25 * 0.03 + 0.25 * 0.035
        self.assertEqual(rp["projected_5y_base_gbp"], round(100000.0 * (1 + base_rate) ** 5, 0))
        self.assertEqual(rp["projected_5y_conservative_gbp"], rp["projected_5y_base_gbp"])
        self.assertEqual(rp["projected_5y_conservative_cagr_pct"], rp["projected_5y_base_cagr_pct"])
        self.assertEqual(rp["projected_5y_target_gbp"], rp["projected_5y_bull_gbp"])
        self.assertEqual(rp["projected_5y_target_cagr_pct"], rp["projected_5y_bull_cagr_pct"])
        self.assertLess(rp["projected_5y_bear_gbp"], rp["projected_5y_base_gbp"])


# ---------------------------------------------------------------------------
# Rebalancing plan: limits, reserve, data quality, earmarks, outside holdings, pending orders
# ---------------------------------------------------------------------------
class TestRebalancePlanLimits(EngineTestCase):
    def test_leverage_cap_trim_sizing_ignores_lockouts(self):
        qqq3 = make_analysis("QQQ3.L", 12000.0, price_gbp=100.0, macro_cluster="High-Beta Technology",
                             is_recently_trimmed=True, days_held=3)
        vwrp = make_analysis("VWRP.L", 88000.0, macro_cluster="Diversified Index Core")
        plan = self.plan([qqq3, vwrp], cash=0.0, official_total_gbp=100000.0)
        trims = [s for s in plan["sell_orders"] if s["ticker"] == "QQQ3.L"]
        self.assertEqual(len(trims), 1)
        expected = (0.12 - 0.08) * 100000.0 / 100.0
        self.assertAlmostEqual(trims[0]["shares"], expected, places=3)
        self.assertEqual(trims[0]["order_reason"], "leverage_cap")
        self.assertIn("TRIM", trims[0]["action"])
        self.assertNotIn("80.7", trims[0]["plain_english_why"])
        self.assertNotIn("76.2", trims[0]["plain_english_why"])

    def test_no_rsi_trim_for_leverage_below_cap(self):
        qqq3 = make_analysis("QQQ3.L", 5000.0, macro_cluster="High-Beta Technology", action_short="TRIM",
                             portfolio_weight_pct=5.0)
        plan = self.plan([qqq3, make_analysis("VWRP.L", 95000.0, macro_cluster="Diversified Index Core")], official_total_gbp=100000.0)
        self.assertEqual(plan["sell_orders"], [])

    def test_zero_leverage_policy_sells_whole_position_despite_trim_lockout(self):
        se.REBALANCE_CONFIG["max_leveraged_etf_pct"] = 0.0
        qqq3 = make_analysis("QQQ3.L", 5000.0, price_gbp=50.0, macro_cluster="High-Beta Technology",
                             is_recently_trimmed=True, days_held=6)
        plan = self.plan([qqq3, make_analysis("VWRP.L", 95000.0, macro_cluster="Diversified Index Core")], official_total_gbp=100000.0)
        self.assertEqual(len(plan["sell_orders"]), 1)
        order = plan["sell_orders"][0]
        self.assertEqual(order["action"], "🔴 SELL (LEVERAGE POLICY — 0% CAP)")
        self.assertAlmostEqual(order["shares"], qqq3["quantity"])
        self.assertEqual(qqq3["signal"], "🔴 SELL (LEVERAGE POLICY — 0% CAP)")
        self.assertEqual(qqq3["action_short"], "SELL")
        self.assertEqual(plan["risk_profile_5y"]["post_lev_3x_pct"], 0.0)

    def test_position_cap_trims_single_stock_regardless_of_signal_and_lockouts(self):
        nvda = make_analysis("NVDA", 20000.0, price_gbp=100.0, macro_cluster="AI Hardware & Semiconductors",
                             action_short="BUY", is_recently_trimmed=True, days_held=5)
        plan = self.plan([nvda, make_analysis("VWRP.L", 80000.0, macro_cluster="Diversified Index Core")], official_total_gbp=100000.0)
        trims = [s for s in plan["sell_orders"] if s["ticker"] == "NVDA"]
        self.assertEqual(len(trims), 1)
        self.assertAlmostEqual(trims[0]["shares"], (0.20 - 0.15) * 100000.0 / 100.0, places=3)
        self.assertEqual(trims[0]["order_reason"], "position_cap")
        self.assertNotIn("3x", nvda["plan_summary"])

    def test_broad_etfs_exempt_unless_configured(self):
        def run():
            vwrp = make_analysis("VWRP.L", 30000.0, price_gbp=100.0, macro_cluster="Diversified Index Core")
            return self.plan([vwrp, make_analysis("MSFT", 10000.0), make_analysis("AAPL", 10000.0, macro_cluster="Consumer & Digital Platforms")],
                             cash=0.0, official_total_gbp=50000.0)

        self.assertEqual([s for s in run()["sell_orders"] if s["ticker"] == "VWRP.L"], [])
        se.REBALANCE_CONFIG["cap_applies_to_broad_etfs"] = True
        trims = [s for s in run()["sell_orders"] if s["ticker"] == "VWRP.L"]
        self.assertEqual(len(trims), 1)
        self.assertAlmostEqual(trims[0]["shares"], (0.60 - 0.15) * 50000.0 / 100.0, places=3)

    def test_max_new_ideas_enforced_and_remaining_slot_goes_to_add(self):
        clusters = ["Financials & Payment Rails", "Healthcare & Biopharma", "Infrastructure & Electrification", "Aerospace & Defense"]
        opps = [
            make_analysis(t, 0.0, price_gbp=50.0, macro_cluster=c, is_new_addition=True, conviction_score=90, quantity=0.0)
            for t, c in zip(["V", "ABT", "ETN", "BA.L"], clusters)
        ]
        add = make_analysis("MSFT", 5000.0, conviction_score=85, macro_cluster="Enterprise Cloud & Software")
        core = make_analysis("VWRP.L", 75000.0, macro_cluster="Diversified Index Core")
        plan = self.plan([add, core], cash=20000.0, opps=opps)
        buys = plan["buy_orders"]
        self.assertLessEqual(sum(1 for b in buys if b["is_new"]), 2)
        self.assertTrue(any(not b["is_new"] for b in buys), "remaining slot should go to an add")

    def test_buy_sizing_respects_post_trade_position_cap(self):
        opp = make_analysis("V", 0.0, price_gbp=10.0, macro_cluster="Financials & Payment Rails", is_new_addition=True, conviction_score=95, quantity=0.0)
        plan = self.plan([make_analysis("VWRP.L", 4000.0, macro_cluster="Diversified Index Core")], cash=10000.0, opps=[opp], official_total_gbp=14000.0)
        for b in plan["buy_orders"]:
            if b["ticker"] == "V":
                self.assertLessEqual(b["estimated_gbp"], 0.15 * 14000.0 + 1e-6)

    def test_cluster_cap_uses_config(self):
        se.REBALANCE_CONFIG["max_cluster_pct"] = 0.50
        opp = make_analysis("ORCL", 0.0, price_gbp=50.0, is_new_addition=True, conviction_score=95, quantity=0.0)
        book = [make_analysis("MSFT", 3000.0), make_analysis("CRM", 3000.0), make_analysis("VWRP.L", 4000.0, macro_cluster="Diversified Index Core")]
        plan = self.plan(book, cash=10000.0, opps=[opp], official_total_gbp=20000.0)
        self.assertIn("ORCL", [b["ticker"] for b in plan["buy_orders"]])  # 30% < 50% configured cap
        se.REBALANCE_CONFIG["max_cluster_pct"] = 0.24
        book = [make_analysis("MSFT", 3000.0), make_analysis("CRM", 3000.0), make_analysis("VWRP.L", 4000.0, macro_cluster="Diversified Index Core")]
        plan = self.plan(book, cash=10000.0, opps=[opp], official_total_gbp=20000.0)
        self.assertNotIn("ORCL", [b["ticker"] for b in plan["buy_orders"]])


class TestRebalancePlanReserveAndData(EngineTestCase):
    def _opp(self, **kw):
        base = dict(price_gbp=50.0, macro_cluster="Financials & Payment Rails", is_new_addition=True, conviction_score=95, quantity=0.0)
        base.update(kw)
        return make_analysis("V", 0.0, **base)

    def test_reserve_invariant_no_buys_below_floor_plus_1000(self):
        core = make_analysis("VWRP.L", 50000.0, macro_cluster="Diversified Index Core")
        for cash in (0.0, 1000.0, 2400.0):
            plan = self.plan([dict(core)], cash=cash, opps=[self._opp()])
            self.assertEqual(plan["buy_orders"], [], cash)
            self.assertEqual(plan["net_budget_to_deploy_gbp"], 0.0)
        plan = self.plan([dict(core)], cash=2600.0, opps=[self._opp()])
        self.assertTrue(plan["buy_orders"], "positive control: cash above floor + £1,000 can deploy")

    def test_reserve_fields(self):
        plan = self.plan([make_analysis("VWRP.L", 50000.0, macro_cluster="Diversified Index Core")], cash=1000.0)
        self.assertEqual(plan["reserve_floor_gbp"], 1500.0)
        self.assertEqual(plan["reserve_funded_gbp"], 1000.0)
        self.assertEqual(plan["reserve_shortfall_gbp"], 500.0)
        plan = self.plan([make_analysis("VWRP.L", 50000.0, macro_cluster="Diversified Index Core")], cash=0.0)
        self.assertEqual(plan["reserve_funded_gbp"], 0.0)
        self.assertEqual(plan["reserve_shortfall_gbp"], 1500.0)
        tldr = se.generate_tldr_summary({"Main": {"rebalancing_plan": plan}})["Main"]
        self.assertEqual(tldr["phase"], "ALL_SET")
        self.assertIn("unfunded", tldr["phase_desc"].lower())
        self.assertNotIn("safe to withdraw", tldr["phase_desc"].lower())

    def test_data_ok_false_candidate_never_becomes_buy(self):
        core = make_analysis("VWRP.L", 50000.0, macro_cluster="Diversified Index Core")
        bad_new = self._opp(data_ok=False, action_short="BUY")
        bad_add = make_analysis("MSFT", 2000.0, data_ok=False, action_short="BUY", conviction_score=95)
        plan = self.plan([core, bad_add], cash=20000.0, opps=[bad_new])
        self.assertEqual([b["ticker"] for b in plan["buy_orders"] if b["ticker"] in ("V", "MSFT")], [])
        good = self.plan([dict(core)], cash=20000.0, opps=[self._opp()])
        self.assertIn("V", [b["ticker"] for b in good["buy_orders"]])

    def test_scan_watchlist_requires_data_ok_and_passed_gates(self):
        watch = [{"ticker": "MSFT", "name": "Microsoft", "theme": "Enterprise Cloud"}]
        base = make_analysis("MSFT", 0.0, quantity=0.0, conviction_score=95, action_short="HOLD", days_to_earnings=None)
        cases = [
            (dict(base, data_ok=False, action_short="BUY"), False),
            (dict(base, risk_reward_ratio=1.5), False),
            (dict(base, stop_breached=True), False),
            (dict(base, signal="🟡 HOLD (WAIT FOR ASYMMETRY)"), False),
            (dict(base, risk_reward_ratio=None), True),
        ]
        for analysis, expected in cases:
            with mock.patch.object(se, "WATCHLIST", watch), mock.patch.object(se, "analyze_position", return_value=dict(analysis)):
                opps = se.scan_watchlist_opportunities(set(), gbp_rate=1.25)
            self.assertEqual(bool(opps), expected, analysis)

    def test_scan_collector_receives_every_non_excluded_analysis(self):
        watch = [
            {"ticker": "MSFT", "name": "Microsoft", "theme": "Cloud"},
            {"ticker": "V", "name": "Visa", "theme": "Payments"},
            {"ticker": "AAPL", "name": "Apple", "theme": "Consumer"},
            {"ticker": "QQQ3.L", "name": "3x Nasdaq", "theme": "Leverage"},
            {"ticker": "COIN", "name": "Coinbase", "theme": "Crypto Exchange"},
        ]
        by_ticker = {
            "MSFT": make_analysis("MSFT", 0.0, quantity=0.0, conviction_score=95, days_to_earnings=None, research={"altman_z": 9.1, "piotroski_score": 8}),
            "V": make_analysis("V", 0.0, quantity=0.0, data_ok=False, days_to_earnings=None, research={}),
            "AAPL": make_analysis("AAPL", 0.0, quantity=0.0, conviction_score=95, days_to_earnings=None),
            "QQQ3.L": make_analysis("QQQ3.L", 0.0, quantity=0.0, conviction_score=95, days_to_earnings=None),
        }
        fake_analyze = mock.Mock(side_effect=lambda t, **kw: dict(by_ticker[t]))
        collector = []
        with mock.patch.object(se, "WATCHLIST", watch), mock.patch.object(se, "analyze_position", fake_analyze):
            opps = se.scan_watchlist_opportunities({"AAPL"}, gbp_rate=1.25, scan_collector=collector)
        with mock.patch.object(se, "WATCHLIST", watch), mock.patch.object(se, "analyze_position", mock.Mock(side_effect=lambda t, **kw: dict(by_ticker[t]))):
            opps_plain = se.scan_watchlist_opportunities({"AAPL"}, gbp_rate=1.25)
        self.assertEqual([o["ticker"] for o in opps], ["MSFT"])
        self.assertEqual([o["ticker"] for o in opps], [o["ticker"] for o in opps_plain])
        self.assertEqual(sorted(c["ticker"] for c in collector), ["AAPL", "MSFT", "QQQ3.L", "V"])  # only COIN (EXCLUDED_TICKERS) skipped
        for c in collector:
            for key in ("ticker", "data_ok", "is_weekly_stage_2", "weekly_trend_label", "risk_reward_ratio",
                        "piotroski_score", "altman_z", "research", "scan_outcome", "scan_passed"):
                self.assertIn(key, c, (c["ticker"], key))
        outcomes = {c["ticker"]: c for c in collector}
        self.assertTrue(outcomes["MSFT"]["scan_passed"])
        self.assertEqual(outcomes["MSFT"]["altman_z"], 9.1)  # back-filled from the research dict
        self.assertFalse(outcomes["V"]["scan_passed"])
        self.assertIn("data incomplete", outcomes["V"]["scan_outcome"])
        self.assertIn("owned", outcomes["AAPL"]["scan_outcome"])
        self.assertIn("leveraged", outcomes["QQQ3.L"]["scan_outcome"])

    def test_invalid_valuation_degrades_plan_and_emits_no_orders(self):
        broken = make_analysis("SGLN.L", 0.0, price_gbp=0.0, quantity=10.0, valuation_ok=False, macro_cluster="Uncorrelated Crash Hedge & Gold")
        seller = make_analysis("NVDA", 30000.0, action_short="SELL", macro_cluster="AI Hardware & Semiconductors")
        plan = self.plan([broken, seller], cash=50000.0, opps=[self._opp()])
        self.assertTrue(plan["data_degraded_reason"])
        self.assertIn("SGLN.L", plan["data_degraded_reason"])
        self.assertEqual(plan["sell_orders"], [])
        self.assertEqual(plan["buy_orders"], [])
        tldr = se.generate_tldr_summary({"Secondary": {"rebalancing_plan": plan}})["Secondary"]
        self.assertEqual(tldr["phase"], "DATA_DEGRADED")
        self.assertEqual(tldr["buys_to_place"], [])
        self.assertEqual(tldr["sells_to_place"], [])

    def test_clean_plan_has_no_degraded_reason(self):
        plan = self.plan([make_analysis("VWRP.L", 50000.0, macro_cluster="Diversified Index Core")], cash=0.0)
        self.assertIsNone(plan["data_degraded_reason"])


class TestEarmarkedFunds(EngineTestCase):
    def test_earmarked_excluded_from_weights_and_never_traded(self):
        with mock.patch.object(config, "EARMARKED_TICKERS", {"CSH2.L"}):
            csh2 = make_analysis("CSH2.L", 20000.0, macro_cluster="Uncorrelated Crash Hedge & Gold", action_short="SELL")
            nvda = make_analysis("NVDA", 13600.0, price_gbp=100.0, macro_cluster="AI Hardware & Semiconductors")
            vwrp = make_analysis("VWRP.L", 66400.0, macro_cluster="Diversified Index Core")
            plan = self.plan([csh2, nvda, vwrp], cash=0.0, official_total_gbp=100000.0)
        self.assertEqual(plan["earmarked_gbp"], 20000.0)
        self.assertEqual(plan["earmarked_tickers"], ["CSH2.L"])
        self.assertFalse([s for s in plan["sell_orders"] if s["ticker"] == "CSH2.L"])
        # NVDA is 13.6% of the gross book but 17% of the investable book -> trimmed back to 15% of £80k
        trims = [s for s in plan["sell_orders"] if s["ticker"] == "NVDA"]
        self.assertEqual(len(trims), 1)
        self.assertAlmostEqual(trims[0]["shares"], (13600.0 - 0.15 * 80000.0) / 100.0, places=3)
        self.assertEqual(nvda["portfolio_weight_pct"], 17.0)
        self.assertEqual(plan["risk_profile_5y"]["investable_total_gbp"], 80000.0)

    def test_analyze_position_labels_earmarked(self):
        with mock.patch.object(config, "EARMARKED_TICKERS", {"CSH2.L"}), \
                mock.patch.object(se.research_agent, "conduct_deep_dive", return_value={"data_ok": True, "company_name": "Amundi SONIA"}):
            res = se.analyze_position("CSH2.L", {"quantity": 100.0, "currentPrice": 120.0, "currency": "GBP"}, gbp_rate=1.25)
        self.assertEqual(res["signal"], "🔒 HOLD (EARMARKED — EXCLUDED FROM PORTFOLIO MATHS)")
        self.assertEqual(res["action_short"], "HOLD")


class TestOutsideHoldings(EngineTestCase):
    OUTSIDE = {"Main": [{
        "ticker": "ACME", "shares_vested": 200.0, "shares_unvested": 50.0, "currency": "USD",
        "macro_cluster": "Consumer & Digital Platforms",
        "value_hint_usd_vested": 30000.0, "value_hint_usd_unvested": 7500.0,
    }]}

    def setUp(self):
        super().setUp()
        p = mock.patch.object(config, "OUTSIDE_HOLDINGS", self.OUTSIDE)
        p.start()
        self._patches.append(p)
        se._CACHED_GBP_RATE = 1.25

    def test_vested_valued_at_live_price_in_gbp(self):
        self.use_tickers({"ACME": FakeTicker(fast_info={"lastPrice": 200.0, "currency": "USD"})})
        rows = se.value_outside_holdings("Main")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vested_gbp"], 32000.0)    # 200 sh x $200 / 1.25
        self.assertEqual(rows[0]["unvested_gbp"], 8000.0)
        self.assertEqual(rows[0]["valuation_source"], "live")

    def test_falls_back_to_usd_hint_when_price_unavailable(self):
        rows = se.value_outside_holdings("Main")  # yf.Ticker raises in this test
        self.assertEqual(rows[0]["vested_gbp"], 24000.0)    # $30,000 / 1.25
        self.assertEqual(rows[0]["valuation_source"], "hint")

    def _main_accounts(self):
        return {"Main": {
            "cash": {"free_gbp": 20000.0, "blocked_gbp": 0.0, "total": 120000.0},
            "analyses": [
                make_analysis("AMZN", 10000.0, macro_cluster="Consumer & Digital Platforms", conviction_score=90, rsi=40.0),
                make_analysis("VWRP.L", 90000.0, macro_cluster="Diversified Index Core"),
            ],
        }}

    def test_consolidated_view_reports_outside_holdings(self):
        self.use_tickers({"ACME": FakeTicker(fast_info={"lastPrice": 200.0, "currency": "USD"})})
        view = se.build_consolidated_portfolio_view(self._main_accounts(), gbp_rate=1.25)
        self.assertEqual(view["outside_vested_gbp_total"], 32000.0)
        self.assertEqual(view["outside_unvested_gbp_total"], 8000.0)
        row = view["outside_holdings"][0]
        for key in ("account", "ticker", "vested_gbp", "unvested_gbp", "macro_cluster"):
            self.assertIn(key, row)
        self.assertEqual(row["account"], "Main")
        weights = {tb["cluster"]: tb["weight_pct"] for tb in view["thematic_breakdown"]}
        # (10,000 AMZN + 32,000 vested outside shares) / (120,000 + 32,000)
        self.assertAlmostEqual(weights["Consumer & Digital Platforms"], round(42000.0 / 152000.0 * 100.0, 1))

    def test_outside_vested_value_counts_towards_cluster_cap(self):
        self.use_tickers({"ACME": FakeTicker(fast_info={"lastPrice": 200.0, "currency": "USD"})})
        accounts = self._main_accounts()
        view = se.build_consolidated_portfolio_view(accounts, gbp_rate=1.25)
        meta = make_analysis("META", 0.0, price_gbp=50.0, macro_cluster="Consumer & Digital Platforms",
                             is_new_addition=True, conviction_score=95, quantity=0.0)
        coin = make_analysis("COIN", 0.0, price_gbp=50.0, macro_cluster="Consumer & Digital Platforms",
                             is_new_addition=True, conviction_score=99, quantity=0.0)
        plan = se.build_rebalancing_plan(
            analyses=accounts["Main"]["analyses"], available_cash_gbp=20000.0, new_opportunities=[meta, coin],
            gbp_rate=1.25, min_cash_reserve_gbp=1500.0, global_context=view, official_total_gbp=120000.0,
        )
        bought = [b["ticker"] for b in plan["buy_orders"]]
        self.assertEqual(plan["account_name"], "Main")
        self.assertEqual(plan["outside_vested_gbp"], 32000.0)
        self.assertNotIn("META", bought)   # cluster saturated once the outside holding is counted
        self.assertNotIn("AMZN", bought)
        self.assertNotIn("COIN", bought)   # EXCLUDED_TICKERS still blocks COIN
        self.assertFalse([s for s in plan["sell_orders"] if s["ticker"] == "ACME"])  # outside holdings are never traded

    def test_without_outside_holdings_same_cluster_is_buyable(self):
        with mock.patch.object(config, "OUTSIDE_HOLDINGS", {}):
            accounts = self._main_accounts()
            view = se.build_consolidated_portfolio_view(accounts, gbp_rate=1.25)
            meta = make_analysis("META", 0.0, price_gbp=50.0, macro_cluster="Consumer & Digital Platforms",
                                 is_new_addition=True, conviction_score=95, quantity=0.0)
            plan = se.build_rebalancing_plan(
                analyses=accounts["Main"]["analyses"], available_cash_gbp=20000.0, new_opportunities=[meta],
                gbp_rate=1.25, min_cash_reserve_gbp=1500.0, global_context=view, official_total_gbp=120000.0,
            )
        self.assertIn("META", [b["ticker"] for b in plan["buy_orders"]])


class TestPendingOrdersAndCash(EngineTestCase):
    def test_stop_orders_are_not_immediate_and_partial_orders_valued_by_qty(self):
        nvda = make_analysis("NVDA", 10000.0, price_gbp=100.0, macro_cluster="AI Hardware & Semiconductors")
        msft = make_analysis("MSFT", 10000.0, price_gbp=100.0, price=100.0)
        pending = {"pending_sells": {
            "NVDA": {"order_type": "STOP", "quantity": 10.0, "limit_price": 0.0, "stop_price": 90.0},
            "MSFT": {"order_type": "LIMIT", "quantity": 25.0, "limit_price": 100.5},
        }, "pending_buys": {}}
        plan = self.plan([nvda, msft, make_analysis("VWRP.L", 80000.0, macro_cluster="Diversified Index Core")],
                         cash=0.0, pending_orders_data=pending)
        by_t = {p["ticker"]: p for p in plan["pending_sell_orders"]}
        self.assertFalse(by_t["NVDA"]["is_immediate_fill"])
        self.assertEqual(by_t["NVDA"]["estimated_gbp"], 1000.0)
        self.assertTrue(by_t["MSFT"]["is_immediate_fill"])
        self.assertEqual(by_t["MSFT"]["estimated_gbp"], 2500.0)
        self.assertEqual(plan["cash_from_sells_gbp"], 2500.0)
        self.assertEqual(plan["sell_orders"], [])

    def test_final_cash_does_not_double_subtract_blocked_cash(self):
        plan = self.plan([make_analysis("VWRP.L", 50000.0, macro_cluster="Diversified Index Core")], cash=10000.0, blocked_cash_gbp=3000.0)
        self.assertEqual(plan["final_cash_balance_gbp"], 10000.0 - sum(b["estimated_gbp"] for b in plan["buy_orders"]))

    def test_tldr_fractional_shares_and_fully_balanced_wording(self):
        plan = {"sell_orders": [{"ticker": "NVDA", "name": "NVIDIA", "search_symbol": "NVDA", "action": "TRIM (POSITION CAP — 15%)",
                                 "shares": 0.5, "estimated_gbp": 75.0, "order_reason": "position_cap", "is_profit": False}],
                "available_cash_gbp": 0.0, "net_budget_to_deploy_gbp": 0.0, "buffer_reserve_gbp": 0.0, "reserve_floor_gbp": 0.0}
        t = se.generate_tldr_summary({"Main": {"rebalancing_plan": plan}})["Main"]
        self.assertIn("Sell 0.5 shares (cap rebalance)", t["sells_to_place"][0]["instruction"])
        self.assertNotIn("PROFIT", t["phase_label"])
        breach_plan = {"available_cash_gbp": 0.0, "reserve_floor_gbp": 0.0,
                       "cap_breaches": [{"kind": "cluster", "name": "AI Hardware & Semiconductors", "weight_pct": 31.0, "cap_pct": 24.0, "actioned": False}]}
        t = se.generate_tldr_summary({"Main": {"rebalancing_plan": breach_plan}})["Main"]
        self.assertEqual(t["phase"], "ALL_SET")
        self.assertNotIn("fully balanced", t["phase_desc"])
        t = se.generate_tldr_summary({"Main": {"rebalancing_plan": {"available_cash_gbp": 5000.0, "reserve_floor_gbp": 1500.0}}})["Main"]
        self.assertEqual(t["phase_desc"], "No action required today.")
        self.assertNotIn("fully balanced", t["phase_desc"])  # never claim balance the engine has not measured
        deferred = {"available_cash_gbp": 3000.0, "reserve_floor_gbp": 1500.0, "household_allocation": {
            "buys_allowed": False, "deployable_gbp": 1350.0,
            "cadence_note": "Monthly buy cadence: £1,350 deployable will be invested on the next buy day (Mon 02 Nov 2026)."}}
        t = se.generate_tldr_summary({"Main": {"rebalancing_plan": deferred}})["Main"]
        self.assertEqual(t["phase"], "ALL_SET")
        self.assertIn("next buy day", t["phase_desc"])


# ---------------------------------------------------------------------------
# analyze_position: data quality, R:R, stops, exit precedence, leverage policy
# ---------------------------------------------------------------------------
class TestAnalyzePosition(EngineTestCase):
    UP = alternating_path(252, 0.011, 0.008)
    DOWN = alternating_path(252, 0.006, 0.012)

    def setUp(self):
        super().setUp()
        for p in (
            mock.patch.object(se.research_agent, "get_macro_regime", return_value={"macro_score": 50, "sector_momentum": {}, "macro_data_ok": True}),
            mock.patch.object(se, "get_spy_history", return_value=None),
            mock.patch.object(se, "evaluate_minervini_template", return_value=dict(MINERVINI_BULL)),
            mock.patch.object(se, "evaluate_institutional_volume", return_value=dict(VOLUME_ACCUM)),
        ):
            p.start()
            self._patches.append(p)

    def run_analysis(self, ticker, research, closes, weekly=None, position=None, currency="GBP"):
        self.use_tickers({ticker: FakeTicker(daily=daily_frame(closes), fast_info={"currency": currency})})
        dive = mock.patch.object(se.research_agent, "conduct_deep_dive",
                                 side_effect=research if isinstance(research, Exception) else None,
                                 return_value=None if isinstance(research, Exception) else research)
        weekly_p = mock.patch.object(se, "evaluate_weekly_trend", return_value=dict(weekly or WEEKLY_UPTREND))
        with dive, weekly_p:
            return se.analyze_position(ticker, position, gbp_rate=1.25)

    ETF_RESEARCH = {"data_ok": True, "is_etf": True, "company_name": "Vanguard FTSE All-World", "fundamental_score": 91,
                    "macro_cluster": "Diversified Index Core", "target_mean_price": 0.0, "expected_5y_cagr_pct": 11.2}

    def test_contract_keys_and_buy_with_complete_data(self):
        res = self.run_analysis("VWRP.L", dict(self.ETF_RESEARCH), self.UP)
        for key in ("data_ok", "price_data_ok", "research_data_ok", "valuation_ok", "stop_breached", "risk_reward_ratio"):
            self.assertIn(key, res)
        self.assertTrue(res["data_ok"])
        self.assertTrue(res["valuation_ok"])
        self.assertFalse(res["stop_breached"])
        self.assertEqual(res["action_short"], "BUY")
        self.assertIsNone(res["risk_reward_ratio"])  # ETFs never get an R:R

    def test_research_failure_blocks_buy(self):
        res = self.run_analysis("VWRP.L", RuntimeError("research down"), self.UP)
        self.assertFalse(res["research_data_ok"])
        self.assertFalse(res["data_ok"])
        self.assertNotEqual(res["action_short"], "BUY")
        failed = {"data_ok": False, "fundamental_score": 50, "expected_5y_cagr_pct": None, "quality_grade": "N/A", "is_etf": True}
        res = self.run_analysis("VWRP.L", failed, self.UP)
        self.assertFalse(res["data_ok"])
        self.assertNotEqual(res["action_short"], "BUY")

    def test_missing_weekly_data_blocks_buy(self):
        weekly_missing = dict(WEEKLY_UPTREND, is_weekly_stage_2=False, data_ok=False, label="⚪ Weekly data unavailable")
        res = self.run_analysis("VWRP.L", dict(self.ETF_RESEARCH), self.UP, weekly=weekly_missing)
        self.assertFalse(res["price_data_ok"])
        self.assertNotEqual(res["action_short"], "BUY")

    def test_gold_and_mmf_have_no_rr(self):
        for t in ("SGLN.L", "CSH2.L"):
            res = self.run_analysis(t, dict(self.ETF_RESEARCH, target_mean_price=999999.0), self.UP)
            self.assertIsNone(res["risk_reward_ratio"], t)

    def test_stock_rr_uses_analyst_target_only(self):
        last = self.UP[-1]
        research = {"data_ok": True, "company_name": "Microsoft", "fundamental_score": 80, "macro_cluster": "Enterprise Cloud & Software",
                    "target_mean_price": last * 1.2, "expected_5y_cagr_pct": 12.0}
        res = self.run_analysis("MSFT", research, self.UP, currency="USD")
        price, stop = res["price"], res["trailing_stop"]
        downside = max(0.02 * price, price - stop) if stop < price else 0.02 * price
        self.assertAlmostEqual(res["risk_reward_ratio"], round((last * 1.2 - price) / downside, 2))
        below = dict(research, target_mean_price=last * 0.9)
        self.assertIsNone(self.run_analysis("MSFT", below, self.UP, currency="USD")["risk_reward_ratio"])
        self.assertLess(res["target_price"], price * 1.15)  # no artificial +15% target floor

    def test_exit_fires_before_trim_lockout_and_names_condition(self):
        today = datetime.now().strftime("%Y-%m-%d")
        position = {"quantity": 10.0, "currentPrice": self.DOWN[-1], "averagePrice": 100.0, "last_trim_date": today,
                    "last_entry_date": today, "currency": "USD"}
        research = {"data_ok": True, "company_name": "NVIDIA", "fundamental_score": 70, "macro_cluster": "AI Hardware & Semiconductors"}
        with mock.patch.object(se, "evaluate_minervini_template", return_value=dict(MINERVINI_BEAR)):
            res = self.run_analysis("NVDA", research, self.DOWN, weekly=WEEKLY_DOWNTREND, position=position, currency="USD")
        self.assertEqual(res["signal"], "🔴 SELL (EXIT)")
        self.assertIn("structural breakdown", res["reason"])

    def test_no_trend_exit_without_weekly_data(self):
        # A data glitch (weekly history unavailable) must never trigger an irreversible SELL on daily data alone.
        position = {"quantity": 10.0, "currentPrice": self.DOWN[-1], "averagePrice": 100.0, "currency": "USD"}
        research = {"data_ok": True, "company_name": "NVIDIA", "fundamental_score": 40, "macro_cluster": "AI Hardware & Semiconductors"}
        weekly_missing = dict(WEEKLY_DOWNTREND, is_weekly_stage_2=False, data_ok=False, label="⚪ Weekly data unavailable")
        with mock.patch.object(se, "evaluate_minervini_template", return_value=dict(MINERVINI_BEAR)):
            res = self.run_analysis("NVDA", research, self.DOWN, weekly=weekly_missing, position=position, currency="USD")
        self.assertNotEqual(res["signal"], "🔴 SELL (EXIT)")
        self.assertNotEqual(res["action_short"], "BUY")

    def test_broad_etf_never_exits_and_stop_breach_is_review_only(self):
        today = datetime.now().strftime("%Y-%m-%d")
        position = {"quantity": 10.0, "currentPrice": self.DOWN[-1], "averagePrice": 100.0, "last_trim_date": today, "currency": "GBP"}
        with mock.patch.object(se, "evaluate_minervini_template", return_value=dict(MINERVINI_BEAR)):
            res = self.run_analysis("VWRP.L", dict(self.ETF_RESEARCH), self.DOWN, weekly=WEEKLY_DOWNTREND, position=position)
        self.assertNotEqual(res["action_short"], "SELL")
        self.assertTrue(res["stop_breached"])
        self.assertEqual(res["signal"], "⚠️ HOLD — BELOW REFERENCE STOP (REVIEW)")

    def test_lockout_label_mentions_profits_only_when_in_profit(self):
        recent = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        research = {"data_ok": True, "company_name": "Microsoft", "fundamental_score": 80, "macro_cluster": "Enterprise Cloud & Software"}
        loss_pos = {"quantity": 5.0, "currentPrice": self.UP[-1], "averagePrice": self.UP[-1] * 1.3, "last_trim_date": recent, "currency": "USD"}
        res = self.run_analysis("MSFT", research, self.UP, position=loss_pos, currency="USD")
        self.assertEqual(res["signal"], "🟢 HOLD (RECENTLY TRIMMED — LOCKOUT)")
        self.assertNotIn("profit", (res["reason"] + res["plain_english_why"]).lower())
        gain_pos = dict(loss_pos, averagePrice=self.UP[-1] * 0.95)
        res = self.run_analysis("MSFT", research, self.UP, position=gain_pos, currency="USD")
        self.assertEqual(res["signal"], "🟢 HOLD (RECENTLY TRIMMED — LOCKOUT)")
        self.assertIn("profit", res["reason"].lower())

    def test_leveraged_etf_is_never_bought_and_zero_cap_sells(self):
        research = {"data_ok": True, "is_etf": True, "company_name": "WisdomTree NASDAQ 100 3x Daily Leveraged", "fundamental_score": 64,
                    "macro_cluster": "High-Beta Technology", "target_mean_price": 0.0}
        res = self.run_analysis("QQQ3.L", dict(research), self.UP)
        self.assertNotEqual(res["action_short"], "BUY")
        self.assertNotIn("80.7", res["reason"])
        se.REBALANCE_CONFIG["max_leveraged_etf_pct"] = 0.0
        res = self.run_analysis("QQQ3.L", dict(research), self.UP, position={"quantity": 3.0, "currentPrice": self.UP[-1], "currency": "GBP"})
        self.assertEqual(res["signal"], "🔴 SELL (LEVERAGE POLICY — 0% CAP)")

    def test_fallback_never_defaults_to_usd_or_stage_2(self):
        self.use_tickers({})  # every yfinance call fails
        with mock.patch.object(se.research_agent, "conduct_deep_dive", return_value={}):
            res = se.analyze_position("SGLN.L", {"quantity": 10.0, "currentPrice": 7500.0}, gbp_rate=1.25)
        self.assertFalse(res["valuation_ok"])
        self.assertFalse(res["data_ok"])
        self.assertFalse(res["price_data_ok"])
        self.assertEqual(res["price_gbp"], 0.0)
        self.assertEqual(res["minervini_stage"], "⚪ Data unavailable")
        self.assertNotIn("Stage 2", res["minervini_stage"])
        self.assertNotEqual(res["currency"], "USD")
        with mock.patch.object(se.research_agent, "conduct_deep_dive", return_value={}):
            res = se.analyze_position("SGLN.L", {"quantity": 10.0, "currentPrice": 7500.0, "currency": "GBX"}, gbp_rate=1.25)
        self.assertTrue(res["valuation_ok"])
        self.assertAlmostEqual(res["price_gbp"], 75.0)
        self.assertAlmostEqual(res["value_gbp"], 750.0)

    def test_fallback_leverage_policy_sell(self):
        se.REBALANCE_CONFIG["max_leveraged_etf_pct"] = 0.0
        self.use_tickers({})
        with mock.patch.object(se.research_agent, "conduct_deep_dive", return_value={}):
            res = se.analyze_position("QQQ3.L", {"quantity": 2.0, "currentPrice": 9000.0, "currency": "GBX", "last_trim_date": "2026-10-12"}, gbp_rate=1.25)
        self.assertEqual(res["signal"], "🔴 SELL (LEVERAGE POLICY — 0% CAP)")
        self.assertEqual(res["action_short"], "SELL")

    def test_signal_branches_run_without_errors(self):
        stock = {"data_ok": True, "company_name": "Microsoft", "fundamental_score": 80, "macro_cluster": "Enterprise Cloud & Software"}
        rally = [100.0] * 199 + alternating_path(53, 0.03, 0.003)  # RSI ~90, ~29% above EMA50 (small down days keep Wilder RSI defined)
        weekly_missing = dict(WEEKLY_UPTREND, is_weekly_stage_2=False, data_ok=False)
        bear_no_accum = dict(VOLUME_ACCUM, is_accumulation=False, up_down_vol_ratio=0.8)
        scenarios = [
            ("earnings", "MSFT", dict(stock, days_to_earnings=3), self.UP, WEEKLY_UPTREND, None, None, "EARNINGS IMMINENT"),
            ("weekly filter", "MSFT", stock, self.UP, WEEKLY_DOWNTREND, None, None, "WEEKLY DOWNTREND FILTER"),
            ("weekly missing", "MSFT", stock, self.UP, weekly_missing, None, None, "WEEKLY DOWNTREND FILTER"),
            ("asymmetry", "MSFT", dict(stock, target_mean_price=self.UP[-1] * 1.01), self.UP, WEEKLY_UPTREND, None, None, "WAIT FOR ASYMMETRY"),
            ("do nothing", "MSFT", stock, self.UP, WEEKLY_UPTREND, MINERVINI_BEAR, bear_no_accum, "DO NOTHING"),
            ("gold", "SGLN.L", dict(self.ETF_RESEARCH, company_name="iShares Physical Gold"), self.UP, WEEKLY_UPTREND, None, None, "CRISIS ALPHA SHIELD"),
            ("mmf", "CSH2.L", dict(self.ETF_RESEARCH, company_name="Amundi SONIA"), self.UP, WEEKLY_UPTREND, None, None, "SONIA CASH VAULT"),
            ("parabolic", "MSFT", stock, rally, WEEKLY_UPTREND, None, None, "PARABOLIC"),
        ]
        for label, ticker, research, closes, weekly, minervini, volume, expected in scenarios:
            position = None
            if label == "parabolic":
                position = {"quantity": 10.0, "currentPrice": closes[-1], "averagePrice": 100.0, "currency": "USD"}
            patches = []
            if minervini is not None:
                patches.append(mock.patch.object(se, "evaluate_minervini_template", return_value=dict(minervini)))
            if volume is not None:
                patches.append(mock.patch.object(se, "evaluate_institutional_volume", return_value=dict(volume)))
            for p in patches:
                p.start()
            try:
                res = self.run_analysis(ticker, dict(research), closes, weekly=weekly, position=position, currency="USD" if ticker == "MSFT" else "GBP")
            finally:
                for p in patches:
                    p.stop()
            self.assertNotIn("Analysis error", res["reason"], label)
            self.assertIn(expected, res["signal"], f"{label}: {res['signal']} / {res['reason']}")


class TestEndToEndOffline(EngineTestCase):
    """Mirrors main.py's call order using the real indicator / weekly-trend code on synthetic market data."""

    def test_pipeline_two_accounts(self):
        se.REBALANCE_CONFIG["max_leveraged_etf_pct"] = 0.0
        n_w = 260
        rng = np.random.RandomState(11)
        fx_w = 1.25 + 0.05 * np.sin(np.arange(n_w) / 8.0)
        spy_w = 400.0 * np.cumprod(1.0 + rng.normal(0.002, 0.02, n_w))
        up_daily = alternating_path(252, 0.011, 0.008)

        def fake(ccy, scale=1.0, weekly=None):
            return FakeTicker(daily=daily_frame([c * scale for c in up_daily]),
                              weekly=weekly_frame(weekly if weekly is not None else spy_w * scale / 4.0),
                              fast_info={"currency": ccy})

        mapping = {
            "SPY": FakeTicker(daily=daily_frame([c * 4.0 for c in up_daily]), weekly=weekly_frame(spy_w)),
            "GBPUSD=X": FakeTicker(weekly=weekly_frame(fx_w)),
            "VWRP.L": fake("GBP"), "QQQ3.L": fake("GBP", 2.0), "CSH2.L": fake("GBP", 1.2),
            "NVDA": fake("USD", 1.5), "MSFT": fake("USD", 3.0), "V": fake("USD", 2.5),
            "ACME": FakeTicker(fast_info={"lastPrice": 200.0, "currency": "USD"}),
        }
        self.use_tickers(mapping)
        research = {
            "VWRP.L": {"data_ok": True, "is_etf": True, "company_name": "Vanguard FTSE All-World", "fundamental_score": 91, "macro_cluster": "Diversified Index Core", "target_mean_price": 0.0, "expected_5y_cagr_pct": 11.2},
            "QQQ3.L": {"data_ok": True, "is_etf": True, "company_name": "WisdomTree NASDAQ 100 3x Daily Leveraged", "fundamental_score": 64, "macro_cluster": "High-Beta Technology", "target_mean_price": 0.0},
            "CSH2.L": {"data_ok": True, "is_etf": True, "company_name": "Amundi Smart Overnight Return", "fundamental_score": 91, "macro_cluster": "Uncorrelated Crash Hedge & Gold", "target_mean_price": 0.0},
            "NVDA": {"data_ok": True, "company_name": "NVIDIA", "fundamental_score": 82, "macro_cluster": "AI Hardware & Semiconductors", "target_mean_price": 400.0, "expected_5y_cagr_pct": 28.5, "expected_5y_cagr_capped": "cap"},
            "MSFT": {"data_ok": True, "company_name": "Microsoft", "fundamental_score": 84, "macro_cluster": "Enterprise Cloud & Software", "target_mean_price": 0.0, "expected_5y_cagr_pct": 13.0},
            "V": {"data_ok": False, "company_name": "Visa", "fundamental_score": 50, "macro_cluster": "Financials & Payment Rails", "expected_5y_cagr_pct": None},
        }
        outside = {"Main": [{"ticker": "ACME", "shares_vested": 100.0, "shares_unvested": 50.0, "currency": "USD",
                        "macro_cluster": "Consumer & Digital Platforms", "value_hint_usd_vested": 15000.0, "value_hint_usd_unvested": 7500.0}]}
        se._CACHED_GBP_RATE = 1.25
        last = up_daily[-1]
        positions = {
            "Main": [("VWRP.L", {"quantity": 300.0, "currentPrice": last, "averagePrice": 100.0}),
                    ("CSH2.L", {"quantity": 100.0, "currentPrice": last * 1.2, "averagePrice": 110.0}),
                    ("NVDA", {"quantity": 40.0, "currentPrice": last * 1.5, "averagePrice": 120.0})],
            "Secondary": [("VWRP.L", {"quantity": 150.0, "currentPrice": last, "averagePrice": 100.0}),
                     ("QQQ3.L", {"quantity": 20.0, "currentPrice": last * 2.0, "averagePrice": 150.0, "last_trim_date": "2026-10-12"}),
                     ("NVDA", {"quantity": 10.0, "currentPrice": last * 1.5, "averagePrice": 120.0})],
        }
        cash = {"Main": 0.0, "Secondary": 6000.0}
        with mock.patch.object(se.research_agent, "conduct_deep_dive", side_effect=lambda t: dict(research.get(t, {}))), \
                mock.patch.object(se.research_agent, "get_macro_regime", return_value={"macro_score": 50, "sector_momentum": {}, "macro_data_ok": True}), \
                mock.patch.object(config, "EARMARKED_TICKERS", {"CSH2.L"}), \
                mock.patch.object(config, "OUTSIDE_HOLDINGS", outside), \
                mock.patch.object(se, "WATCHLIST", [{"ticker": "MSFT", "name": "Microsoft", "theme": "Cloud"},
                                                    {"ticker": "V", "name": "Visa", "theme": "Payments"},
                                                    {"ticker": "COIN", "name": "Coinbase", "theme": "Crypto Exchange"}]):
            raw = {}
            for acc, plist in positions.items():
                analyses = [se.analyze_position(t, dict(p), gbp_rate=1.25) for t, p in plist]
                invested = sum(a["value_gbp"] for a in analyses)
                raw[acc] = {"cash": {"free_gbp": cash[acc], "blocked_gbp": 0.0, "total": invested + cash[acc]}, "analyses": analyses}
            for acc in raw:
                for a in raw[acc]["analyses"]:
                    self.assertTrue(a["valuation_ok"], a["ticker"])
                    self.assertEqual(a["fx_basis"], "GBP", a["ticker"])
            view = se.build_consolidated_portfolio_view(raw, gbp_rate=1.25)
            opps = se.scan_watchlist_opportunities(set(), gbp_rate=1.25)
            plans = {
                acc: se.build_rebalancing_plan(
                    analyses=raw[acc]["analyses"], available_cash_gbp=cash[acc], new_opportunities=opps, gbp_rate=1.25,
                    min_cash_reserve_gbp=1500.0 if acc == "Main" else 500.0, global_context=view,
                    official_total_gbp=raw[acc]["cash"]["total"],
                )
                for acc in raw
            }
            view2 = se.build_consolidated_portfolio_view(raw, gbp_rate=1.25, account_plans=plans)
            tldr = se.generate_tldr_summary({acc: {"rebalancing_plan": plans[acc]} for acc in plans})

        self.assertNotIn("V", [o["ticker"] for o in opps])      # research data_ok False
        self.assertNotIn("COIN", [o["ticker"] for o in opps])   # excluded via EXCLUDED_TICKERS
        self.assertEqual(plans["Main"]["account_name"], "Main")
        self.assertEqual(plans["Secondary"]["account_name"], "Secondary")
        self.assertEqual(plans["Main"]["earmarked_tickers"], ["CSH2.L"])
        self.assertGreater(plans["Main"]["outside_vested_gbp"], 0.0)
        for acc, plan in plans.items():
            self.assertIsNone(plan["data_degraded_reason"], acc)
            orders = plan["sell_orders"] + plan["buy_orders"]
            self.assertFalse([o for o in orders if o["ticker"] in ("CSH2.L", "ACME")], acc)
            self.assertTrue(all(b.get("data_ok") for b in plan["buy_orders"]))
            self.assertEqual(plan["buy_orders"], [] if acc == "Main" else plan["buy_orders"])  # Main has £0 cash
            for b in plan["buy_orders"]:
                self.assertFalse(se.is_leveraged_etf(b["ticker"]))
        secondary_sells = {s["ticker"]: s for s in plans["Secondary"]["sell_orders"]}
        self.assertEqual(secondary_sells["QQQ3.L"]["action"], "🔴 SELL (LEVERAGE POLICY — 0% CAP)")
        self.assertAlmostEqual(secondary_sells["QQQ3.L"]["shares"], 20.0)
        rp = view2["risk_profile_5y"]
        self.assertEqual(rp["post_lev_3x_pct"], 0.0)
        self.assertGreater(rp["current_lev_3x_pct"], 0.0)
        self.assertEqual(view2["holdings"]["NVDA"]["accounts"], ["Main", "Secondary"])
        self.assertIn(tldr["Main"]["phase"], ("ALL_SET", "SELL_FIRST", "TRIM_FIRST"))
        self.assertIn("unfunded", tldr["Main"]["reserve_note"].lower())


# ---------------------------------------------------------------------------
# Household strategic allocation (Phase 1, 2026-10-19)
# ---------------------------------------------------------------------------
def fund(ticker, price_gbp, cluster, value_gbp=0.0, **overrides):
    """A priced fund analysis (e.g. from the watchlist scan) that may or may not be held."""
    return make_analysis(ticker, value_gbp, price_gbp=price_gbp, macro_cluster=cluster, **overrides)


def bucket_universe():
    return [
        fund("VWRP.L", 120.0, "Diversified Index Core"),
        fund("SGLN.L", 60.0, "Uncorrelated Crash Hedge & Gold"),
        fund("IGLS.L", 130.0, "Defensive Gilts & Cash"),
    ]


class TestHouseholdHelpers(EngineTestCase):
    allocation_cfg = HOUSEHOLD_ALLOCATION_CFG

    def test_bucket_classification(self):
        expected = {
            "VWRP.L": "core", "VUAA.L": "core", "VUAG.L": "core", "CSPX.L": "core",
            "CNX1.L": "satellite", "CNDX.L": "satellite", "QQQ3.L": "satellite", "NVDA": "satellite",
            "BRK-B": "satellite", "SMGB.L": "satellite",
            "SGLN.L": "defensive", "IGLS.L": "defensive", "ERNS.L": "defensive", "CSH2.L": "defensive",
        }
        for ticker, bucket in expected.items():
            self.assertEqual(se.classify_bucket(ticker), bucket, ticker)

    def test_crash_shield_is_strict(self):
        for t in ("BRK-B", "LLY", "ABBV", "VWRP.L", "MSFT"):
            self.assertFalse(se.is_crash_shield_asset(t, "Defensive Ballast & Value", 0.2, 10.0), t)
        for t in ("SGLN.L", "IGLS.L", "CSH2.L"):
            self.assertTrue(se.is_crash_shield_asset(t), t)

    def test_big_tech_lookthrough_weights(self):
        self.assertEqual(se.big_tech_weight("NVDA"), 1.0)
        self.assertEqual(se.big_tech_weight("GOOGL"), 1.0)
        self.assertAlmostEqual(se.big_tech_weight("VWRP.L"), 0.22)
        self.assertAlmostEqual(se.big_tech_weight("VUAA.L"), 0.35)
        self.assertAlmostEqual(se.big_tech_weight("CNX1.L"), 0.50)
        self.assertAlmostEqual(se.big_tech_weight("QQQ3.L"), 1.50)
        self.assertEqual(se.big_tech_weight("BRK-B"), 0.0)
        self.assertEqual(se.big_tech_weight("SGLN.L"), 0.0)

    def test_uk_bank_holidays_and_buy_days(self):
        from datetime import date
        self.assertIn(date(2026, 12, 28), se.uk_bank_holidays(2026))  # Boxing Day (Sat) substitute
        self.assertIn(date(2028, 1, 3), se.uk_bank_holidays(2028))    # New Year's Day (Sat) substitute
        self.assertIn(date(2027, 3, 26), se.uk_bank_holidays(2027))   # Good Friday
        self.assertEqual(se.first_london_business_day(2026, 1), date(2026, 1, 2))
        self.assertEqual(se.first_london_business_day(2026, 11), date(2026, 11, 2))  # 1 Nov is a Sunday
        self.assertEqual(se.first_london_business_day(2028, 5), date(2028, 5, 2))    # 1 May 2028 = bank holiday
        self.assertEqual(se.next_monthly_buy_day(date(2026, 10, 19)), date(2026, 11, 2))
        self.assertEqual(se.next_monthly_buy_day(date(2026, 11, 2)), date(2026, 11, 2))
        self.assertEqual(se.next_monthly_buy_day(date(2026, 12, 31)), date(2027, 1, 4))

    def test_allocate_new_money_follows_shortfall_and_never_feeds_overweight(self):
        alloc = se.allocate_new_money(5000.0, {"core": 22000.0, "satellite": 75000.0, "defensive": 3000.0}, 105000.0)
        self.assertEqual(alloc["satellite"], 0.0)
        self.assertAlmostEqual(sum(alloc.values()), 5000.0, places=6)
        core_gap, def_gap = 0.65 * 105000 - 22000, 0.15 * 105000 - 3000
        self.assertAlmostEqual(alloc["core"], 5000.0 * core_gap / (core_gap + def_gap), places=6)

    def test_allocate_new_money_out_of_band_bucket_first(self):
        # Core 50% (more than 5pp below 65%) is filled before in-band shortfalls
        alloc = se.allocate_new_money(5000.0, {"core": 50000.0, "satellite": 19000.0, "defensive": 26000.0}, 100000.0)
        self.assertAlmostEqual(alloc["core"], 5000.0, places=6)
        # All in band: split across the under-target buckets in proportion to their shortfall
        alloc = se.allocate_new_money(5000.0, {"core": 64000.0, "satellite": 21000.0, "defensive": 10000.0}, 100000.0)
        self.assertEqual(alloc["satellite"], 0.0)
        self.assertAlmostEqual(alloc["core"], 5000.0 * 1000.0 / 6000.0, places=6)
        self.assertAlmostEqual(alloc["defensive"], 5000.0 * 5000.0 / 6000.0, places=6)
        self.assertEqual(se.allocate_new_money(0.0, {"core": 1.0}, 10.0), {"core": 0.0, "satellite": 0.0, "defensive": 0.0})

    def test_household_mix_values_status_and_post_trade(self):
        holdings = [{"ticker": "VWRP.L", "value_gbp": 20000.0}, {"ticker": "NVDA", "value_gbp": 70000.0},
                    {"ticker": "SGLN.L", "value_gbp": 2000.0}]
        outside = [{"account": "Main", "ticker": "GOOGL", "vested_gbp": 5000.0, "unvested_gbp": 9000.0}]
        mix = se.compute_household_mix(holdings, 3000.0, outside, excluded_cash_gbp=6500.0,
                                       sell_orders=[{"ticker": "NVDA", "estimated_gbp": 10000.0}],
                                       buy_orders=[{"ticker": "VWRP.L", "estimated_gbp": 2000.0, "bucket": "core"}])
        self.assertEqual(mix["basis_gbp"], 100000.0)  # unvested and excluded top-up cash never count
        self.assertEqual(mix["values_gbp"], {"core": 20000.0, "satellite": 75000.0, "defensive": 5000.0})
        rows = {r["bucket"]: r for r in mix["buckets"]}
        self.assertEqual(rows["core"]["status"], "under")
        self.assertEqual(rows["satellite"]["status"], "over")
        self.assertEqual(rows["defensive"]["status"], "under")
        self.assertEqual(rows["core"]["post_trade_gbp"], 22000.0)
        self.assertEqual(rows["satellite"]["post_trade_gbp"], 65000.0)
        self.assertEqual(rows["defensive"]["post_trade_gbp"], 13000.0)
        self.assertEqual(mix["excluded_cash_gbp"], 6500.0)
        self.assertTrue(mix["post_trade_available"])
        self.assertFalse(se.compute_household_mix(holdings, 0.0)["post_trade_available"])

    def test_big_tech_exposure_with_lookthrough_and_outside_shares(self):
        holdings = [{"account": "Main", "ticker": "NVDA", "value_gbp": 10000.0},
                    {"account": "Second", "ticker": "VUAA.L", "value_gbp": 10000.0},
                    {"account": "Second", "ticker": "LIN", "value_gbp": 10000.0}]
        outside = [{"account": "Main", "ticker": "GOOGL", "vested_gbp": 20000.0}]
        bt = se.compute_big_tech_exposure(holdings, outside, 100000.0,
                                          sell_orders=[{"ticker": "NVDA", "estimated_gbp": 2000.0}],
                                          buy_orders=[{"ticker": "VWRP.L", "estimated_gbp": 5000.0}])
        self.assertAlmostEqual(bt["exposure_gbp"], 10000.0 + 3500.0 + 20000.0)
        self.assertEqual(bt["exposure_pct"], 33.5)
        self.assertTrue(bt["over_cap"])
        self.assertAlmostEqual(bt["post_trade_exposure_gbp"], 33500.0 - 2000.0 + 1100.0)
        self.assertEqual(bt["breakdown"][0]["ticker"], "GOOGL")
        self.assertEqual({r["source"] for r in bt["breakdown"]}, {"direct", "fund", "outside"})
        self.assertNotIn("LIN", [r["ticker"] for r in bt["breakdown"]])


class TestHouseholdPlan(EngineTestCase):
    allocation_cfg = HOUSEHOLD_ALLOCATION_CFG

    def setUp(self):
        super().setUp()
        from datetime import date
        self.off_cycle_day = date(2026, 10, 19)  # not the first London business day of the month
        self.buy_day = date(2026, 11, 2)

    def hh_plan(self, analyses, cash, opps=None, universe=None, as_of=None, **kwargs):
        kwargs.setdefault("min_cash_reserve_gbp", 500.0)
        return self.plan(analyses, cash=cash, opps=opps, account_name="Second",
                         instrument_universe=bucket_universe() if universe is None else universe,
                         as_of=as_of or self.off_cycle_day, **kwargs)

    def tech_heavy_book(self):
        return [make_analysis("NVDA", 60000.0, macro_cluster="AI Hardware & Semiconductors"),
                make_analysis("MSFT", 30000.0, action_short="BUY", conviction_score=90),
                fund("VUAA.L", 100.0, "Diversified Index Core", value_gbp=10000.0)]

    def test_off_cycle_buy_goes_to_core_and_defensive_only(self):
        meta = make_analysis("META", 0.0, conviction_score=95, is_new_addition=True)
        plan = self.hh_plan(self.tech_heavy_book(), cash=6000.0, opps=[meta])
        buys = {b["ticker"]: b for b in plan["buy_orders"]}
        self.assertEqual(set(buys), {"VWRP.L", "SGLN.L", "IGLS.L"})
        self.assertEqual(buys["VWRP.L"]["bucket"], "core")
        self.assertEqual(buys["SGLN.L"]["bucket"], "defensive")
        self.assertEqual(buys["IGLS.L"]["bucket"], "defensive")
        self.assertAlmostEqual(sum(b["estimated_gbp"] for b in plan["buy_orders"]), 5500.0, delta=1.0)
        self.assertGreater(buys["VWRP.L"]["estimated_gbp"], buys["SGLN.L"]["estimated_gbp"] + buys["IGLS.L"]["estimated_gbp"])
        alloc = plan["household_allocation"]
        self.assertTrue(alloc["buys_allowed"])
        self.assertIn("Off-cycle", alloc["cadence_note"])
        self.assertEqual(alloc["by_bucket_gbp"]["satellite"], 0.0)
        notes = " ".join(alloc["notes"])
        self.assertIn("Satellites", notes)
        self.assertIn("big tech", notes)
        self.assertTrue(all(b["is_new"] for b in plan["buy_orders"]))  # none of the bucket funds is held here

    def test_monthly_cadence_defers_small_buys_until_buy_day(self):
        book = self.cap_compliant_book()
        plan = self.hh_plan(book, cash=2200.0)  # £1,700 deployable < £2,000 off-cycle threshold
        self.assertEqual(plan["buy_orders"], [])
        self.assertEqual(plan["sell_orders"], [])
        alloc = plan["household_allocation"]
        self.assertFalse(alloc["buys_allowed"])
        self.assertEqual(alloc["next_buy_day"], "2026-11-02")
        self.assertEqual(alloc["deployable_gbp"], 1700.0)
        self.assertEqual(plan["net_budget_to_deploy_gbp"], 0.0)
        tldr = se.generate_tldr_summary({"Second": {"rebalancing_plan": plan}})["Second"]
        self.assertEqual(tldr["phase"], "ALL_SET")
        self.assertIn("next buy day", tldr["phase_desc"])
        plan = self.hh_plan(self.cap_compliant_book(), cash=2200.0, as_of=self.buy_day)
        self.assertTrue(plan["buy_orders"])
        self.assertIn("Monthly buy day", plan["household_allocation"]["cadence_note"])

    def test_daily_cadence_buys_small_amounts_any_day(self):
        config.ALLOCATION_CONFIG["buy_cadence"] = "daily"
        plan = self.hh_plan(self.cap_compliant_book(), cash=2200.0)
        self.assertTrue(plan["buy_orders"])
        self.assertTrue(plan["household_allocation"]["buys_allowed"])

    def test_reserve_invariant_still_blocks_all_buys(self):
        plan = self.hh_plan(self.cap_compliant_book(), cash=1400.0, as_of=self.buy_day)  # < £500 reserve + £1,000
        self.assertEqual(plan["buy_orders"], [])
        tldr = se.generate_tldr_summary({"Second": {"rebalancing_plan": plan}})["Second"]
        self.assertEqual(tldr["phase_label"], "✅ ALL SET — SIT BACK & RELAX")

    def cap_compliant_book(self):
        """Tech-heavy, but no single position above the 15% account cap (so no cap trims interfere)."""
        names = ("NVDA", "MSFT", "AAPL", "AMZN", "META", "LIN")
        return [make_analysis(t, 14000.0, macro_cluster=f"Cluster {t}") for t in names] + [
            fund("VUAA.L", 100.0, "Diversified Index Core", value_gbp=10000.0)]

    def test_underweight_satellites_use_stock_selection_with_5pct_cap(self):
        book = [fund("VWRP.L", 100.0, "Diversified Index Core", value_gbp=80000.0),
                fund("SGLN.L", 60.0, "Uncorrelated Crash Hedge & Gold", value_gbp=8000.0),
                fund("IGLS.L", 130.0, "Defensive Gilts & Cash", value_gbp=7000.0)]
        lin = make_analysis("LIN", 0.0, price_gbp=400.0, conviction_score=80, is_new_addition=True,
                            macro_cluster="Infrastructure & Electrification")
        plan = self.hh_plan(book, cash=6000.0, opps=[lin])
        buys = {b["ticker"]: b for b in plan["buy_orders"]}
        self.assertIn("LIN", buys)
        self.assertEqual(buys["LIN"]["bucket"], "satellite")
        basis = plan["household_allocation"]["deployable_gbp"] + 0.0
        self.assertEqual(plan["household_allocation"]["by_bucket_gbp"]["satellite"], basis)
        self.assertLessEqual(buys["LIN"]["estimated_gbp"], 0.05 * 101000.0 + 0.01)  # 5% household single-stock cap
        self.assertAlmostEqual(sum(b["estimated_gbp"] for b in plan["buy_orders"]), 5500.0, delta=1.0)
        self.assertIn("Model forward 5Y estimate", buys["LIN"]["plain_english_why"])
        self.assertIn("info only", buys["LIN"]["plain_english_why"])

    def test_big_tech_cap_blocks_big_tech_satellites(self):
        book = [fund("VUAG.L", 100.0, "Diversified Index Core", value_gbp=65000.0),
                make_analysis("MSFT", 10000.0),
                fund("SGLN.L", 60.0, "Uncorrelated Crash Hedge & Gold", value_gbp=10000.0),
                fund("IGLS.L", 130.0, "Defensive Gilts & Cash", value_gbp=10000.0)]
        meta = make_analysis("META", 0.0, conviction_score=95, is_new_addition=True,
                             macro_cluster="Consumer & Digital Platforms")
        lin = make_analysis("LIN", 0.0, price_gbp=400.0, conviction_score=70, is_new_addition=True,
                            macro_cluster="Infrastructure & Electrification")
        plan = self.hh_plan(book, cash=6000.0, opps=[meta, lin])
        tickers = [b["ticker"] for b in plan["buy_orders"]]
        self.assertNotIn("META", tickers)  # big tech is ~32% of the household vs the 30% cap
        self.assertIn("LIN", tickers)
        self.assertIn("big tech", " ".join(plan["household_allocation"]["notes"]))

    def test_household_single_stock_cap_blocks_adds(self):
        book = [fund("VWRP.L", 100.0, "Diversified Index Core", value_gbp=80000.0),
                make_analysis("LIN", 6000.0, action_short="BUY", conviction_score=85,
                              macro_cluster="Infrastructure & Electrification")]
        plan = self.hh_plan(book, cash=6000.0)
        self.assertNotIn("LIN", [b["ticker"] for b in plan["buy_orders"]])  # 6.5% of the household > 5%
        self.assertEqual(plan["household_allocation"]["ordered_by_bucket_gbp"]["satellite"], 0.0)
        self.assertTrue(any(b["bucket"] == "defensive" for b in plan["buy_orders"]))  # unspent money still invested

    def test_usd_lines_never_get_new_money(self):
        config.ALLOCATION_CONFIG.update(LEGACY_ALLOCATION_CFG)  # all new money to satellites
        book = [fund("CNDX.L", 100.0, "Diversified Index Core", value_gbp=5000.0, action_short="BUY", conviction_score=90)]
        cnx1 = fund("CNX1.L", 100.0, "Diversified Index Core", conviction_score=90, is_new_addition=True)
        plan = self.hh_plan(book, cash=6000.0, opps=[fund("VUAA.L", 100.0, "Diversified Index Core", is_new_addition=True), cnx1])
        tickers = [b["ticker"] for b in plan["buy_orders"]]
        self.assertNotIn("CNDX.L", tickers)
        self.assertIn("CNX1.L", tickers)

    def test_unpriced_core_fund_stays_in_cash_with_note(self):
        universe = [fund("SGLN.L", 60.0, "Uncorrelated Crash Hedge & Gold"), fund("IGLS.L", 130.0, "Defensive Gilts & Cash")]
        plan = self.hh_plan(self.tech_heavy_book(), cash=6000.0, universe=universe)
        self.assertNotIn("VWRP.L", [b["ticker"] for b in plan["buy_orders"]])
        self.assertIn("VWRP.L not bought because no valid GBP price", " ".join(plan["household_allocation"]["notes"]))
        self.assertTrue(any(b["bucket"] == "defensive" for b in plan["buy_orders"]))

    def test_pending_core_buy_is_not_duplicated(self):
        pending = {"pending_buys": {"VWRP.L": {"estimated_value": 4000.0}}, "pending_sells": {}, "has_orders_scope": True}
        plan = self.hh_plan(self.tech_heavy_book(), cash=6000.0, pending_orders_data=pending)
        self.assertNotIn("VWRP.L", [b["ticker"] for b in plan["buy_orders"]])
        self.assertIn("already pending", " ".join(plan["household_allocation"]["notes"]))

    def test_forward_cagr_no_longer_changes_sizing(self):
        config.ALLOCATION_CONFIG.update(LEGACY_ALLOCATION_CFG)
        hi = make_analysis("AAA", 0.0, conviction_score=80, is_new_addition=True, expected_5y_cagr_pct=28.5,
                           macro_cluster="Financials & Payment Rails")
        lo = make_analysis("BBB", 0.0, conviction_score=80, is_new_addition=True, expected_5y_cagr_pct=5.5,
                           macro_cluster="Healthcare & Biopharma")
        plan = self.hh_plan([make_analysis("VWRP.L", 50000.0, macro_cluster="Diversified Index Core")], cash=8000.0,
                            opps=[hi, lo])
        buys = {b["ticker"]: b for b in plan["buy_orders"]}
        self.assertEqual(set(buys), {"AAA", "BBB"})
        self.assertAlmostEqual(buys["AAA"]["estimated_gbp"], buys["BBB"]["estimated_gbp"], delta=0.01)
        self.assertEqual(buys["AAA"]["rebalance_score"], buys["BBB"]["rebalance_score"])

    def _two_accounts(self, topup=0.0):
        return {
            "Main": {"cfg": {"reserve_topup_gbp": topup},
                    "cash": {"free_gbp": 6000.0, "blocked_gbp": 0.0},
                    "analyses": [fund("VWRP.L", 100.0, "Diversified Index Core", value_gbp=40000.0),
                                 make_analysis("NVDA", 40000.0, macro_cluster="AI Hardware & Semiconductors")]},
            "Second": {"cfg": {}, "cash": {"free_gbp": 6000.0, "blocked_gbp": 0.0},
                     "analyses": [make_analysis("MSFT", 10000.0)]},
        }

    def _account_plan(self, accounts, name, view):
        return self.plan(accounts[name]["analyses"], cash=accounts[name]["cash"]["free_gbp"], account_name=name,
                         min_cash_reserve_gbp=500.0, global_context=view, instrument_universe=bucket_universe(),
                         as_of=self.off_cycle_day)

    def test_shared_ledger_prevents_double_filling_across_accounts(self):
        accounts = self._two_accounts()
        view = se.build_consolidated_portfolio_view(accounts, gbp_rate=1.25)
        self.assertEqual(view["household_mix"]["basis_gbp"], 102000.0)
        main = self._account_plan(accounts, "Main", view)
        ledger = view["household_mix"]["ledger"]
        main_core = sum(b["estimated_gbp"] for b in main["buy_orders"] if b["bucket"] == "core")
        self.assertAlmostEqual(ledger["core"], main_core, places=6)
        second = self._account_plan(accounts, "Second", view)
        fresh_view = se.build_consolidated_portfolio_view(self._two_accounts(), gbp_rate=1.25)
        second_fresh = self._account_plan(self._two_accounts(), "Second", fresh_view)
        # Main already filled part of the household core gap, so Second's plan puts less into core
        self.assertLess(second["household_allocation"]["by_bucket_gbp"]["core"],
                        second_fresh["household_allocation"]["by_bucket_gbp"]["core"])
        rebuilt = se.build_consolidated_portfolio_view(accounts, gbp_rate=1.25, account_plans={"Main": main, "Second": second})
        rows = {r["bucket"]: r for r in rebuilt["household_mix"]["buckets"]}
        spent = sum(b["estimated_gbp"] for p in (main, second) for b in p["buy_orders"])
        self.assertAlmostEqual(rows["core"]["post_trade_gbp"] - rows["core"]["current_gbp"],
                               sum(b["estimated_gbp"] for p in (main, second) for b in p["buy_orders"] if b["bucket"] == "core"), places=2)
        self.assertAlmostEqual(sum(r["post_trade_gbp"] for r in rows.values()), rebuilt["household_mix"]["basis_gbp"], places=2)
        self.assertGreater(spent, 0.0)
        self.assertTrue(rebuilt["household_mix"]["post_trade_available"])

    def test_topup_cash_and_earmarked_funds_are_outside_the_household_basis(self):
        accounts = self._two_accounts(topup=6500.0)
        accounts["Main"]["analyses"].append(make_analysis("CSH2.L", 20000.0, macro_cluster="Uncorrelated Crash Hedge & Gold"))
        with mock.patch.object(config, "EARMARKED_TICKERS", {"CSH2.L"}):
            view = se.build_consolidated_portfolio_view(accounts, gbp_rate=1.25)
        mix = view["household_mix"]
        self.assertEqual(mix["excluded_cash_gbp"], 6000.0)  # all of Main's free cash is the parked top-up
        self.assertEqual(mix["basis_gbp"], 40000.0 + 40000.0 + 10000.0 + 6000.0)  # CSH2 and top-up cash excluded
        self.assertIn("big_tech", view)
        self.assertAlmostEqual(view["big_tech"]["exposure_gbp"], 40000.0 * 0.22 + 40000.0 + 10000.0)

    def test_risk_profile_shield_uses_strict_definition(self):
        book = [make_analysis("BRK-B", 10000.0, macro_cluster="Defensive Ballast & Value", beta_5y=0.3, crash_2022_ret_pct=4.0),
                fund("IGLS.L", 130.0, "Defensive Gilts & Cash", value_gbp=10000.0, beta_5y=0.05, crash_2022_ret_pct=-3.0),
                make_analysis("MSFT", 80000.0)]
        rp = se.compute_portfolio_5y_risk_profile(book, 0.0, 0.0, 100000.0)
        self.assertEqual(rp["current_shield_pct"], 10.0)     # IGLS only; BRK-B is a stock
        self.assertEqual(rp["current_defensive_pct"], 10.0)


class TestCompiles(unittest.TestCase):
    def test_py_compile(self):
        py_compile.compile(os.path.join(REPO_ROOT, "strategy_engine.py"), doraise=True)


if __name__ == "__main__":
    unittest.main()

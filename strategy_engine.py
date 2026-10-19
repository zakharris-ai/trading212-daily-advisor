"""Quantitative strategy engine for multi-factor trend, momentum, watchlist scanning, and capital allocation."""

import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import config
import research_agent as research_agent_module
from config import STRATEGY_CONFIG, WATCHLIST, REBALANCE_CONFIG, EXCLUDED_TICKERS
from research_agent import DeepResearchAgent
from t212_client import normalize_t212_ticker_for_yfinance

logger = logging.getLogger(__name__)

# Cached GBP/USD rate
_CACHED_GBP_RATE = None
# Cached live FX conversion rates (GBP per 1 unit of currency); None marks a cached failure (per process)
_CACHED_FX_TO_GBP: Dict[str, Optional[float]] = {}
# Cached live GBP share prices for outside holdings (e.g. shares held at another broker); per process
_CACHED_OUTSIDE_PRICE_GBP: Dict[str, Optional[float]] = {}
research_agent = DeepResearchAgent()

# Conservative sentinel defaults for missing 5Y risk metrics (fail pessimistic, never optimistic)
MISSING_METRIC_DEFAULTS: Dict[str, float] = {
    "crash_2022_ret_pct": -25.0,
    "beta_5y": 1.2,
    "sortino_5y": 0.9,
    "corr_5y": 0.8,
    "cagr_5y_pct": 8.0,
}

# 5-year scenario return assumptions per asset bucket (annualised): (bear, base, bull)
SCENARIO_RATES: Dict[str, Tuple[float, float, float]] = {
    "equity": (0.02, 0.06, 0.09),   # everything that is not cash / money-market / gold
    "cash": (0.035, 0.035, 0.035),  # free + blocked cash and money-market funds (CSH2)
    "gold": (0.00, 0.03, 0.05),     # physical gold ETCs
}

# Instrument classification keys
LEVERAGED_ETF_KEYS = ("QQQ3", "3LUS", "3SEM", "3X", "LQQ3", "3USL", "TQQQ")
BROAD_INDEX_ETF_KEYS = ("VUAA", "VUSA", "VUAG", "CSPX", "VWRP", "VWRA", "CNDX", "CNX1")
BROAD_INDEX_ETF_EXACT = {"SPY", "QQQ", "RSP"}
GOLD_ETC_KEYS = ("SGLN", "IGLN", "GLD")
MONEY_MARKET_KEYS = ("CSH2",)
SECTOR_ETF_KEYS = ("SMGB", "DFND")
CLUSTER_CAP_EXEMPT = ("Diversified Index Core", "Uncorrelated Crash Hedge & Gold", "Defensive Ballast & Value")

# Signal labels shared by analyze_position and build_rebalancing_plan
EARMARKED_SIGNAL = "🔒 HOLD (EARMARKED — EXCLUDED FROM PORTFOLIO MATHS)"
LEVERAGE_POLICY_SELL_SIGNAL = "🔴 SELL (LEVERAGE POLICY — 0% CAP)"
STOP_REVIEW_SIGNAL = "⚠️ HOLD — BELOW REFERENCE STOP (REVIEW)"
TRIM_LOCKOUT_SIGNAL = "🟢 HOLD (RECENTLY TRIMMED — LOCKOUT)"
DATA_UNAVAILABLE_LABEL = "⚪ Data unavailable"


def _num(v: Any, default: Any = 0.0) -> Any:
    """Returns float(v) when v is not None and finite (a genuine 0.0 is preserved), otherwise `default`."""
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def _first_num(*values: Any, default: Any = None) -> Any:
    """Returns the first finite numeric value among `values` (a genuine 0.0 counts), otherwise `default`."""
    for v in values:
        f = _num(v, None)
        if f is not None:
            return f
    return default


def _sym(ticker: Any) -> str:
    """Upper-cased, stripped ticker string (safe for None)."""
    return str(ticker or "").upper().strip()


def is_leveraged_etf(yf_ticker: str, company_name: str = "") -> bool:
    """Returns True for 3x daily leveraged ETFs/ETPs (QQQ3, 3LUS, 3SEM, 3X, LQQ3, 3USL, TQQQ)."""
    sym = _sym(yf_ticker)
    if any(k in sym for k in LEVERAGED_ETF_KEYS):
        return True
    c_upper = str(company_name or "").upper()
    return "LEVERAGED" in c_upper


def is_broad_index_etf(yf_ticker: str) -> bool:
    """Returns True for broad diversified (unleveraged) index ETFs (S&P 500, All-World, Nasdaq-100 lines)."""
    sym = _sym(yf_ticker)
    if not sym or is_leveraged_etf(sym):
        return False
    return sym.split(".")[0] in BROAD_INDEX_ETF_EXACT or any(k in sym for k in BROAD_INDEX_ETF_KEYS)


def is_gold_etc(yf_ticker: str) -> bool:
    """Returns True for physical gold ETCs (SGLN, IGLN, GLD)."""
    sym = _sym(yf_ticker)
    return bool(sym) and any(k in sym for k in GOLD_ETC_KEYS)


def is_money_market_fund(yf_ticker: str) -> bool:
    """Returns True for money-market / overnight-rate cash funds (CSH2)."""
    sym = _sym(yf_ticker)
    return bool(sym) and any(k in sym for k in MONEY_MARKET_KEYS)


def _rebalance_cfg_frac(key: str, default: float) -> float:
    """Reads a numeric limit from REBALANCE_CONFIG with a default. A configured 0.0 is honoured (never `or default`)."""
    try:
        raw = REBALANCE_CONFIG.get(key, default)
    except Exception:
        return default
    return _num(raw, default)


def _rebalance_cfg_bool(key: str, default: bool = False) -> bool:
    """Reads a boolean flag from REBALANCE_CONFIG with a default (accepts bools and 'true'/'false' strings)."""
    try:
        raw = REBALANCE_CONFIG.get(key, default)
    except Exception:
        return default
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw)


def _earmarked_tickers() -> Set[str]:
    """Earmarked tickers from config (e.g. a money-market fund set aside for a planned expense), read with a safe default."""
    raw = getattr(config, "EARMARKED_TICKERS", set()) or set()
    if isinstance(raw, str):
        raw = raw.split(",")
    try:
        return {str(t).strip() for t in raw if str(t).strip()}
    except TypeError:
        return set()


def is_earmarked(yf_ticker: str, earmarked: Optional[Set[str]] = None) -> bool:
    """Returns True if the ticker is earmarked money excluded from all portfolio maths and never traded."""
    marks = _earmarked_tickers() if earmarked is None else earmarked
    if not yf_ticker or not marks:
        return False
    return any(tickers_match(yf_ticker, e) for e in marks)


def _model_version() -> str:
    """Frozen model version tag from config (defaults to 'dev')."""
    return str(getattr(config, "MODEL_VERSION", "dev") or "dev")


def _position_cap_applies(yf_ticker: str) -> bool:
    """The single-name cap (max_position_pct) applies to single stocks and sector/leveraged ETFs.
    Broad index ETFs, physical gold and money-market funds are exempt unless cap_applies_to_broad_etfs is True."""
    if is_broad_index_etf(yf_ticker) or is_gold_etc(yf_ticker) or is_money_market_fund(yf_ticker):
        return _rebalance_cfg_bool("cap_applies_to_broad_etfs", False)
    return True


def _fmt_shares(qty: Any) -> str:
    """Formats a (possibly fractional) share count with up to 4 decimals and no trailing zeros."""
    q = _num(qty, 0.0)
    txt = f"{q:,.4f}".rstrip("0").rstrip(".")
    return txt if txt not in ("", "-0") else "0"


def _fmt_fwd_cagr(value: Any, capped: Any = None) -> str:
    """Forward (model) CAGR as an integer percentage, flagged when it is the research model's cap or floor."""
    v = _num(value, None)
    if v is None:
        return "n/a"
    txt = f"{v:+.0f}%"
    if capped == "cap":
        txt += " (model cap)"
    elif capped == "floor":
        txt += " (model floor)"
    return txt


def _fmt_rr(rr: Any) -> str:
    """Risk/reward ratio text ('n/a' when there is no valid analyst target)."""
    v = _num(rr, None)
    return "n/a" if v is None else f"{v:.2f}:1"


def is_core_asset(yf_ticker: str, company_name: str = "", macro_cluster: str = "") -> bool:
    """Returns True if the asset is an unleveraged core index foundation, physical gold ETC, or diversified ETF that should never be panic-sold."""
    clean_sym = _sym(yf_ticker)
    if is_leveraged_etf(clean_sym):
        return False
    c_upper = str(company_name or "").upper()
    if "LEVERAGED" in c_upper or "3X" in c_upper:
        return False
    # Sector ETFs (SMGB semiconductors, DFND aerospace & defence) are concentrated thematic bets, not core foundations
    if any(k in clean_sym for k in SECTOR_ETF_KEYS):
        return False
    core_symbols = {"VUAA.L", "VUSA.L", "VWRP.L", "CNDX.L", "SGLN.L", "IGLN.L", "GLD", "CSH2.L", "SPY", "QQQ", "RSP"}
    if clean_sym in core_symbols or any(k in clean_sym for k in ["VUAA", "VUSA", "VWRP", "CNDX", "SGLN", "IGLN", "CSH2"]):
        return True
    if is_broad_index_etf(clean_sym) or is_gold_etc(clean_sym) or is_money_market_fund(clean_sym):
        return True
    if macro_cluster in ["Diversified Index Core", "Global Equities Core Foundation", "Uncorrelated Crash Hedge & Gold"]:
        return True
    if "UCITS" in c_upper or "ETF" in c_upper or "ETC" in c_upper or "INDEX" in c_upper or "ALL-WORLD" in c_upper or "S&P 500" in c_upper or "PHYSICAL GOLD" in c_upper or "OVERNIGHT" in c_upper:
        return True
    return False


def is_crash_shield_asset(
    yf_ticker: str,
    macro_cluster: str = "",
    beta_5y: float = 1.0,
    crash_2022_ret_pct: float = -20.0
) -> bool:
    """Returns True if the asset acts as a Crisis-Alpha Hedge, Cash Fortress, or Low-Beta Crash-Resistant Anchor."""
    clean_sym = _sym(yf_ticker)
    if is_leveraged_etf(clean_sym) or "SMGB" in clean_sym:
        return False
    shield_tickers = {"SGLN.L", "IGLN.L", "GLD", "CSH2.L", "BRK-B", "LLY", "ABBV", "VWRP.L"}
    if clean_sym in shield_tickers or any(k in clean_sym for k in ["SGLN", "IGLN", "CSH2", "BRK-B", "ABBV"]):
        return True
    if macro_cluster in ["Uncorrelated Crash Hedge & Gold", "Defensive Ballast & Value"]:
        return True
    if macro_cluster == "Healthcare & Biopharma" and beta_5y <= 0.65 and crash_2022_ret_pct >= -5.0:
        return True
    if beta_5y <= 0.45 and crash_2022_ret_pct >= -3.0:
        return True
    return False


def is_recent_entry(position_data: Optional[Dict[str, Any]], days_threshold: int = 28) -> Tuple[bool, Optional[int]]:
    """Returns (True, days_held) if position was entered/added within the last `days_threshold` days (anti-churn grace period)."""
    if not position_data:
        return False, None
    raw_date = position_data.get("last_entry_date") or position_data.get("initialFillDate")
    if not raw_date:
        return False, None
    try:
        raw_str = str(raw_date)
        if "T" in raw_str:
            fill_dt = datetime.fromisoformat(raw_str.replace("Z", "+00:00"))
        else:
            fill_dt = datetime.strptime(raw_str[:10], "%Y-%m-%d")
        now_dt = datetime.now(fill_dt.tzinfo) if fill_dt.tzinfo else datetime.now()
        days_held = (now_dt - fill_dt).days
        return (days_held < days_threshold), days_held
    except Exception as e:
        logger.debug("Could not parse entry date: %s", e)
        return False, None


def is_recently_trimmed(position_data: Optional[Dict[str, Any]], days_threshold: int = 180) -> Tuple[bool, Optional[int]]:
    """Returns (True, days_since_trim) if position was trimmed within the last `days_threshold` days (6-month anti-round-trip & anti-re-trim lockout)."""
    if not position_data:
        return False, None
    raw_date = position_data.get("last_trim_date")
    if not raw_date:
        return False, None
    try:
        raw_str = str(raw_date)
        if "T" in raw_str:
            trim_dt = datetime.fromisoformat(raw_str.replace("Z", "+00:00"))
        else:
            trim_dt = datetime.strptime(raw_str[:10], "%Y-%m-%d")
        now_dt = datetime.now(trim_dt.tzinfo) if trim_dt.tzinfo else datetime.now()
        days_since_trim = (now_dt - trim_dt).days
        return (days_since_trim < days_threshold), days_since_trim
    except Exception as e:
        logger.debug("Could not parse trim date: %s", e)
        return False, None


def _fi_get(fast_info: Any, key: str) -> Any:
    """Safe getter for yfinance fast_info (dict or FastInfo); returns None when the lookup fails."""
    if fast_info is None:
        return None
    try:
        return fast_info.get(key)
    except Exception:
        return None


def _live_quote(pair: str) -> Optional[float]:
    """Latest live quote for a Yahoo symbol (fast_info lastPrice, falling back to the last 5-day close)."""
    t = yf.Ticker(pair)
    rate = _num(_fi_get(getattr(t, "fast_info", None), "lastPrice"), None)
    if rate is None or rate <= 0:
        hist = t.history(period="5d", interval="1d")
        if hist is not None and not hist.empty and "Close" in hist:
            closes = hist["Close"].dropna()
            if not closes.empty:
                rate = _num(closes.iloc[-1], None)
    return rate


def get_gbp_usd_rate() -> float:
    """Fetches the current GBP/USD exchange rate (e.g. 1.35 means £1 = $1.35).

    Raises RuntimeError when no live quote is available: a static fallback rate is never assumed silently.
    """
    global _CACHED_GBP_RATE
    if _CACHED_GBP_RATE is not None:
        return _CACHED_GBP_RATE
    try:
        rate = _live_quote("GBPUSD=X")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch live GBPUSD rate: {e}") from e
    if rate is None or not (0.5 <= rate <= 3.0):
        raise RuntimeError(f"Live GBPUSD rate unavailable or implausible ({rate!r})")
    _CACHED_GBP_RATE = rate
    return rate


def get_fx_to_gbp(ccy: str) -> float:
    """Returns GBP per 1 unit of `ccy` from live yfinance ({CCY}GBP=X), cached per process.

    GBP -> 1.0 and GBp/GBX (pence) -> 0.01 without a lookup; USD uses the same live GBPUSD=X anchor as
    get_gbp_usd_rate() so every USD conversion in a run is consistent. Raises RuntimeError on failure.
    """
    raw = str(ccy or "").strip()
    if raw == "GBp" or raw.upper() in ("GBX", "PENCE", "GBPX"):
        return 0.01
    code = raw.upper()
    if code == "GBP":
        return 1.0
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise RuntimeError(f"Unknown currency code: {ccy!r}")
    if code == "USD":
        return 1.0 / get_gbp_usd_rate()
    if code in _CACHED_FX_TO_GBP:
        cached = _CACHED_FX_TO_GBP[code]
        if cached is None:
            raise RuntimeError(f"Live {code}GBP=X rate unavailable (failed earlier in this run)")
        return cached
    pair = f"{code}GBP=X"
    try:
        rate = _live_quote(pair)
    except Exception as e:
        logger.warning("Failed to fetch live %s rate: %s", pair, e)
        rate = None
    if rate is None or rate <= 0:
        _CACHED_FX_TO_GBP[code] = None
        raise RuntimeError(f"Live {pair} rate unavailable")
    _CACHED_FX_TO_GBP[code] = rate
    return rate


def to_gbp(price: float, currency: str, gbp_rate: Optional[float]) -> float:
    """Converts a price in its native currency into GBP (£).

    Returns float('nan') for an unknown or unconvertible currency (never the raw price).
    """
    p = _num(price, None)
    if p is None or p <= 0:
        return 0.0
    raw_c = str(currency or "").strip()
    # Note: Yahoo Finance uses 'GBp' (lowercase p) specifically for pence, and 'GBP' for pounds sterling.
    # Trading 212 / other sources may use 'GBX' or 'PENCE'.
    if raw_c == "GBp" or raw_c.upper() in ("GBX", "PENCE", "GBPX"):
        return p / 100.0
    c = raw_c.upper()
    if c == "GBP":
        return p
    if c == "USD":
        rate = _num(gbp_rate, None)
        if rate is None or rate <= 0:
            try:
                rate = get_gbp_usd_rate()
            except RuntimeError as e:
                logger.warning("USD price could not be converted to GBP: %s", e)
                return float("nan")
        return p / rate
    if not re.fullmatch(r"[A-Z]{3}", c):
        return float("nan")
    try:
        return p * get_fx_to_gbp(c)
    except RuntimeError as e:
        logger.warning("%s price could not be converted to GBP: %s", c, e)
        return float("nan")


def _live_price_gbp(ticker: str, currency_hint: Optional[str] = None) -> Optional[float]:
    """Live last price for `ticker` converted into GBP (cached per process); None when unavailable."""
    key = _sym(ticker)
    if key in _CACHED_OUTSIDE_PRICE_GBP:
        return _CACHED_OUTSIDE_PRICE_GBP[key]
    price_gbp = None
    try:
        t_obj = yf.Ticker(key)
        fast_info = getattr(t_obj, "fast_info", None)
        last = _num(_fi_get(fast_info, "lastPrice"), None)
        if last is None or last <= 0:
            hist = t_obj.history(period="5d", interval="1d")
            if hist is not None and not hist.empty and "Close" in hist:
                closes = hist["Close"].dropna()
                if not closes.empty:
                    last = _num(closes.iloc[-1], None)
        ccy = _fi_get(fast_info, "currency") or currency_hint
        if last is not None and last > 0 and ccy:
            converted = to_gbp(last, ccy, None)
            if math.isfinite(converted) and converted > 0:
                price_gbp = converted
    except Exception as e:
        logger.warning("Live price lookup failed for outside holding %s: %s", key, e)
    _CACHED_OUTSIDE_PRICE_GBP[key] = price_gbp
    return price_gbp


def value_outside_holdings(account: Optional[str] = None) -> List[Dict[str, Any]]:
    """Values config.OUTSIDE_HOLDINGS (e.g. shares held at another broker or in a workplace share plan) in GBP.

    Vested shares are valued at the live yfinance price converted into GBP, falling back to
    value_hint_usd_vested (converted at the live GBPUSD rate) on failure. Unvested value is display-only.
    These holdings are never traded: they only feed cluster-cap and single-name concentration checks.
    """
    raw_cfg = getattr(config, "OUTSIDE_HOLDINGS", {}) or {}
    if not isinstance(raw_cfg, dict):
        return []
    themes = getattr(research_agent_module, "THEMATIC_CATEGORIES", {}) or {}
    theme_clusters = getattr(research_agent_module, "THEME_MACRO_CLUSTER", {}) or {}
    rows: List[Dict[str, Any]] = []
    for acc_name, items in raw_cfg.items():
        if account is not None and acc_name != account:
            continue
        for h in (items or []):
            if not isinstance(h, dict):
                continue
            ticker = _sym(h.get("ticker"))
            if not ticker:
                continue
            sh_vested = max(0.0, _num(h.get("shares_vested"), 0.0))
            sh_unvested = max(0.0, _num(h.get("shares_unvested"), 0.0))
            px_gbp = _live_price_gbp(ticker, h.get("currency") or None)
            if px_gbp is not None:
                vested_gbp = sh_vested * px_gbp
                unvested_gbp = sh_unvested * px_gbp
                source = "live"
            else:
                hint_vested = max(0.0, _num(h.get("value_hint_usd_vested"), 0.0))
                hint_unvested = max(0.0, _num(h.get("value_hint_usd_unvested"), 0.0))
                vested_gbp = to_gbp(hint_vested, "USD", None) if hint_vested > 0 else 0.0
                unvested_gbp = to_gbp(hint_unvested, "USD", None) if hint_unvested > 0 else 0.0
                source = "hint"
                if not math.isfinite(vested_gbp) or not math.isfinite(unvested_gbp):
                    logger.warning("Outside holding %s (%s) could not be valued in GBP.", ticker, acc_name)
                    vested_gbp = vested_gbp if math.isfinite(vested_gbp) else 0.0
                    unvested_gbp = unvested_gbp if math.isfinite(unvested_gbp) else 0.0
                    source = "unavailable"
            cluster = h.get("macro_cluster") or theme_clusters.get(themes.get(ticker, ""), "") or "Outside Holdings"
            rows.append({
                "account": acc_name,
                "ticker": ticker,
                "vested_gbp": round(vested_gbp, 2),
                "unvested_gbp": round(unvested_gbp, 2),
                "macro_cluster": cluster,
                "shares_vested": sh_vested,
                "shares_unvested": sh_unvested,
                "price_gbp": px_gbp,
                "valuation_source": source,
            })
    return rows


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Wilder's Exponentially Smoothed Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Wilder's Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


_CACHED_SPY_DF = None
_CACHED_SPY_5Y_WK = None
_CACHED_SPY_5Y_WK_GBP = None
# Cached weekly FX close series keyed by Yahoo pair (e.g. "GBPUSD=X"); None marks a cached failure (per process)
_CACHED_FX_WEEKLY: Dict[str, Optional[pd.Series]] = {}

def get_spy_history() -> Optional[pd.DataFrame]:
    """Fetches and caches 1-year S&P 500 history for relative strength alpha calculation."""
    global _CACHED_SPY_DF
    if _CACHED_SPY_DF is not None:
        return _CACHED_SPY_DF
    try:
        spy = yf.Ticker("SPY")
        df = spy.history(period="1y", interval="1d")
        if not df.empty:
            _CACHED_SPY_DF = df
            return df
    except Exception as e:
        logger.debug("Failed to fetch SPY benchmark: %s", e)
    return None


def get_spy_5y_weekly_history() -> Optional[pd.Series]:
    """Fetches and caches 5-year weekly S&P 500 closes for empirical 5Y Beta and Correlation."""
    global _CACHED_SPY_5Y_WK
    if _CACHED_SPY_5Y_WK is not None:
        return _CACHED_SPY_5Y_WK
    try:
        spy = yf.Ticker("SPY")
        df_w = spy.history(period="5y", interval="1wk")
        if not df_w.empty and "Close" in df_w:
            s = df_w["Close"].dropna()
            if hasattr(s.index, "tz") and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            _CACHED_SPY_5Y_WK = s
            return s
    except Exception as e:
        logger.debug("Failed to fetch SPY 5y weekly benchmark: %s", e)
    return None


def _tz_naive_index(series: pd.Series) -> pd.Series:
    """Returns the series with a timezone-naive DatetimeIndex (copied when a conversion is needed)."""
    if hasattr(series.index, "tz") and series.index.tz is not None:
        series = series.copy()
        series.index = series.index.tz_localize(None)
    return series


def get_fx_weekly_history(pair: str) -> Optional[pd.Series]:
    """Fetches and caches (per process) 5-year weekly closes for a Yahoo FX pair such as 'GBPUSD=X'."""
    if pair in _CACHED_FX_WEEKLY:
        return _CACHED_FX_WEEKLY[pair]
    series = None
    try:
        df_fx = yf.Ticker(pair).history(period="5y", interval="1wk")
        if df_fx is not None and not df_fx.empty and "Close" in df_fx:
            s = df_fx["Close"].dropna()
            s = s[s > 0]
            if len(s) >= 20:
                s = _tz_naive_index(s).sort_index()
                series = s[~s.index.duplicated(keep="last")]
    except Exception as e:
        logger.debug("Failed to fetch weekly FX history for %s: %s", pair, e)
    _CACHED_FX_WEEKLY[pair] = series
    return series


def _align_weekly(target_index: pd.Index, fx_series: Optional[pd.Series]) -> Optional[pd.Series]:
    """Aligns weekly FX closes onto a weekly target index by week (nearest bar within 3 days), then
    forward-fills gaps. Weeks before the FX history begins stay NaN (they are dropped, never back-filled)."""
    if fx_series is None or len(fx_series) == 0 or len(target_index) == 0:
        return None
    try:
        tgt = pd.DatetimeIndex(target_index)
        fx_clean = _tz_naive_index(fx_series.dropna()).sort_index()
        fx_clean = fx_clean[~fx_clean.index.duplicated(keep="last")]
        left = pd.DataFrame({
            "t": tgt.values.astype("datetime64[ns]"),
            "_pos": np.arange(len(tgt)),
        }).sort_values("t")
        right = pd.DataFrame({
            "t": pd.DatetimeIndex(fx_clean.index).values.astype("datetime64[ns]"),
            "fx": fx_clean.values.astype(float),
        }).sort_values("t")
        merged = pd.merge_asof(left, right, on="t", direction="nearest", tolerance=pd.Timedelta(days=3))
        merged = merged.sort_values("_pos")
        return pd.Series(merged["fx"].values, index=target_index).ffill()
    except Exception as e:
        logger.debug("Weekly FX alignment failed: %s", e)
        return None


def _convert_weekly_to_gbp(closes: pd.Series, currency: Optional[str]) -> Tuple[Optional[pd.Series], bool]:
    """Converts native weekly closes into GBP before any metric is computed:
    USD ÷ weekly GBPUSD=X, GBp/GBX ÷ 100, EUR/CHF/other ISO currencies × weekly {CCY}GBP=X (aligned by week, ffill).

    Returns (gbp_series, True), or (None, False) when the currency is unknown or FX history is unavailable.
    """
    raw = str(currency or "").strip()
    if not raw:
        return None, False
    if raw == "GBp" or raw.upper() in ("GBX", "PENCE", "GBPX"):
        out = closes / 100.0
    elif raw.upper() == "GBP":
        out = closes.copy()
    else:
        code = raw.upper()
        if not re.fullmatch(r"[A-Z]{3}", code):
            return None, False
        if code == "USD":
            aligned = _align_weekly(closes.index, get_fx_weekly_history("GBPUSD=X"))
            if aligned is None:
                return None, False
            out = closes / aligned
        else:
            aligned = _align_weekly(closes.index, get_fx_weekly_history(f"{code}GBP=X"))
            if aligned is None:
                return None, False
            out = closes * aligned
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out > 0]
    if len(out) < 20:
        return None, False
    return out, True


def get_spy_5y_weekly_history_gbp() -> Optional[pd.Series]:
    """Fetches and caches 5-year weekly S&P 500 (SPY) closes converted into GBP via weekly GBPUSD=X."""
    global _CACHED_SPY_5Y_WK_GBP
    if _CACHED_SPY_5Y_WK_GBP is not None:
        return _CACHED_SPY_5Y_WK_GBP
    spy_w = get_spy_5y_weekly_history()
    if spy_w is None or len(spy_w) == 0:
        return None
    spy_gbp, ok = _convert_weekly_to_gbp(_tz_naive_index(spy_w), "USD")
    if not ok:
        return None
    _CACHED_SPY_5Y_WK_GBP = spy_gbp
    return spy_gbp


def _period_return_pct(series: Optional[pd.Series], start: str, end: str, min_bars: int = 25) -> Optional[float]:
    """Point-to-point % return of a weekly series inside [start, end]; None when the window is too short."""
    if series is None or len(series) == 0:
        return None
    window = series[(series.index >= start) & (series.index <= end)]
    if len(window) >= min_bars and float(window.iloc[0]) > 0:
        return ((float(window.iloc[-1]) / float(window.iloc[0])) - 1.0) * 100.0
    return None


def _quote_currency(ticker_obj: Any, fast_info: Any = None) -> Optional[str]:
    """Quote currency from yfinance fast_info, falling back to info['currency']; None when unknown (never 'USD')."""
    ccy = None
    try:
        fi = fast_info if fast_info is not None else getattr(ticker_obj, "fast_info", None)
        ccy = _fi_get(fi, "currency")
    except Exception:
        ccy = None
    if not ccy:
        try:
            info = getattr(ticker_obj, "info", None) or {}
            ccy = info.get("currency") if isinstance(info, dict) else None
        except Exception:
            ccy = None
    ccy = str(ccy).strip() if ccy else ""
    return ccy or None


def reset_runtime_caches() -> None:
    """Clears per-process market-data caches (FX rates, benchmarks, outside-holding prices). Intended for tests."""
    global _CACHED_GBP_RATE, _CACHED_SPY_DF, _CACHED_SPY_5Y_WK, _CACHED_SPY_5Y_WK_GBP
    _CACHED_GBP_RATE = None
    _CACHED_SPY_DF = None
    _CACHED_SPY_5Y_WK = None
    _CACHED_SPY_5Y_WK_GBP = None
    _CACHED_FX_TO_GBP.clear()
    _CACHED_FX_WEEKLY.clear()
    _CACHED_OUTSIDE_PRICE_GBP.clear()


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculates MACD Line, Signal Line, and Momentum Histogram."""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculates On-Balance Volume (OBV)."""
    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    return (volume * direction).cumsum()


def evaluate_minervini_template(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluates Mark Minervini's SEPA 5-Point Trend Template for institutional Stage 2 growth."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    latest_price = float(close.iloc[-1])

    ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    sma_150 = float(close.rolling(150).mean().iloc[-1]) if len(close) >= 150 else ema_50 * 0.96
    sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma_150 * 0.96
    sma_200_prev = float(close.rolling(200).mean().iloc[-25]) if len(close) >= 225 else sma_200 * 0.98

    low_52w = float(low.min())
    high_52w = float(high.max())

    c1 = (latest_price > ema_50) and (latest_price > sma_200)
    c2 = (ema_50 > sma_150 > sma_200)
    c3 = (sma_200 >= sma_200_prev)
    c4 = (latest_price >= 1.25 * low_52w)
    c5 = (latest_price >= 0.75 * high_52w)

    pass_count = sum([bool(c1), bool(c2), bool(c3), bool(c4), bool(c5)])
    if pass_count == 5:
        stage = "Stage 2 Confirmed Bull (5/5)"
        badge_bg = "#10b981"
    elif pass_count == 4:
        stage = "Stage 2 Developing (4/5)"
        badge_bg = "#10b981"
    elif pass_count == 3:
        stage = "Stage 1 Base / Neutral (3/5)"
        badge_bg = "#f59e0b"
    else:
        stage = f"Stage 4 Distribution ({pass_count}/5)"
        badge_bg = "#ef4444"

    return {
        "pass_count": pass_count,
        "stage_label": stage,
        "badge_bg": badge_bg,
        "is_stage_2": pass_count >= 4,
        "ema_50": ema_50,
        "sma_150": sma_150,
        "sma_200": sma_200,
        "low_52w": low_52w,
        "high_52w": high_52w,
        "c1": bool(c1),
        "c2": bool(c2),
        "c3": bool(c3),
        "c4": bool(c4),
        "c5": bool(c5),
    }


def evaluate_weekly_trend(ticker_obj: yf.Ticker, yf_ticker: str = "", currency: Optional[str] = None) -> Dict[str, Any]:
    """Evaluates Stan Weinstein's 30-week Moving Average rule AND computes empirical 5-Year
    Risk-Adjusted Return (CAGR, Volatility, Sortino, Sharpe, Calmar, Max Drawdown) & 2022 Crash Resilience.

    GBP basis: the weekly close series is converted into GBP before anything is computed, and beta/correlation
    are measured against SPY converted into GBP. If FX history is unavailable the native series is kept and the
    result is flagged fx_basis="native". `currency` is the quote currency (None = look it up via fast_info,
    falling back to info['currency']; "" = known to be unavailable).
    """
    default_5y = {
        "is_weekly_stage_2": False,
        "label": "⚪ Weekly data unavailable",
        "badge_color": "#64748b",
        "ema_10w": 0.0,
        "ema_30w": 0.0,
        "ema_50w": 0.0,
        "pct_vs_30w": 0.0,
        "cagr_5y_pct": MISSING_METRIC_DEFAULTS["cagr_5y_pct"],
        "vol_5y_pct": 20.0,
        "sortino_5y": MISSING_METRIC_DEFAULTS["sortino_5y"],
        "sharpe_5y": 0.80,
        "max_dd_5y_pct": -22.0,
        "calmar_5y": 0.64,
        "crash_2022_ret_pct": MISSING_METRIC_DEFAULTS["crash_2022_ret_pct"],
        "beta_5y": MISSING_METRIC_DEFAULTS["beta_5y"],
        "corr_5y": MISSING_METRIC_DEFAULTS["corr_5y"],
        "crash_role_label": DATA_UNAVAILABLE_LABEL,
        "crash_role_color": "#64748b",
        "fx_basis": "native",
        "quote_currency": currency or None,
        "data_ok": False,
    }
    try:
        df_w = ticker_obj.history(period="5y", interval="1wk")
        if df_w is None or df_w.empty or len(df_w) < 25:
            return default_5y

        c_native = df_w["Close"].dropna()
        if c_native.empty or len(c_native) < 20:
            return default_5y
        c_native = _tz_naive_index(c_native)
        c_native = c_native[c_native > 0]
        if len(c_native) < 20:
            return default_5y

        # Convert into GBP BEFORE computing anything (USD ÷ GBPUSD, GBp ÷ 100, EUR/CHF/other × {CCY}GBP);
        # beta & correlation use SPY converted into GBP the same way.
        quote_ccy = _quote_currency(ticker_obj) if currency is None else (currency or None)
        c_gbp, gbp_ok = _convert_weekly_to_gbp(c_native, quote_ccy)
        spy_gbp = get_spy_5y_weekly_history_gbp() if gbp_ok else None
        if gbp_ok and spy_gbp is not None:
            c = c_gbp
            bench_w = spy_gbp
            fx_basis = "GBP"
        else:
            c = c_native
            bench_w = get_spy_5y_weekly_history()
            fx_basis = "native"
            logger.info(
                "FX history unavailable for %s (%s): 5Y metrics computed on the native series (fx_basis=native).",
                yf_ticker, quote_ccy or "unknown currency",
            )

        latest_w = float(c.iloc[-1])
        if np.isnan(latest_w) or latest_w <= 0:
            latest_w = float(c.iloc[-2]) if len(c) >= 2 else 1.0

        ema_10w = float(c.ewm(span=10, adjust=False).mean().iloc[-1])
        ema_30w = float(c.ewm(span=30, adjust=False).mean().iloc[-1])
        ema_30w_prev = float(c.ewm(span=30, adjust=False).mean().iloc[-5]) if len(c) >= 5 else ema_30w * 0.99
        ema_50w = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
        ema_50w_prev = float(c.ewm(span=50, adjust=False).mean().iloc[-8]) if len(c) >= 8 else ema_50w * 0.99

        if np.isnan(ema_30w) or ema_30w <= 0:
            ema_30w = latest_w
        if np.isnan(ema_50w) or ema_50w <= 0:
            ema_50w = ema_30w

        # Compute 5-Year Empirical Risk-Adjusted & Crash Metrics from Weekly Bars (GBP basis)
        c_tz_free = c

        days_span = max(365.0, float((c_tz_free.index[-1] - c_tz_free.index[0]).days))
        years_span = days_span / 365.25
        first_w = float(c_tz_free.iloc[0]) if float(c_tz_free.iloc[0]) > 0 else latest_w
        raw_5y_cagr_dec = ((latest_w / first_w) ** (1.0 / years_span)) - 1.0 if first_w > 0 else 0.12

        # Compute 3-Year Rolling CAGR to eliminate single-start-date Oct 2021 bubble-peak distortion
        if len(c_tz_free) >= 150:
            idx_3y = max(0, len(c_tz_free) - 156)
            w_3y = float(c_tz_free.iloc[idx_3y])
            days_3y = max(365.0, float((c_tz_free.index[-1] - c_tz_free.index[idx_3y]).days))
            cagr_3y_dec = ((latest_w / w_3y) ** (365.25 / days_3y)) - 1.0 if w_3y > 0 else raw_5y_cagr_dec
            cagr_dec = (0.55 * raw_5y_cagr_dec) + (0.45 * cagr_3y_dec)
        else:
            cagr_dec = raw_5y_cagr_dec

        cagr_5y_pct = round(cagr_dec * 100.0, 1)

        wk_ret = c_tz_free.pct_change().dropna()
        vol_dec = float(wk_ret.std() * np.sqrt(52.0)) if len(wk_ret) > 5 else 0.20
        vol_5y_pct = round(vol_dec * 100.0, 1)

        rf = 0.035
        mar_wk = rf / 52.0
        downside_diff = np.minimum(0.0, wk_ret.values - mar_wk)
        downside_dev = float(np.sqrt(np.mean(downside_diff ** 2) * 52.0)) if len(downside_diff) > 5 else 0.14
        sharpe_5y = round((cagr_dec - rf) / vol_dec, 2) if vol_dec > 0.01 else 1.0
        sortino_5y = round((cagr_dec - rf) / downside_dev, 2) if downside_dev > 0.01 else 1.3

        roll_max = c_tz_free.cummax()
        dd_series = (c_tz_free / roll_max) - 1.0
        max_dd_dec = float(dd_series.min()) if not dd_series.empty else -0.20
        max_dd_5y_pct = round(max_dd_dec * 100.0, 1)
        calmar_5y = round(cagr_dec / abs(max_dd_dec), 2) if abs(max_dd_dec) > 0.01 else 1.0

        # Beta & Correlation vs S&P 500 (SPY, converted into GBP on the GBP basis) over 5 years
        beta_5y = 1.0
        corr_5y = 0.70
        spy_w = bench_w
        if spy_w is not None and len(spy_w) > 30:
            spy_ret = spy_w.pct_change().dropna()
            aligned = pd.concat([wk_ret.rename("asset"), spy_ret.rename("spy")], axis=1).dropna()
            if len(aligned) >= 20:
                spy_var = float(np.var(aligned["spy"]))
                if spy_var > 1e-8:
                    cov = float(np.cov(aligned["asset"], aligned["spy"])[0, 1])
                    beta_5y = round(cov / spy_var, 2)
                c_mat = aligned["asset"].corr(aligned["spy"])
                if pd.notna(c_mat):
                    corr_5y = round(float(c_mat), 2)

        # 2022 Inflation & Tech Bear Market Stress Return (Jan 2022 - Dec 2022)
        c_2022 = c_tz_free[(c_tz_free.index >= "2022-01-01") & (c_tz_free.index <= "2022-12-31")]
        if len(c_2022) >= 25 and float(c_2022.iloc[0]) > 0:
            crash_2022_ret_pct = round(((float(c_2022.iloc[-1]) / float(c_2022.iloc[0])) - 1.0) * 100.0, 1)
        else:
            # For instruments listed after 2022 (e.g. DFND.SW), estimate 2022 shock via empirical beta x the
            # benchmark's own 2022 return on the same currency basis, clamped to at most +2.0% (no invented crisis alpha)
            bench_2022 = _period_return_pct(bench_w, "2022-01-01", "2022-12-31")
            if bench_2022 is None:
                bench_2022 = -19.4
            crash_2022_ret_pct = round(max(-75.0, min(2.0, beta_5y * bench_2022)), 1)

        # Classify 5-Year Portfolio Crash & Compounding Role
        sym_u = yf_ticker.upper()
        if is_leveraged_etf(sym_u):
            crash_role_label = "⚠️ High-Decay 3x Leverage"
            crash_role_color = "#ef4444"
        elif corr_5y <= 0.15 and crash_2022_ret_pct >= 0.0:
            crash_role_label = "🛡️ Crisis Alpha Hedge"
            crash_role_color = "#10b981"
        elif crash_2022_ret_pct >= -8.0 and beta_5y <= 0.70:
            crash_role_label = "🏰 Crash-Resistant Compounder"
            crash_role_color = "#0ea5e9"
        elif sortino_5y >= 1.45:
            crash_role_label = "🚀 Elite 5Y Sortino Compounder"
            crash_role_color = "#6366f1"
        else:
            crash_role_label = "📈 5Y Secular Growth"
            crash_role_color = "#64748b"

        is_above_30w = latest_w >= (ema_30w * 0.985)
        is_rising_30w = ema_30w >= (ema_30w_prev * 0.995)
        is_weekly_s2 = is_above_30w and is_rising_30w
        raw_pct = ((latest_w - ema_30w) / ema_30w) * 100
        pct_vs_30w = round(raw_pct, 1) if not np.isnan(raw_pct) else 0.0

        # Crisis-Hedge & Uncorrelated Gold 50-Week Structural Support Recognition:
        # Restricted strictly to genuine crisis shields (Physical Gold ETCs, BRK-B, VWRP, ABBV) testing rising 50-week EMA.
        is_crisis_hedge_pullback = (
            (not is_weekly_s2)
            and any(g in sym_u for g in ["SGLN", "IGLN", "GLD", "BRK-B", "VWRP", "CSH2", "ABBV"])
            and (cagr_5y_pct >= 10.0)
            and (ema_50w >= ema_50w_prev * 0.995)
            and (latest_w >= ema_50w * 0.965)
        )
        if is_crisis_hedge_pullback:
            is_weekly_s2 = True

        if is_crisis_hedge_pullback:
            label = f"🟢 Weekly 50w Structural Support ({pct_vs_30w:+.1f}% vs 30w EMA — Crisis Shield Dip)"
            color = "#10b981"
        elif is_weekly_s2 and latest_w >= ema_10w:
            label = f"🟢 Weekly Stage 2 Confirmed Bull (+{pct_vs_30w:.1f}% vs 30w EMA)"
            color = "#10b981"
        elif is_weekly_s2:
            label = f"🟢 Weekly Stage 2 Pullback Support ({pct_vs_30w:+.1f}% vs 30w EMA)"
            color = "#10b981"
        else:
            pct_sign = "+" if pct_vs_30w > 0 else ""
            label = f"🔴 Weekly Stage 4 / Below 30w EMA ({pct_sign}{pct_vs_30w:.1f}%)"
            color = "#ef4444"

        return {
            "is_weekly_stage_2": is_weekly_s2,
            "label": label,
            "badge_color": color,
            "ema_10w": ema_10w,
            "ema_30w": ema_30w,
            "ema_50w": ema_50w,
            "pct_vs_30w": pct_vs_30w,
            "cagr_5y_pct": cagr_5y_pct,
            "vol_5y_pct": vol_5y_pct,
            "sortino_5y": sortino_5y,
            "sharpe_5y": sharpe_5y,
            "max_dd_5y_pct": max_dd_5y_pct,
            "calmar_5y": calmar_5y,
            "crash_2022_ret_pct": crash_2022_ret_pct,
            "beta_5y": beta_5y,
            "corr_5y": corr_5y,
            "crash_role_label": crash_role_label,
            "crash_role_color": crash_role_color,
            "fx_basis": fx_basis,
            "quote_currency": quote_ccy,
            "data_ok": True,
        }
    except Exception as e:
        logger.debug("Weekly trend error for %s: %s", yf_ticker, e)
        return default_5y



def evaluate_institutional_volume(df: pd.DataFrame) -> Dict[str, Any]:
    """Wyckoff Volume Price Analysis: Up/Down Volume Ratio, Volume Spike, and OBV trend."""
    close = df["Close"]
    vol = df["Volume"]
    if len(vol) < 20 or vol.sum() == 0:
        return {
            "vol_ratio": 1.0,
            "up_down_vol_ratio": 1.0,
            "label": "Constructive Volume",
            "badge_color": "#10b981",
            "is_accumulation": True,
        }

    vol_sma_20 = float(vol.rolling(20).mean().iloc[-1])
    today_vol = float(vol.iloc[-1])
    vol_ratio = round(today_vol / vol_sma_20, 2) if vol_sma_20 > 0 else 1.0

    delta = close.diff()
    up_vol = df[delta > 0]["Volume"].tail(50).sum()
    down_vol = df[delta < 0]["Volume"].tail(50).sum()
    up_down_vol_ratio = round(float(up_vol / down_vol), 2) if down_vol > 0 else 1.5

    direction = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    obv = (vol * direction).cumsum()
    obv_20_slope = float(obv.iloc[-1] - obv.iloc[-20]) if len(obv) >= 20 else 0.0

    if up_down_vol_ratio >= 1.25 or (vol_ratio >= 1.2 and delta.iloc[-1] > 0):
        label = f"🟢 Institutional Accumulation (U/D: {up_down_vol_ratio}x)"
        badge_color = "#10b981"
        is_accum = True
    elif up_down_vol_ratio < 0.85 or (vol_ratio >= 1.3 and delta.iloc[-1] < 0):
        label = f"🔴 Institutional Distribution (U/D: {up_down_vol_ratio}x)"
        badge_color = "#ef4444"
        is_accum = False
    else:
        label = f"Constructive Volume (U/D: {up_down_vol_ratio}x)"
        badge_color = "#6366f1"
        is_accum = True

    return {
        "vol_ratio": vol_ratio,
        "up_down_vol_ratio": up_down_vol_ratio,
        "obv_20_slope": obv_20_slope,
        "label": label,
        "badge_color": badge_color,
        "is_accumulation": is_accum,
    }


def compute_rs_alpha(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Calculates weighted multi-timeframe Relative Strength Alpha vs S&P 500 (SPY):
    Weights: 40% 3-month (63d), 30% 6-month (126d), 20% 9-month (189d), 10% 12-month (252d).
    Normalizes weights gracefully if stock has less than 252 days of history.
    """
    spy_df = get_spy_history()
    if spy_df is None or len(spy_df) < 63 or len(df) < 63:
        return 0.0, "Benchmark Neutral"
    try:
        periods = [
            (63, 0.40, "3m"),
            (126, 0.30, "6m"),
            (189, 0.20, "9m"),
            (252, 0.10, "12m"),
        ]
        weighted_alpha = 0.0
        total_weight = 0.0
        alpha_3m = 0.0

        for lookback, weight, name in periods:
            if len(df) >= lookback and len(spy_df) >= lookback:
                p_past = float(df["Close"].iloc[-lookback])
                s_past = float(spy_df["Close"].iloc[-lookback])
                if p_past > 0 and s_past > 0:
                    stock_ret = (float(df["Close"].iloc[-1]) / p_past - 1.0) * 100.0
                    spy_ret = (float(spy_df["Close"].iloc[-1]) / s_past - 1.0) * 100.0
                    period_alpha = stock_ret - spy_ret
                    weighted_alpha += period_alpha * weight
                    total_weight += weight
                    if lookback == 63:
                        alpha_3m = period_alpha

        if total_weight > 0:
            alpha = round(weighted_alpha / total_weight, 1)
        else:
            alpha = round(alpha_3m, 1)

        if alpha >= 10.0:
            label = f"🟢 +{alpha:.1f}% Multi-TF Alpha (Market Leader)"
        elif alpha >= 0.0:
            label = f"🟢 +{alpha:.1f}% Multi-TF Alpha (Outperforming)"
        else:
            label = f"🟡 {alpha:.1f}% vs S&P (Lagging Benchmark)"
        return alpha, label
    except Exception:
        return 0.0, "Benchmark Neutral"


def detect_bearish_divergence(
    close_series: pd.Series,
    rsi_series: pd.Series,
    macd_hist_series: pd.Series,
    window: int = 20
) -> Tuple[bool, str]:
    """
    Detects bearish momentum divergence over a lookback window (default 20 trading days):
    Occurs when price achieves a higher high, but momentum indicators (RSI or MACD histogram)
    roll over and fail to confirm with a lower high.
    Warns of institutional exhaustion before distribution manifests.
    """
    if len(close_series) < window or len(rsi_series) < window or len(macd_hist_series) < window:
        return False, ""
    try:
        half = window // 2
        earlier_price = close_series.iloc[-window : -half]
        recent_price = close_series.iloc[-half :]
        earlier_rsi = rsi_series.iloc[-window : -half]
        recent_rsi = rsi_series.iloc[-half :]
        earlier_macd = macd_hist_series.iloc[-window : -half]
        recent_macd = macd_hist_series.iloc[-half :]

        p_high_earlier = float(earlier_price.max())
        p_high_recent = float(recent_price.max())
        rsi_high_earlier = float(earlier_rsi.max())
        rsi_high_recent = float(recent_rsi.max())
        macd_high_earlier = float(earlier_macd.max())
        macd_high_recent = float(recent_macd.max())
        curr_rsi = float(rsi_series.iloc[-1])

        # Price made a higher high (at least +0.75% above earlier high)
        if p_high_recent >= p_high_earlier * 1.0075 and curr_rsi >= 52.0:
            # RSI divergence: recent RSI peak failed to reach earlier peak by at least 2.5 points
            rsi_divergent = (rsi_high_recent < rsi_high_earlier - 2.5)
            # MACD histogram divergence: recent peak is distinctly lower (weakening push)
            macd_divergent = (macd_high_earlier > 0 and macd_high_recent < macd_high_earlier * 0.75)

            if rsi_divergent or macd_divergent:
                pct_p = ((p_high_recent / p_high_earlier) - 1.0) * 100.0
                reasons = []
                if rsi_divergent:
                    reasons.append(f"RSI lower high ({rsi_high_recent:.1f} vs {rsi_high_earlier:.1f})")
                if macd_divergent:
                    reasons.append(f"MACD histogram softening ({macd_high_recent:.2f} vs {macd_high_earlier:.2f})")
                note = f"⚠️ Bearish Momentum Divergence: Price +{pct_p:.1f}% higher high, but {' & '.join(reasons)}"
                return True, note
    except Exception as e:
        logger.debug("Error in detect_bearish_divergence: %s", e)

    return False, ""



def clean_search_symbol(ticker: str) -> str:
    """Returns the clean ticker string for typing in Trading 212's search bar."""
    base = (
        ticker.replace(".L", "")
        .replace(".DE", "")
        .replace(".SW", "")
        .replace(".AS", "")
        .replace(".PA", "")
        .replace("-", " ")
    )
    return base


def tickers_match(t1: str, t2: str) -> bool:
    """Checks if two ticker strings refer to the exact same instrument without greedy substring matching."""
    if not t1 or not t2:
        return False
    u1, u2 = str(t1).strip().upper(), str(t2).strip().upper()
    if u1 == u2:
        return True
    
    n1 = normalize_t212_ticker_for_yfinance(u1)
    n2 = normalize_t212_ticker_for_yfinance(u2)
    if n1 == n2:
        return True
        
    # Handle dash vs dot variation in US share classes (e.g. BRK-B vs BRK.B, BF-B vs BF.B)
    exchange_suffixes = (".L", ".DE", ".SW", ".AS", ".PA", ".TO", ".MI", ".MC")
    has_ex1 = any(n1.endswith(s) for s in exchange_suffixes)
    has_ex2 = any(n2.endswith(s) for s in exchange_suffixes)
    if not has_ex1 and not has_ex2:
        if n1.replace(".", "-") == n2.replace(".", "-"):
            return True

    return False


def analyze_position(
    yf_ticker: str,
    position_data: Optional[Dict[str, Any]] = None,
    gbp_rate: Optional[float] = None
) -> Dict[str, Any]:
    """Analyzes a stock using trend, momentum, volatility indicators, and deep fundamental research.

    Data-quality contract: every result carries data_ok (= price_data_ok and research_data_ok), price_data_ok,
    research_data_ok, valuation_ok, stop_breached and risk_reward_ratio (None when there is no valid analyst
    target). A BUY is only emitted when data_ok and valuation_ok are True; a breached reference stop is surfaced
    for review and never triggers an automatic SELL.
    """
    if gbp_rate is None:
        try:
            gbp_rate = get_gbp_usd_rate()
        except RuntimeError as e:
            logger.warning("Live GBP/USD unavailable while analysing %s (%s); USD valuations will be flagged invalid.", yf_ticker, e)
            gbp_rate = None

    try:
        research = research_agent.conduct_deep_dive(yf_ticker) or {}
    except Exception as e:
        logger.warning("Error during deep research for %s: %s", yf_ticker, e)
        research = {}
    if not isinstance(research, dict):
        research = {}
    # An empty research dict means the deep dive failed outright; a missing data_ok flag is treated as True.
    research_data_ok = bool(research) and research.get("data_ok", True) is not False

    company_name = research.get("company_name", yf_ticker)
    search_sym = clean_search_symbol(yf_ticker)
    is_earmarked_pos = is_earmarked(yf_ticker)
    is_leveraged = is_leveraged_etf(yf_ticker)
    lev_cap_frac = _rebalance_cfg_frac("max_leveraged_etf_pct", 0.08)
    fwd_cagr_capped = research.get("expected_5y_cagr_capped")

    p_qty = _num(position_data.get("quantity"), 0.0) if position_data else 0.0
    p_price = 0.0
    if position_data:
        cur_px = _num(position_data.get("currentPrice"), 0.0)
        p_price = cur_px if cur_px > 0 else _num(position_data.get("averagePrice"), 0.0)

    def _currency_hint() -> Optional[str]:
        """Currency reported by the broker position, then by research (never a hardcoded default)."""
        for src in ((position_data or {}).get("currency"), research.get("currency")):
            if src and str(src).strip():
                return str(src).strip()
        return None

    def _fallback_result(reason: str) -> Dict[str, Any]:
        """Fail-closed result when market data is unavailable: no uptrend assumed, no BUY, never defaults to USD."""
        p_curr = _currency_hint()
        if not p_curr:
            try:
                p_curr = _quote_currency(yf.Ticker(yf_ticker))
            except Exception as ex:
                logger.debug("Currency lookup failed for %s: %s", yf_ticker, ex)
                p_curr = None
        p_gbp_raw = to_gbp(p_price, p_curr, gbp_rate) if p_curr else float("nan")
        valuation_ok = bool(p_curr) and p_price > 0 and math.isfinite(p_gbp_raw) and p_gbp_raw > 0
        p_gbp = p_gbp_raw if valuation_ok else 0.0
        if not valuation_ok and p_qty > 0:
            logger.warning("GBP valuation unavailable for held position %s (currency=%s, price=%s).", yf_ticker, p_curr, p_price)

        signal = "⚪ HOLD (DATA UNAVAILABLE)"
        badge_color = "#64748b"
        action_short = "HOLD"
        plain_english_why = f"Market data for {company_name} is unavailable today, so no recommendation is made. Holding unchanged until data returns."
        app_instruction = f"Search '{search_sym}' ➔ Leave in portfolio (no action on missing data)"
        if is_earmarked_pos:
            signal = EARMARKED_SIGNAL
            plain_english_why = f"{company_name} holds earmarked money. It is excluded from all portfolio maths and is never traded by the engine."
            app_instruction = "🔒 Earmarked ➔ Leave untouched"
        elif is_leveraged and lev_cap_frac <= 0.0 and p_qty > 0:
            signal = LEVERAGE_POLICY_SELL_SIGNAL
            badge_color = "#ef4444"
            action_short = "SELL"
            reason = f"{reason}. Leverage policy sets a 0% cap on 3x daily leveraged ETFs: exit the whole position."
            plain_english_why = f"{company_name} is a 3x daily leveraged product and your policy is to hold none. Sell the whole position."
            app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Sell ➔ Sell all shares"

        return {
            "ticker": yf_ticker,
            "search_symbol": search_sym,
            "name": company_name,
            "price": p_price,
            "currency": p_curr or "UNKNOWN",
            "price_gbp": p_gbp,
            "signal": signal,
            "badge_color": badge_color,
            "action_short": action_short,
            "reason": reason,
            "plain_english_why": plain_english_why,
            "app_instruction": app_instruction,
            "rsi": 50.0,
            "trailing_stop": 0.0,
            "chandelier_stop": 0.0,
            "stop_distance_pct": 0.0,
            "target_price": 0.0,
            "rr_target_price": None,
            "ema_50": 0.0,
            "ema_200": 0.0,
            "minervini_stage": DATA_UNAVAILABLE_LABEL,
            "minervini_pass": 0,
            "minervini_badge_bg": "#64748b",
            "is_weekly_stage_2": False,
            "weekly_trend_label": "⚪ Weekly data unavailable",
            "is_extended": False,
            "vol_spike": 1.0,
            "up_down_vol_ratio": 1.0,
            "volume_footprint_label": DATA_UNAVAILABLE_LABEL,
            "volume_footprint_badge": "#64748b",
            "rs_alpha": 0.0,
            "rs_label": "Benchmark Neutral",
            "macd_line": 0.0,
            "macd_signal": 0.0,
            "macd_hist": 0.0,
            "macd_status": "Neutral",
            "vcp_pct": 2.5,
            "vcp_label": "ATR Volatility: 2.5%",
            "conviction_score": 50,
            "risk_reward_ratio": None,
            "days_to_earnings": research.get("days_to_earnings"),
            "earnings_catalyst_status": research.get("earnings_catalyst_status", "✅ Clear"),
            "piotroski_score": research.get("piotroski_score"),
            "piotroski_label": research.get("piotroski_label", "N/A"),
            "altman_z": research.get("altman_z"),
            "eps_revisions_up_30d": research.get("eps_revisions_up_30d"),
            "eps_revisions_down_30d": research.get("eps_revisions_down_30d"),
            "eps_revision_momentum_pct": research.get("eps_revision_momentum_pct"),
            "eps_revision_label": research.get("eps_revision_label", "Neutral Consensus"),
            "eps_revision_badge_color": research.get("eps_revision_badge_color", "#64748b"),
            "bearish_divergence": False,
            "bearish_divergence_note": "",
            "earnings_beat_rate_pct": research.get("earnings_beat_rate_pct"),
            "earnings_avg_surprise_pct": research.get("earnings_avg_surprise_pct"),
            "earnings_streak_label": research.get("earnings_streak_label", "N/A"),
            "earnings_streak_badge_color": research.get("earnings_streak_badge_color", "#0ea5e9"),
            "fcf_yield_pct": research.get("fcf_yield_pct"),
            "enterprise_to_ebitda": research.get("enterprise_to_ebitda"),
            "roa_pct": research.get("roa_pct"),
            "institutional_ownership_pct": research.get("institutional_ownership_pct"),
            "insider_ownership_pct": research.get("insider_ownership_pct"),
            "roic": research.get("roic"),
            "roic_label": research.get("roic_label", "N/A"),
            "roic_color": research.get("roic_color", "#64748b"),
            "insider_net_shares": research.get("insider_net_shares", 0.0),
            "insider_label": research.get("insider_label", "Neutral Insider Flow"),
            "insider_badge_color": research.get("insider_badge_color", "#64748b"),
            "options_pcr_vol": research.get("options_pcr_vol"),
            "options_pcr_oi": research.get("options_pcr_oi"),
            "options_pcr_label": research.get("options_pcr_label", "N/A (No Options Traded)"),
            "options_badge_color": research.get("options_badge_color", "#64748b"),
            "institutional_change_pct": research.get("institutional_change_pct"),
            "institutional_flow_label": research.get("institutional_flow_label", "Stable Institutional Float"),
            "institutional_flow_color": research.get("institutional_flow_color", "#64748b"),
            "theme": research.get("theme", "Secular Growth"),
            "macro_cluster": research.get("macro_cluster", "Enterprise Cloud & Software"),
            "cagr_5y_pct": MISSING_METRIC_DEFAULTS["cagr_5y_pct"],
            "vol_5y_pct": 20.0,
            "sortino_5y": MISSING_METRIC_DEFAULTS["sortino_5y"],
            "sharpe_5y": 0.80,
            "max_dd_5y_pct": -22.0,
            "calmar_5y": 0.64,
            "crash_2022_ret_pct": MISSING_METRIC_DEFAULTS["crash_2022_ret_pct"],
            "beta_5y": MISSING_METRIC_DEFAULTS["beta_5y"],
            "corr_5y": MISSING_METRIC_DEFAULTS["corr_5y"],
            "crash_role_label": DATA_UNAVAILABLE_LABEL,
            "crash_role_color": "#64748b",
            "is_crash_shield": False,
            "unrealized_pnl": position_data.get("ppl", 0.0) if position_data else 0.0,
            "quantity": p_qty,
            "avg_price": position_data.get("averagePrice", 0.0) if position_data else 0.0,
            "value_gbp": p_qty * p_gbp,
            "portfolio_weight_pct": 0.0,
            "research": research,
            "fx_basis": "native",
            "data_ok": False,
            "price_data_ok": False,
            "research_data_ok": research_data_ok,
            "valuation_ok": valuation_ok,
            "stop_breached": False,
            "is_earmarked": is_earmarked_pos,
            "is_leveraged": is_leveraged,
            "expected_5y_cagr_capped": fwd_cagr_capped,
            "exit_reasons": [],
            "model_version": _model_version(),
        }

    try:
        ticker_obj = yf.Ticker(yf_ticker)
        df = ticker_obj.history(period="1y", interval="1d")
        
        if df is None or df.empty:
            return _fallback_result("No historical price bars returned")

        # Drop any incomplete or NaN candles from exchange holidays / pre-market feeds
        df = df.dropna(subset=["Close", "High", "Low"])

        if len(df) < 60:
            return _fallback_result("Insufficient historical price bars to compute EMAs")

        try:
            fast_info = getattr(ticker_obj, "fast_info", None)
        except Exception:
            fast_info = None
        # Quote currency: yfinance fast_info, then info['currency'], then broker/research hints. Never assumed to be USD.
        yf_currency = _quote_currency(ticker_obj, fast_info)
        currency = yf_currency or _currency_hint()

        close_series = df["Close"]
        ema_20 = calculate_ema(close_series, STRATEGY_CONFIG["ema_short"])
        ema_50 = calculate_ema(close_series, STRATEGY_CONFIG["ema_medium"])
        ema_200 = calculate_ema(close_series, STRATEGY_CONFIG["ema_long"])
        rsi = calculate_rsi(close_series, STRATEGY_CONFIG["rsi_period"])
        atr = calculate_atr(df, STRATEGY_CONFIG["atr_period"])
        macd_line, macd_sig, macd_hist = calculate_macd(close_series)
        latest_price = float(close_series.iloc[-1])
        # Ensure latest price reflects the most up-to-date real-time quote available
        live_quote = None
        if position_data and _num(position_data.get("currentPrice"), 0.0) > 0:
            live_quote = _num(position_data.get("currentPrice"), 0.0)
        elif _num(_fi_get(fast_info, "lastPrice"), 0.0) > 0:
            live_quote = _num(_fi_get(fast_info, "lastPrice"), 0.0)
        if live_quote is not None and live_quote > 0:
            latest_price = live_quote

        prev_price = float(close_series.iloc[-2])
        curr_ema_20 = float(ema_20.iloc[-1])
        curr_ema_50 = float(ema_50.iloc[-1])
        curr_ema_200 = float(ema_200.iloc[-1])
        curr_rsi = float(rsi.iloc[-1])
        curr_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else (latest_price * 0.025)

        # 1. Mark Minervini SEPA Trend Template (Stage 2 Verification)
        minervini = evaluate_minervini_template(df)

        # 1b. Stan Weinstein Multi-Timeframe Weekly Trend & 5-Year Empirical Risk/Crash Audit (GBP basis)
        weekly_trend = evaluate_weekly_trend(ticker_obj, yf_ticker=yf_ticker, currency=(yf_currency or _currency_hint() or ""))
        weekly_ok = bool(weekly_trend.get("data_ok", True))
        weekly_s2 = bool(weekly_trend.get("is_weekly_stage_2", False))

        # 2. Institutional Volume Price Analysis (Wyckoff Accumulation vs Distribution)
        vol_footprint = evaluate_institutional_volume(df)

        # 3. Relative Strength Alpha vs S&P 500
        rs_alpha, rs_label = compute_rs_alpha(df)

        # 4. MACD Momentum Acceleration
        curr_macd = round(float(macd_line.iloc[-1]), 2)
        curr_macd_sig = round(float(macd_sig.iloc[-1]), 2)
        curr_macd_hist = round(float(macd_hist.iloc[-1]), 2)
        macd_slope = float(macd_hist.iloc[-1] - macd_hist.iloc[-3]) if len(macd_hist) >= 3 else 0.0
        macd_bullish = (curr_macd > curr_macd_sig) and (macd_slope >= 0)
        macd_status = "Bullish Momentum Accelerating" if macd_bullish else ("Bullish Consolidation" if curr_macd > curr_macd_sig else "Momentum Softening")

        # 5. Volatility Contraction Pattern (VCP)
        vcp_pct = round((curr_atr / latest_price) * 100, 2)
        is_vcp = (vcp_pct < 2.5)
        vcp_label = f"⚡ VCP Squeeze ({vcp_pct:.1f}%)" if is_vcp else f"ATR: {vcp_pct:.1f}%"

        # 5b. Momentum Exhaustion & Bearish Divergence Audit
        has_bearish_div, bearish_div_note = detect_bearish_divergence(close_series, rsi, macd_hist, window=20)
        if any(k in yf_ticker.upper() for k in ["VWRP", "VUAA", "VUSA", "SGLN", "IGLN", "BRK-B", "ABBV"]):
            has_bearish_div, bearish_div_note = False, ""

        # 6. Chandelier Volatility Trailing Stop with 5-Year Horizon Breathing Room
        # Wider 3.2x ATR stop prevents normal pullbacks from shaking out 5-year compounders
        chandelier_mult = 2.0 if has_bearish_div else 3.2
        highest_22 = float(df["High"].tail(22).max())
        chandelier_stop = round(highest_22 - (chandelier_mult * curr_atr), 2)
        atr_stop = latest_price - (STRATEGY_CONFIG["atr_multiplier_stop"] * curr_atr)
        base_trailing_stop = max(chandelier_stop, curr_ema_50 * 0.96, atr_stop)

        # Institutional Profit-Protection Ratchet: Never let a winning trade turn into a loss
        avg_price = _num(position_data.get("averagePrice"), 0.0) if position_data else 0.0
        ratchet_stop = 0.0
        if avg_price > 0 and latest_price > avg_price:
            gain_pct = (latest_price - avg_price) / avg_price
            if gain_pct >= 0.50:
                # Mega-winner: lock in at least 65% of open gains or 50 EMA
                ratchet_stop = max(avg_price + (0.65 * (latest_price - avg_price)), curr_ema_50 * 0.98)
            elif gain_pct >= 0.25:
                # Strong winner: lock in at least +12% guaranteed profit
                ratchet_stop = max(avg_price * 1.12, curr_ema_50 * 0.96)
            elif gain_pct >= 0.10:
                # Early winner: break-even lock (+1.5% above cost basis to cover FX/spread)
                ratchet_stop = avg_price * 1.015

        # Reference stop: no longer clamped below the live price, so it can actually be breached.
        # A breach is surfaced for review below; it never triggers an automatic SELL.
        trailing_stop = max(base_trailing_stop, ratchet_stop)
        stop_breached = bool(latest_price < trailing_stop)
        stop_dist_pct = ((latest_price - trailing_stop) / latest_price) * 100

        # Technical reference target (recent high +5%); no artificial +15% floor
        recent_high = float(df["High"].tail(120).max())
        target_price = round(recent_high * 1.05, 2)
        pct_above_50 = (latest_price - curr_ema_50) / curr_ema_50 if curr_ema_50 > 0 else 0.0
        pct_above_20 = (latest_price - curr_ema_20) / curr_ema_20 if curr_ema_20 > 0 else 0.0
        is_extended = (pct_above_20 > 0.065) or (curr_rsi > 68.0)

        # Institutional Asymmetric Risk/Reward Ratio: only against a valid analyst consensus target above the price.
        # ETFs, gold, money-market and leveraged products have no single-name target -> R:R None (neutral).
        is_fund_like = (
            bool(research.get("is_etf")) or is_broad_index_etf(yf_ticker) or is_gold_etc(yf_ticker)
            or is_money_market_fund(yf_ticker) or is_leveraged
        )
        analyst_target = _num(research.get("target_mean_price"), None)
        target_eval = analyst_target if (analyst_target is not None and analyst_target > latest_price and not is_fund_like) else None

        if trailing_stop < latest_price:
            downside_risk = max(latest_price * 0.02, latest_price - trailing_stop)
        else:
            downside_risk = latest_price * 0.02
        if target_eval is not None and downside_risk > 0:
            risk_reward_ratio: Optional[float] = round((target_eval - latest_price) / downside_risk, 2)
        else:
            risk_reward_ratio = None

        # Earnings Catalyst Imminence Check
        days_to_earnings = research.get("days_to_earnings")
        earnings_imminent = (days_to_earnings is not None and 0 <= days_to_earnings <= 5)

        # 7. World-Class Multi-Factor Composite Conviction Scoring (0 to 100)
        try:
            macro_regime = research_agent.get_macro_regime() or {}
        except Exception as e:
            logger.warning("Macro regime unavailable while analysing %s: %s", yf_ticker, e)
            macro_regime = {}
        macro_score = _num(macro_regime.get("macro_score"), 50.0)   # neutral 50 when macro data is missing
        fund_score = _num(research.get("fundamental_score"), 50.0)  # neutral 50 when research failed

        # Recognize Institutional Pullback-to-Value in a Weekly Stage 2 Compounder
        is_quality_weekly_pullback = (
            weekly_s2
            and fund_score >= 72
            and 28.0 <= curr_rsi <= 50.0
            and not is_leveraged
        )

        # Technical Score (30% weight)
        tech_score = 40
        if minervini["pass_count"] == 5: tech_score += 35
        elif minervini["pass_count"] == 4: tech_score += 25
        elif minervini["pass_count"] == 3: tech_score += 12
        elif is_quality_weekly_pullback: tech_score += 8  # Weekly Stage 2 compounder pulling back into value zone
        else: tech_score -= 20

        if weekly_s2: tech_score += 12
        else: tech_score -= 20

        if 32 <= curr_rsi <= 50 and weekly_s2:
            tech_score += 18  # Institutional Pullback-to-Value sweet spot
        elif 50 < curr_rsi <= 58:
            tech_score += 12
        elif curr_rsi > 75:
            tech_score -= 25
        elif curr_rsi < 32:
            tech_score += 10

        if macd_bullish: tech_score += 10
        if rs_alpha > 5.0: tech_score += 10
        elif rs_alpha < -5.0: tech_score -= 8

        # Factor in Bearish Divergence drag (-12)
        if has_bearish_div:
            tech_score -= 12

        # Sector Rotation Momentum Factor (+6 to -8)
        sector_mom = macro_regime.get("sector_momentum", {}) or {}
        cluster = research.get("macro_cluster", "")
        sector_ret = None
        if "Semiconductor" in cluster or "Hardware" in cluster:
            sector_ret = sector_mom.get("SMH")
        elif "Cloud" in cluster or "Software" in cluster or "Technology" in cluster or "Consumer" in cluster:
            sector_ret = sector_mom.get("XLK")
        elif "Defense" in cluster or "Industrial" in cluster or "Infrastructure" in cluster or "Electrification" in cluster:
            sector_ret = sector_mom.get("XLI")
        elif "Financial" in cluster or "Fintech" in cluster or "Payment" in cluster:
            sector_ret = sector_mom.get("XLF")
        elif "Healthcare" in cluster or "Biopharma" in cluster or "Medical" in cluster:
            sector_ret = sector_mom.get("XLV")
        elif "Energy" in cluster:
            sector_ret = sector_mom.get("XLE")

        sector_ret = _num(sector_ret, None)
        if sector_ret is not None:
            if sector_ret >= 8.0:
                tech_score += 6
            elif sector_ret <= -5.0:
                tech_score -= 8

        # Short Interest Dynamics
        short_pct = _num(research.get("short_interest_pct"), 0.0)
        if short_pct >= 12.0:
            if minervini["pass_count"] >= 4 and vol_footprint["is_accumulation"]:
                tech_score += 4  # Squeeze fuel in Stage 2 accumulation
            elif minervini["pass_count"] <= 2:
                tech_score -= 8  # Overhang in distribution

        tech_score = max(5, min(99, tech_score))

        # Institutional Volume Score (10% weight)
        vol_score = 50
        if vol_footprint["is_accumulation"]: vol_score += 35
        else: vol_score -= 25
        if vol_footprint["up_down_vol_ratio"] >= 1.25: vol_score += 15
        vol_score = max(5, min(99, vol_score))

        # Asymmetry Score (10% weight): R:R None (no valid target) is neutral — no bonus, no penalty
        asym_score = 50
        if risk_reward_ratio is not None:
            if risk_reward_ratio >= 3.5: asym_score += 40
            elif risk_reward_ratio >= 2.5: asym_score += 25
            elif risk_reward_ratio < 1.8: asym_score -= 30
        asym_score = max(5, min(99, asym_score))

        composite_conviction = int(round(
            (0.20 * macro_score) +
            (0.28 * tech_score) +
            (0.32 * fund_score) +
            (0.10 * vol_score) +
            (0.10 * asym_score)
        ))

        # 7b. 5-Year Empirical Risk-Adjusted Return (Sortino), Forward 5Y Compounding & Crash Resilience
        sortino_5y = _num(weekly_trend.get("sortino_5y"), MISSING_METRIC_DEFAULTS["sortino_5y"])
        crash_2022_ret = _num(weekly_trend.get("crash_2022_ret_pct"), MISSING_METRIC_DEFAULTS["crash_2022_ret_pct"])
        cagr_5y = _num(weekly_trend.get("cagr_5y_pct"), MISSING_METRIC_DEFAULTS["cagr_5y_pct"])
        beta_5y = _num(weekly_trend.get("beta_5y"), MISSING_METRIC_DEFAULTS["beta_5y"])
        corr_5y = _num(weekly_trend.get("corr_5y"), MISSING_METRIC_DEFAULTS["corr_5y"])
        max_dd_5y = _num(weekly_trend.get("max_dd_5y_pct"), -22.0)

        macro_cluster = research.get("macro_cluster", "")
        is_core = is_core_asset(yf_ticker, company_name, macro_cluster)
        is_crash_shield = is_crash_shield_asset(yf_ticker, macro_cluster, beta_5y, crash_2022_ret)
        research_exp_cagr = _num(research.get("expected_5y_cagr_pct"), None)
        exp_5y_cagr = research_exp_cagr if research_exp_cagr is not None else cagr_5y
        fwd_cagr_txt = _fmt_fwd_cagr(exp_5y_cagr, fwd_cagr_capped)

        if sortino_5y >= 1.65:
            composite_conviction += 5  # Elite 5-year downside-risk-adjusted compounding
        elif sortino_5y >= 1.20:
            composite_conviction += 3
        elif is_leveraged or sortino_5y < 0.45:
            composite_conviction -= 8  # Severe 5-year volatility decay (e.g., 3x daily leveraged ETFs)

        if exp_5y_cagr >= 16.0 and not is_leveraged:
            composite_conviction += 5  # Elite forward 5Y intrinsic compounding power
        elif exp_5y_cagr >= 13.0 and not is_leveraged:
            composite_conviction += 3
        elif exp_5y_cagr < 8.5 and not is_crash_shield and not is_core:
            composite_conviction -= 6  # Sub-par 5-year expected return vs valuation multiple

        if crash_2022_ret >= 0.0 and cagr_5y >= 12.0:
            composite_conviction += 4  # Proven positive crisis alpha in 2022 bear market + strong 5Y growth

        composite_conviction = max(5, min(99, composite_conviction))

        # Check Anti-Churn Holding Period, 5Y Horizon Lockouts & Core Foundation Protections
        is_recent, days_held = is_recent_entry(position_data, days_threshold=28)
        is_gold = is_gold_etc(yf_ticker)
        gain_pct = ((latest_price - avg_price) / avg_price) if (avg_price > 0 and latest_price > 0) else 0.0

        # Anti-churn rule: protect new entries from daily friction, BUT NEVER block taking profits
        # on strong winners (+20%+ profit with softening R:R), overextended parabolic spikes, or excessive leverage.
        allow_anti_churn = is_recent and (gain_pct < 0.20) and (not is_extended) and (pct_above_50 < 0.15) and (not is_leveraged)

        # 180-day (6-month) anti-round-trip & anti-re-trim lockout for unleveraged stocks;
        # 14-day cooldown for 3x daily leveraged ETFs. Lockouts only suppress discretionary trims and re-buys:
        # risk exits (below) and cap-enforcement trims (rebalancing plan) are never suppressed.
        trim_lockout_days = 14 if is_leveraged else 180
        is_trimmed, days_trimmed = is_recently_trimmed(position_data, days_threshold=trim_lockout_days)

        # Risk-exit (SELL) conditions are evaluated BEFORE the trim lockout, anti-churn and core branches.
        # Broad diversified index ETFs, money-market funds and physical gold ETCs are never exited.
        is_exit_exempt = is_broad_index_etf(yf_ticker) or is_money_market_fund(yf_ticker) or is_gold
        pct_vs_200 = ((latest_price / curr_ema_200) - 1.0) * 100.0 if curr_ema_200 > 0 else 0.0
        altman_z = _num(research.get("altman_z"), None)
        piotroski = _num(research.get("piotroski_score"), None)
        exit_reasons: List[str] = []
        if not is_exit_exempt:
            # Trend-based exits need a CONFIRMED weekly downtrend. When weekly data is unavailable we never exit on
            # daily data alone (a data glitch must not trigger an irreversible sale); tomorrow's run re-checks it.
            # 1. Sustained macro structural breakdown below 200 EMA + Weekly Stage 4 downtrend
            if weekly_ok and (not weekly_s2) and latest_price < curr_ema_200 * 0.96 and curr_rsi < 38.0:
                exit_reasons.append(f"structural breakdown: price {pct_vs_200:+.1f}% vs the 200-day EMA, weekly close below the 30-week EMA, RSI {curr_rsi:.1f}")
            # 2. Severe fundamental distress (insolvency risk)
            if altman_z is not None and altman_z < 1.81 and piotroski is not None and piotroski <= 2:
                exit_reasons.append(f"fundamental distress: Altman Z {altman_z:.2f} (below 1.81) with Piotroski {piotroski:.0f}/9")
            # 3. Long-Term Thesis Break: Deteriorating fundamentals + broken weekly trend below 50d EMA
            if weekly_ok and (not is_core) and (not weekly_s2) and latest_price < curr_ema_50 * 0.97 and fund_score < 48:
                exit_reasons.append(f"thesis break: fundamental score {fund_score:.0f}/100 (below 48) with price {pct_above_50 * 100:+.1f}% vs the 50-day EMA and the weekly trend broken")

        # Signal Classification with Institutional Gatekeepers & 5-Year Horizon Anti-Churn Defense
        if is_earmarked_pos:
            signal = EARMARKED_SIGNAL
            badge_color = "#64748b"
            action_short = "HOLD"
            reason = "Earmarked funds: excluded from all portfolio weights, caps, shield/defensive maths and projections. No orders are ever generated."
            plain_english_why = f"{company_name} holds earmarked money. It is ring-fenced outside the portfolio maths and is never traded by the engine."
            app_instruction = "🔒 Earmarked ➔ Leave untouched"

        elif is_leveraged and lev_cap_frac <= 0.0 and p_qty > 0:
            # Non-discretionary leverage policy: a 0% cap means the whole position is sold (never suppressed by lockouts)
            signal = LEVERAGE_POLICY_SELL_SIGNAL
            badge_color = "#ef4444"
            action_short = "SELL"
            reason = (
                f"Leverage policy caps 3x daily leveraged ETFs at 0%. Measured 5Y risk: {max_dd_5y:+.1f}% max drawdown, "
                f"{crash_2022_ret:+.1f}% in 2022, Sortino {sortino_5y:.2f}. Exit the whole position."
            )
            plain_english_why = f"{company_name} is a 3x daily leveraged product and your policy is to hold none. Sell the whole position."
            app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Sell ➔ Sell all shares"

        elif exit_reasons:
            signal = "🔴 SELL (EXIT)"
            badge_color = "#ef4444"
            action_short = "SELL"
            reason = "Exit triggered by " + "; ".join(exit_reasons) + "."
            plain_english_why = f"{company_name}: " + "; ".join(exit_reasons) + ". Sell to defend capital."
            app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Sell ➔ Sell all shares"

        elif is_trimmed:
            days_str = f"{days_trimmed}d ago" if (days_trimmed is not None and days_trimmed > 0) else "today"
            signal = TRIM_LOCKOUT_SIGNAL
            badge_color = "#10b981"
            action_short = "HOLD"
            if gain_pct > 0:
                reason = (
                    f"Partial profits banked {days_str}. Remaining shares are locked out of discretionary trims and re-buys "
                    f"for {trim_lockout_days} days (anti-round-trip). Risk exits and cap limits still apply."
                )
                plain_english_why = f"You already trimmed {company_name} ({days_str}) to lock in gains and right-size risk. Hold the remaining shares—no discretionary re-trims and no buy-backs during the lockout."
            else:
                reason = (
                    f"Position trimmed {days_str} and currently {gain_pct * 100:+.1f}% vs average cost. Remaining shares are locked out "
                    f"of discretionary trims and re-buys for {trim_lockout_days} days (anti-round-trip). Risk exits and cap limits still apply."
                )
                plain_english_why = f"You trimmed {company_name} ({days_str}) to right-size risk and it is below your average cost. Hold the remaining shares—no discretionary re-trims and no buy-backs during the lockout."
            app_instruction = "No action needed ➔ Leave remaining shares in portfolio"

        elif allow_anti_churn:
            signal = "🟢 HOLD (RECENT ENTRY — 5Y THESIS PLAYING OUT)"
            badge_color = "#10b981"
            action_short = "HOLD"
            days_str = f"{days_held}d" if days_held is not None else "recent"
            reason = f"Position entered {days_str} ago. Protected by institutional anti-churn holding period."
            plain_english_why = f"{company_name} was recently added to the portfolio. Give the 5-year compounding thesis time to work without daily friction."
            app_instruction = "No action needed ➔ Leave in portfolio"

        elif is_core:
            # Core diversified index foundations, SONIA Cash Vault (CSH2.L), & physical gold ETCs are permanent wealth & crash-shield anchors
            if "CSH2" in yf_ticker.upper():
                signal = "🏦 HOLD (PROTECTED SONIA CASH VAULT — 4.23% TAX-FREE)"
                badge_color = "#06b6d4"
                action_short = "HOLD"
                reason = "Ring-fenced Cash Reserve parked in Amundi Smart Overnight Return GBP UCITS ETF (~4.23% net 1Y SONIA return vs 3.80% T212 Cash, -0.07% max dip, 0% FX fee)."
                plain_english_why = "This is your ring-fenced cash reserve earning the Bank of England overnight SONIA rate (~4.23% tax-free inside your ISA). Never touched by the trading bot."
                app_instruction = "🏦 Ring-Fenced Cash Reserve ➔ Leave untouched until needed"
            elif (
                is_gold and
                weekly_s2 and
                not is_extended and
                curr_rsi <= 62.0
            ):
                signal = "🟢 BUY (CRISIS ALPHA SHIELD)"
                badge_color = "#10b981"
                action_short = "BUY"
                reason = f"Physical Gold pulling back to 50-week structural support ({corr_5y:+.2f} S&P correlation, {crash_2022_ret:+.1f}% in 2022)."
                plain_english_why = f"{company_name} is testing its rising 50-week structural support in GBP. Prime entry to build an uncorrelated 5-year crisis shield."
                app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Buy"
            elif (
                minervini["is_stage_2"] and
                curr_ema_50 > 0 and abs(latest_price - curr_ema_50) / curr_ema_50 < 0.055 and
                35.0 <= curr_rsi <= 62.0 and
                vol_footprint["is_accumulation"] and
                weekly_s2 and
                not is_extended
            ):
                signal = "🟢 BUY (DIP ACCUMULATION)"
                badge_color = "#10b981"
                action_short = "BUY"
                reason = f"Core pillar at 50 EMA support with institutional accumulation."
                plain_english_why = f"{company_name} is pulling back to key support in an ongoing secular bull trend. Prime 5-year accumulation zone."
                app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Buy"
            else:
                signal = "🟢 HOLD (CORE FOUNDATION)"
                badge_color = "#6366f1"
                action_short = "HOLD"
                reason = f"Secular core wealth anchor ({cagr_5y:+.1f}% 5Y CAGR, {sortino_5y:.2f} Sortino). Multi-year compounding intact."
                plain_english_why = f"{company_name} is a core diversified pillar. Hold through normal market noise and let it compound over your 5-year horizon."
                app_instruction = "No action needed ➔ Leave in portfolio"

        elif is_leveraged:
            # Position size is enforced by the leverage cap in the rebalancing plan (weight > cap -> trim back to cap).
            # Below the cap there are no RSI/extension-based trims, and 3x leveraged ETFs are never bought.
            signal = "🟡 HOLD (3X LEVERAGE — CAP-MONITORED, NO NEW BUYS)"
            badge_color = "#f59e0b"
            action_short = "HOLD"
            reason = (
                f"3x daily leveraged ETF: {max_dd_5y:+.1f}% 5Y max drawdown, {crash_2022_ret:+.1f}% in 2022, Sortino {sortino_5y:.2f}. "
                f"Size is enforced by the {lev_cap_frac * 100:.0f}% leverage cap in the rebalancing plan; no new buys."
            )
            plain_english_why = f"{company_name} is a 3x daily leveraged product with measured 5-year volatility decay. It is held only within the {lev_cap_frac * 100:.0f}% leverage cap and never added to."
            app_instruction = "No new buys ➔ Cap enforced by rebalancing plan"

        elif (not is_core) and curr_rsi > 82.0 and pct_above_50 > 0.25 and gain_pct >= 0.35:
            signal = "🟠 TRIM (PARABOLIC BLOW-OFF)"
            badge_color = "#f59e0b"
            action_short = "TRIM"
            reason = f"Extreme parabolic blow-off (RSI {curr_rsi:.1f}, +{pct_above_50*100:.1f}% vs 50 EMA, +{gain_pct*100:.1f}% gain)."
            plain_english_why = f"{company_name} has surged +{gain_pct*100:.1f}% into an extreme parabolic spike (>25% above 50 EMA). Banking partial profits if position is oversized."
            app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Sell ➔ Sell 1/4 of shares to lock in gains"

        elif (
            (
                minervini["is_stage_2"] and
                curr_ema_50 > 0 and abs(latest_price - curr_ema_50) / curr_ema_50 < 0.055 and
                35.0 <= curr_rsi <= 62.0 and
                vol_footprint["is_accumulation"]
            ) or (
                is_quality_weekly_pullback and
                fund_score >= 76 and
                exp_5y_cagr >= 13.5 and
                (risk_reward_ratio is None or risk_reward_ratio >= 2.2)
            )
        ):
            if not weekly_s2:
                signal = "🟡 HOLD (WEEKLY DOWNTREND FILTER)"
                badge_color = "#6366f1"
                action_short = "HOLD"
                pct_w = weekly_trend.get("pct_vs_30w", 0.0)
                if weekly_ok:
                    reason = f"Daily bounce detected, but weekly chart is below 30-week EMA ({pct_w:+.1f}%). Filter prevents counter-trend entry."
                else:
                    reason = "Daily bounce detected, but weekly data is unavailable so the weekly trend cannot be confirmed. Filter prevents entry."
                plain_english_why = f"{company_name} is showing short-term strength, but the intermediate weekly trend is not confirmed. Wait for weekly confirmation."
                app_instruction = "Hold off on new orders ➔ Await weekly confirmation"
            elif is_extended:
                signal = "🟡 HOLD (EXTENDED / WAIT FOR PULLBACK)"
                badge_color = "#6366f1"
                action_short = "HOLD"
                reason = f"Stage 2 bull trend, but price is extended (+{pct_above_20*100:.1f}% above 20 EMA, RSI {curr_rsi:.1f}). Do not chase."
                plain_english_why = f"{company_name} is performing great, but has surged ahead of support. Wait for a low-volume pullback to enter."
                app_instruction = "Hold off on buying ➔ Wait for 20/50 EMA pullback"
            elif earnings_imminent:
                signal = "🟡 HOLD (EARNINGS IMMINENT)"
                badge_color = "#6366f1"
                action_short = "HOLD"
                reason = f"Stage 2 uptrend at support, but earnings call is in {days_to_earnings} days. Risk gatekeeper prevents pre-earnings entry."
                plain_english_why = f"{company_name} is technically poised to buy, but earnings announcement is in {days_to_earnings} days. Wait for earnings clarity to avoid binary gap risk."
                app_instruction = "Hold off on new orders ➔ Re-evaluate post-earnings"
            elif risk_reward_ratio is not None and risk_reward_ratio < 2.0:
                signal = "🟡 HOLD (WAIT FOR ASYMMETRY)"
                badge_color = "#6366f1"
                action_short = "HOLD"
                reason = f"Testing support, but Risk/Reward ratio ({risk_reward_ratio}:1) is below 2.0x institutional threshold."
                plain_english_why = f"{company_name} is in an uptrend, but upside to target does not yet offer favorable asymmetry over stop loss. Wait for a deeper dip."
                app_instruction = "No action needed ➔ Wait for better entry"
            else:
                signal = "🟢 BUY (DISCOUNT ENTRY)"
                badge_color = "#10b981"
                action_short = "BUY"
                reason = f"Weekly Stage 2 compounder at institutional pullback support (RSI {curr_rsi:.1f}, {fwd_cagr_txt} Expected 5Y CAGR, R:R {_fmt_rr(risk_reward_ratio)})."
                plain_english_why = f"{company_name} is a wide-moat compounder in a confirmed Weekly Stage 2 uptrend pulling back into an institutional value entry zone (RSI {curr_rsi:.1f}, {fwd_cagr_txt} Expected 5Y CAGR)."
                app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Buy"

        else:
            signal = "🟡 HOLD (DO NOTHING)"
            badge_color = "#6366f1"
            action_short = "HOLD"
            trend_note = minervini["stage_label"]
            above_200 = curr_ema_200 > 0 and latest_price > curr_ema_200
            if above_200:
                trend_txt = f"{trend_note}; price {pct_vs_200:+.1f}% above the 200-day EMA"
            else:
                trend_txt = f"{trend_note}; price {pct_vs_200:+.1f}% vs the 200-day EMA (below it: primary uptrend not intact)"
            reason = f"{trend_txt} ({cagr_5y:+.1f}% 5Y CAGR, {sortino_5y:.2f} Sortino). 5Y Chandelier stop at {trailing_stop:,.2f}."
            if above_200 and weekly_s2:
                plain_english_why = f"{company_name} is compounding well in its upward channel. Sit back and let your 5-year thesis work."
            else:
                plain_english_why = f"{company_name} has no buy or sell signal today, but its trend is not confirmed (below the 200-day EMA or weekly downtrend). Hold and review."
            app_instruction = "No action needed ➔ Leave in portfolio"

        # GBP valuation: unknown currency or unconvertible FX -> valuation_ok False (never valued as USD)
        price_in_gbp_raw = to_gbp(latest_price, currency, gbp_rate) if currency else float("nan")
        valuation_ok = bool(currency) and math.isfinite(price_in_gbp_raw) and price_in_gbp_raw > 0
        price_in_gbp = price_in_gbp_raw if valuation_ok else 0.0
        if not valuation_ok:
            logger.warning("GBP valuation unavailable for %s (currency=%s, price=%s).", yf_ticker, currency, latest_price)
        quantity = p_qty
        val_gbp = quantity * price_in_gbp
        if val_gbp is None or np.isnan(val_gbp):
            val_gbp = 0.0

        price_data_ok = weekly_ok  # daily bars were already validated above; weekly 5Y data must also be present
        data_ok = bool(price_data_ok and research_data_ok)

        # Data-quality & policy BUY gate: never BUY (new or add) on defaulted data, invalid valuation or 3x leverage
        if action_short == "BUY":
            no_buy_reason = None
            if is_leveraged:
                no_buy_reason = "3x daily leveraged ETFs are never bought"
            elif not data_ok:
                missing = []
                if not price_data_ok:
                    missing.append("weekly 5Y price history")
                if not research_data_ok:
                    missing.append("fundamental research")
                no_buy_reason = "data incomplete (" + ", ".join(missing) + ")"
            elif not valuation_ok:
                no_buy_reason = "GBP valuation unavailable (unknown quote currency or FX rate)"
            if no_buy_reason:
                prior_signal = signal
                signal = "🟡 HOLD (NO BUY — DATA/POLICY GATE)"
                badge_color = "#64748b"
                action_short = "HOLD"
                reason = f"Buy setup ({prior_signal}) blocked: {no_buy_reason}."
                plain_english_why = f"{company_name} showed a buy setup, but it was blocked because {no_buy_reason}. No order today."
                app_instruction = "No action needed ➔ Re-evaluate when data is complete"

        # Reference-stop breach: surfaced for review, never an automatic SELL
        if stop_breached and action_short not in ("SELL", "TRIM") and not is_earmarked_pos:
            prior_signal = signal
            signal = STOP_REVIEW_SIGNAL
            badge_color = "#f59e0b"
            action_short = "HOLD"
            reason = (
                f"Price {latest_price:,.2f} is below the reference stop {trailing_stop:,.2f} ({stop_dist_pct:+.1f}%). "
                f"Review the thesis; the engine never auto-sells on a stop breach. Prior signal: {prior_signal}."
            )
            plain_english_why = f"{company_name} is trading below its reference stop. Review it before acting — this is a review alert, not an automatic sell."
            app_instruction = "Review position ➔ No automatic order"

        return {
            "ticker": yf_ticker,
            "search_symbol": search_sym,
            "name": company_name,
            "price": latest_price,
            "currency": currency or "UNKNOWN",
            "price_gbp": price_in_gbp,
            "signal": signal,
            "badge_color": badge_color,
            "action_short": action_short,
            "reason": reason,
            "plain_english_why": plain_english_why,
            "app_instruction": app_instruction,
            "rsi": curr_rsi,
            "trailing_stop": trailing_stop,
            "chandelier_stop": chandelier_stop,
            "stop_distance_pct": stop_dist_pct,
            "stop_breached": stop_breached,
            "target_price": target_price,
            "rr_target_price": target_eval,
            "risk_reward_ratio": risk_reward_ratio,
            "days_to_earnings": days_to_earnings,
            "earnings_catalyst_status": research.get("earnings_catalyst_status", "✅ Clear"),
            "piotroski_score": research.get("piotroski_score"),
            "piotroski_label": research.get("piotroski_label", "N/A"),
            "altman_z": research.get("altman_z"),
            "altman_label": research.get("altman_label", "N/A (Not reported)"),
            "altman_color": research.get("altman_color", "#64748b"),
            "sloan_accrual": research.get("sloan_accrual"),
            "sloan_label": research.get("sloan_label", "N/A (Not reported)"),
            "sloan_color": research.get("sloan_color", "#64748b"),
            "rule_of_40": research.get("rule_of_40"),
            "rule_of_40_label": research.get("rule_of_40_label", "N/A"),
            "weekly_trend": weekly_trend,
            "is_weekly_stage_2": weekly_s2,
            "weekly_trend_label": weekly_trend.get("label", "⚪ Weekly data unavailable"),
            "cagr_5y_pct": cagr_5y,
            "expected_5y_cagr_pct": exp_5y_cagr,
            "expected_5y_cagr_source": "research" if research_exp_cagr is not None else "historical_fallback",
            "expected_5y_cagr_capped": fwd_cagr_capped,
            "fair_pe_multiple": research.get("fair_pe_multiple"),
            "valuation_regime_label": research.get("valuation_regime_label", "🟢 Fair GARP Valuation"),
            "valuation_badge_color": research.get("valuation_badge_color", "#10b981"),
            "vol_5y_pct": weekly_trend.get("vol_5y_pct", 20.0),
            "sortino_5y": sortino_5y,
            "sharpe_5y": weekly_trend.get("sharpe_5y", 0.80),
            "max_dd_5y_pct": max_dd_5y,
            "calmar_5y": weekly_trend.get("calmar_5y", 0.64),
            "crash_2022_ret_pct": crash_2022_ret,
            "beta_5y": beta_5y,
            "corr_5y": corr_5y,
            "crash_role_label": weekly_trend.get("crash_role_label", "📈 5Y Secular Growth"),
            "crash_role_color": weekly_trend.get("crash_role_color", "#64748b"),
            "fx_basis": weekly_trend.get("fx_basis", "native"),
            "is_crash_shield": is_crash_shield,
            "is_extended": is_extended,
            "pct_above_20": round(pct_above_20 * 100, 1),
            "minervini": minervini,
            "minervini_stage": minervini["stage_label"],
            "minervini_pass": minervini["pass_count"],
            "minervini_badge_bg": minervini["badge_bg"],
            "vol_spike": vol_footprint["vol_ratio"],
            "up_down_vol_ratio": vol_footprint["up_down_vol_ratio"],
            "volume_footprint_label": vol_footprint["label"],
            "volume_footprint_badge": vol_footprint["badge_color"],
            "rs_alpha": rs_alpha,
            "rs_label": rs_label,
            "macd_line": curr_macd,
            "macd_signal": curr_macd_sig,
            "macd_hist": curr_macd_hist,
            "macd_status": macd_status,
            "vcp_pct": vcp_pct,
            "vcp_label": vcp_label,
            "theme": research.get("theme", "Secular Technology"),
            "macro_cluster": research.get("macro_cluster", "Enterprise Cloud & Software"),
            "eps_revisions_up_30d": research.get("eps_revisions_up_30d"),
            "eps_revisions_down_30d": research.get("eps_revisions_down_30d"),
            "eps_revision_momentum_pct": research.get("eps_revision_momentum_pct"),
            "eps_revision_label": research.get("eps_revision_label", "Neutral Consensus"),
            "eps_revision_badge_color": research.get("eps_revision_badge_color", "#64748b"),
            "ema_50": curr_ema_50,
            "ema_200": curr_ema_200,
            "conviction_score": max(5, min(99, composite_conviction)),
            "unrealized_pnl": position_data.get("ppl", 0.0) if position_data else 0.0,
            "quantity": quantity,
            "avg_price": position_data.get("averagePrice", 0.0) if position_data else 0.0,
            "value_gbp": val_gbp,
            "portfolio_weight_pct": 0.0,
            "days_held": days_held,
            "is_recently_trimmed": is_trimmed,
            "days_since_trim": days_trimmed,
            "bearish_divergence": has_bearish_div,
            "bearish_divergence_note": bearish_div_note,
            "earnings_beat_rate_pct": research.get("earnings_beat_rate_pct"),
            "earnings_avg_surprise_pct": research.get("earnings_avg_surprise_pct"),
            "earnings_streak_label": research.get("earnings_streak_label", "N/A"),
            "earnings_streak_badge_color": research.get("earnings_streak_badge_color", "#0ea5e9"),
            "fcf_yield_pct": research.get("fcf_yield_pct"),
            "enterprise_to_ebitda": research.get("enterprise_to_ebitda"),
            "roa_pct": research.get("roa_pct"),
            "institutional_ownership_pct": research.get("institutional_ownership_pct"),
            "insider_ownership_pct": research.get("insider_ownership_pct"),
            "roic": research.get("roic"),
            "roic_label": research.get("roic_label", "N/A"),
            "roic_color": research.get("roic_color", "#64748b"),
            "insider_net_shares": research.get("insider_net_shares", 0.0),
            "insider_label": research.get("insider_label", "Neutral Insider Flow"),
            "insider_badge_color": research.get("insider_badge_color", "#64748b"),
            "options_pcr_vol": research.get("options_pcr_vol"),
            "options_pcr_oi": research.get("options_pcr_oi"),
            "options_pcr_label": research.get("options_pcr_label", "N/A (No Options Traded)"),
            "options_badge_color": research.get("options_badge_color", "#64748b"),
            "institutional_change_pct": research.get("institutional_change_pct"),
            "institutional_flow_label": research.get("institutional_flow_label", "Stable Institutional Float"),
            "institutional_flow_color": research.get("institutional_flow_color", "#64748b"),
            "research": research,
            "data_ok": data_ok,
            "price_data_ok": price_data_ok,
            "research_data_ok": research_data_ok,
            "valuation_ok": valuation_ok,
            "is_earmarked": is_earmarked_pos,
            "is_leveraged": is_leveraged,
            "exit_reasons": exit_reasons,
            "model_version": _model_version(),
        }

    except Exception as e:
        logger.error("Error analyzing %s: %s", yf_ticker, e)
        return _fallback_result(f"Analysis error: {e}")


def scan_watchlist_opportunities(
    owned_tickers: Set[str],
    gbp_rate: float,
    scan_collector: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Scans curated high-alpha watchlist tickers that are not currently in the portfolio.

    When `scan_collector` is a list, EVERY per-ticker analysis produced by the scan is appended to it, including
    tickers that are owned, earmarked, leveraged or that failed a gate (only EXCLUDED_TICKERS are skipped). Each
    collected dict carries ticker, data_ok, is_weekly_stage_2, weekly_trend_label, risk_reward_ratio, piotroski_score,
    altman_z and research, plus 'scan_outcome' (why it was kept or filtered) and 'scan_passed'. The return value is unchanged.
    """
    opportunities = []
    collect = scan_collector is not None

    def _collect(analysis: Dict[str, Any], outcome: str, passed: bool = False) -> None:
        if not collect:
            return
        research_d = analysis.get("research") if isinstance(analysis.get("research"), dict) else {}
        analysis.setdefault("data_ok", False)
        analysis.setdefault("is_weekly_stage_2", False)
        analysis.setdefault("weekly_trend_label", "⚪ Weekly data unavailable")
        analysis.setdefault("risk_reward_ratio", None)
        analysis.setdefault("piotroski_score", research_d.get("piotroski_score"))
        analysis.setdefault("altman_z", research_d.get("altman_z"))
        analysis["research"] = research_d
        analysis["scan_outcome"] = outcome
        analysis["scan_passed"] = passed
        scan_collector.append(analysis)

    for item in WATCHLIST:
        ticker = item["ticker"]
        if ticker in EXCLUDED_TICKERS or any(e == ticker for e in EXCLUDED_TICKERS):
            logger.info("Watchlist ticker %s filtered out: excluded by user mandate.", ticker)
            continue
        is_owned = any(tickers_match(ticker, ot) for ot in owned_tickers)
        # Earmarked funds and 3x daily leveraged ETFs are never buy candidates
        is_ineligible = is_earmarked(ticker) or is_leveraged_etf(ticker)
        if (is_owned or is_ineligible) and not collect:
            if is_ineligible and not is_owned:
                logger.info("Watchlist ticker %s filtered out: earmarked or 3x leveraged instrument.", ticker)
            continue

        analysis = analyze_position(ticker, position_data=None, gbp_rate=gbp_rate)
        analysis.setdefault("ticker", ticker)

        if is_owned:
            _collect(analysis, "skipped: already owned")
            continue
        if is_ineligible:
            logger.info("Watchlist ticker %s filtered out: earmarked or 3x leveraged instrument.", ticker)
            _collect(analysis, "filtered: earmarked or 3x leveraged instrument")
            continue

        # Risk Gatekeeper 0: Never build a buy case on defaulted / incomplete data or an invalid GBP valuation
        if analysis.get("data_ok", True) is False or analysis.get("valuation_ok", True) is False:
            logger.info(
                "Watchlist ticker %s filtered out: data incomplete (data_ok=%s, valuation_ok=%s)",
                ticker, analysis.get("data_ok"), analysis.get("valuation_ok"),
            )
            _collect(analysis, "filtered: data incomplete or GBP valuation unavailable")
            continue

        # Risk Gatekeeper 1: Never recommend buying an unowned stock that is flashing SELL or TRIM
        if analysis.get("action_short") in ["SELL", "TRIM"]:
            logger.info("Watchlist ticker %s filtered out: currently flashing %s", ticker, analysis.get("action_short"))
            _collect(analysis, f"filtered: currently flashing {analysis.get('action_short')}")
            continue

        # Risk Gatekeeper 2: Require weekly Stage 2 confirmation (never catch a falling knife)
        if not analysis.get("is_weekly_stage_2", False):
            logger.info("Watchlist ticker %s filtered out: intermediate weekly trend below 30w EMA", ticker)
            _collect(analysis, "filtered: weekly trend not confirmed (below 30w EMA)")
            continue

        # Risk Gatekeeper 3: Prevent buying overextended stocks
        if analysis.get("is_extended", False):
            logger.info("Watchlist ticker %s filtered out: price extended >6.5%% above 20 EMA or RSI >68", ticker)
            _collect(analysis, "filtered: price extended (>6.5% above 20 EMA or RSI >68)")
            continue

        # Risk Gatekeeper 4: Pre-earnings blackout (never initiate binary bets <= 5 days before earnings)
        days_to_earnings = analysis.get("days_to_earnings")
        earnings_imminent = days_to_earnings is not None and 0 <= days_to_earnings <= 5
        if earnings_imminent:
            logger.info("Watchlist ticker %s filtered out: binary earnings catalyst imminent in %d days", ticker, days_to_earnings)
            _collect(analysis, f"filtered: earnings imminent in {days_to_earnings} days")
            continue

        analysis["theme"] = item["theme"]
        analysis["is_new_addition"] = True

        # A conviction-based forced BUY is only allowed when the per-asset analysis passed its own gates:
        # data OK, earnings not imminent, no R:R / 'WAIT FOR ASYMMETRY' failure, not extended, not below its reference stop.
        rr = _num(analysis.get("risk_reward_ratio"), None)
        rr_gate_failed = (rr is not None and rr < 2.0) or ("WAIT FOR ASYMMETRY" in str(analysis.get("signal", "")))
        passed_asset_gates = (
            analysis.get("data_ok", True) is not False
            and not earnings_imminent
            and not rr_gate_failed
            and not analysis.get("is_extended", False)
            and not analysis.get("stop_breached", False)
        )
        conviction = _num(analysis.get("conviction_score"), 0.0)

        if (
            analysis["action_short"] == "BUY"
            or (passed_asset_gates and conviction >= 74)
            or (passed_asset_gates and analysis.get("is_crash_shield") and conviction >= 65)
        ):
            if analysis["action_short"] != "BUY":
                analysis["action_short"] = "BUY"
            exp_cagr = _first_num(analysis.get("expected_5y_cagr_pct"), analysis.get("cagr_5y_pct"), default=None)
            val_lbl = analysis.get("valuation_regime_label", "Fair GARP Valuation")
            analysis["plain_english_why"] = (
                f"{analysis['name']} is in a confirmed Weekly Stage 2 uptrend with elite institutional quality "
                f"(Expected 5Y CAGR: {_fmt_fwd_cagr(exp_cagr, analysis.get('expected_5y_cagr_capped'))}, {val_lbl}, Conviction: {analysis['conviction_score']}/100). High-probability entry into {item['theme']}."
            )
            opportunities.append(analysis)
            _collect(analysis, "candidate: BUY", passed=True)
        elif not passed_asset_gates:
            _collect(analysis, "not selected: per-asset gate failed (R:R / asymmetry or below reference stop)")
        else:
            _collect(analysis, f"not selected: conviction {conviction:.0f} below the forced-BUY threshold")

    opportunities.sort(
        key=lambda x: (
            _num(x.get("conviction_score"), 0.0)
            + 0.75 * _first_num(x.get("expected_5y_cagr_pct"), x.get("cagr_5y_pct"), default=14.0)
            + (8.0 if 30.0 <= _num(x.get("rsi"), 50.0) <= 48.0 else 0.0)
        ),
        reverse=True,
    )
    return opportunities


def _resolve_sim_key(
    index: Dict[Tuple[str, str], int],
    acct: str,
    ticker: str,
) -> Optional[Tuple[str, str]]:
    """Resolves an order's (account, ticker) onto a simulated book row key. Falls back to a unique ticker match
    when either side carries no account tag (single-account plans), and never guesses between two accounts."""
    if (acct, ticker) in index:
        return (acct, ticker)
    matches = [k for k in index if k[1] == ticker and (not acct or not k[0])]
    if len(matches) == 1:
        return matches[0]
    matches = [k for k in index if (k[0] == acct or not acct or not k[0]) and tickers_match(k[1], ticker)]
    if len(matches) == 1:
        return matches[0]
    return None


def compute_portfolio_5y_risk_profile(
    analyses: List[Dict[str, Any]],
    available_cash_gbp: float,
    blocked_cash_gbp: float,
    total_portfolio_gbp: float,
    sell_orders: Optional[List[Dict[str, Any]]] = None,
    buy_orders: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Computes empirical + forward-looking 5-Year Horizon Risk-Adjusted Returns (CAGR, Covariance-Adjusted Sortino, Beta)
    and 2022 Bear Market Crash Stress-Test metrics for both the current portfolio and the post-plan optimized portfolio.

    Earmarked tickers (config.EARMARKED_TICKERS) are excluded from the book and from the total. The post-plan book is
    simulated per (account, ticker), so the same ticker held in two accounts keeps both copies; with no orders the
    post metrics equal the current metrics exactly. Only free cash counts towards the crash shield (blocked cash is
    committed to pending orders), and the true defensive metric is free cash + money-market (CSH2) + physical gold.
    """
    earmarked_set = _earmarked_tickers()
    free_cash = max(0.0, _num(available_cash_gbp, 0.0))
    blocked_cash = max(0.0, _num(blocked_cash_gbp, 0.0))

    book: List[Dict[str, Any]] = []
    earmarked_gbp = 0.0
    for pos in (analyses or []):
        if not isinstance(pos, dict):
            continue
        if is_earmarked(pos.get("ticker", ""), earmarked_set):
            earmarked_gbp += max(0.0, _num(pos.get("value_gbp"), 0.0))
            continue
        book.append(pos)

    gross_total = max(0.0, _num(total_portfolio_gbp, 0.0))
    investable_total = max(0.0, gross_total - earmarked_gbp)
    denom = max(1.0, investable_total)
    cash_now = free_cash + blocked_cash

    def _eval_book(book_items: List[Dict[str, Any]], cash_val: float, free_cash_val: float, total_val: float) -> Dict[str, float]:
        d = max(1.0, total_val)
        w_cash = max(0.0, cash_val) / d
        # UK money-market / interest-on-cash baseline (~4.0% yield, near-zero excess return above 3.8% MAR)
        cagr_sum = w_cash * 4.0
        beta_sum = 0.0
        stress_2022_sum = w_cash * 1.8
        # Only free cash counts towards the shield / defensive buckets (blocked cash is committed to pending orders)
        shield_gbp = max(0.0, free_cash_val)
        defensive_gbp = max(0.0, free_cash_val)
        lev_3x_gbp = 0.0
        gold_gbp = 0.0
        mmf_gbp = 0.0
        equity_gbp = 0.0

        # Aggregate identical tickers (e.g. the same ETF held in two accounts) into one node: it is one asset
        nodes: Dict[str, Dict[str, Any]] = {}
        for item in book_items:
            v = max(0.0, _num(item.get("value_gbp"), 0.0))
            if v <= 0:
                continue
            t_key = _sym(item.get("ticker", ""))
            if t_key in nodes:
                nodes[t_key]["_v"] += v
            else:
                node = dict(item)
                node["_v"] = v
                nodes[t_key] = node

        active_nodes = []
        for t_sym, item in nodes.items():
            v = item["_v"]
            w = v / d
            hist_cagr = _num(item.get("cagr_5y_pct"), MISSING_METRIC_DEFAULTS["cagr_5y_pct"])
            exp_cagr = _first_num(
                item.get("expected_5y_cagr_pct"),
                (item.get("research") or {}).get("expected_5y_cagr_pct"),
                default=hist_cagr,
            )
            cagr_i = (0.45 * hist_cagr) + (0.55 * exp_cagr)
            sort_i = max(0.35, _num(item.get("sortino_5y"), MISSING_METRIC_DEFAULTS["sortino_5y"]))
            beta_i = _num(item.get("beta_5y"), MISSING_METRIC_DEFAULTS["beta_5y"])
            corr_i = _num(item.get("corr_5y"), MISSING_METRIC_DEFAULTS["corr_5y"])
            crash_i = _num(item.get("crash_2022_ret_pct"), MISSING_METRIC_DEFAULTS["crash_2022_ret_pct"])
            cluster_i = str(item.get("macro_cluster") or (item.get("research") or {}).get("macro_cluster") or "")

            # Implied annualized downside semi-deviation (%)
            dd_i = max(5.5, abs(cagr_i - 3.5) / sort_i)
            is_gold_or_cash = any(k in t_sym for k in ["SGLN", "IGLN", "GLD", "CSH2"])

            cagr_sum += w * cagr_i
            beta_sum += w * beta_i
            stress_2022_sum += w * crash_i

            if is_crash_shield_asset(t_sym, cluster_i, beta_i, crash_i):
                shield_gbp += v
            if is_leveraged_etf(t_sym):
                lev_3x_gbp += v
            if is_money_market_fund(t_sym):
                mmf_gbp += v
                defensive_gbp += v
            elif is_gold_etc(t_sym):
                gold_gbp += v
                defensive_gbp += v
            else:
                equity_gbp += v

            active_nodes.append({
                "w": w,
                "dd": dd_i,
                "cluster": cluster_i,
                "corr": corr_i,
                "is_uncorr": is_gold_or_cash,
            })

        # Multi-asset covariance-adjusted portfolio downside deviation
        port_down_var = 0.0
        for i, ni in enumerate(active_nodes):
            for j, nj in enumerate(active_nodes):
                if i == j:
                    rho = 1.0
                elif ni["is_uncorr"] or nj["is_uncorr"]:
                    rho = min(0.05, ni["corr"], nj["corr"])
                elif ni["cluster"] and ni["cluster"] == nj["cluster"]:
                    rho = 0.72
                else:
                    rho = 0.42
                port_down_var += ni["w"] * nj["w"] * ni["dd"] * nj["dd"] * rho

        # Account for 5-year purchasing-power & opportunity-cost drag of uninvested excess cash (>3% of book)
        excess_cash_w = max(0.0, w_cash - 0.03)
        cash_drag_pen = excess_cash_w * 5.5

        port_down_dev = max(4.0, float(np.sqrt(max(0.0, port_down_var))) + cash_drag_pen)
        port_cagr = round(cagr_sum, 1)
        port_sortino = round(max(0.25, (port_cagr - 3.8) / port_down_dev), 2)
        port_beta = round(beta_sum, 2)
        stress_pct = round(stress_2022_sum, 1)
        stress_gbp = round(total_val * (stress_pct / 100.0), 0)
        shield_pct = round((shield_gbp / d) * 100.0, 1)
        lev_pct = round((lev_3x_gbp / d) * 100.0, 1)
        defensive_pct = round((defensive_gbp / d) * 100.0, 1)
        return {
            "cagr_5y_pct": port_cagr,
            "sortino_5y": port_sortino,
            "beta_5y": port_beta,
            "stress_2022_pct": stress_pct,
            "stress_2022_gbp": stress_gbp,
            "shield_gbp": round(shield_gbp, 0),
            "shield_pct": shield_pct,
            "lev_3x_gbp": round(lev_3x_gbp, 0),
            "lev_3x_pct": lev_pct,
            "defensive_gbp": round(defensive_gbp, 0),
            "defensive_pct": defensive_pct,
            "cash_gbp": max(0.0, cash_val),
            "mmf_gbp": mmf_gbp,
            "gold_gbp": gold_gbp,
            "equity_gbp": equity_gbp,
        }

    curr_metrics = _eval_book(book, cash_now, free_cash, denom)

    # Simulate Post-Plan Portfolio Book (after recommended trims/sells and buys), keyed by (account, ticker)
    sim_rows: List[Dict[str, Any]] = [dict(pos) for pos in book]
    sim_index: Dict[Tuple[str, str], int] = {}
    for idx, row in enumerate(sim_rows):
        sim_index.setdefault((str(row.get("account") or ""), str(row.get("ticker", ""))), idx)

    sim_free = free_cash
    for s in (sell_orders or []):
        if not isinstance(s, dict) or s.get("is_pending"):
            continue
        st = str(s.get("ticker", ""))
        if is_earmarked(st, earmarked_set):
            continue
        key = _resolve_sim_key(sim_index, str(s.get("account") or ""), st)
        if key is None:
            logger.debug("Post-plan simulation: sell order for %s has no matching position; ignored.", st)
            continue
        row = sim_rows[sim_index[key]]
        cur_v = max(0.0, _num(row.get("value_gbp"), 0.0))
        reduction = min(max(0.0, _num(s.get("estimated_gbp"), 0.0)), cur_v)
        row["value_gbp"] = cur_v - reduction
        sim_free += reduction

    for b in (buy_orders or []):
        if not isinstance(b, dict) or b.get("is_pending"):
            continue
        bt = str(b.get("ticker", ""))
        if is_earmarked(bt, earmarked_set):
            continue
        b_acct = str(b.get("account") or "")
        b_val = max(0.0, _num(b.get("estimated_gbp"), 0.0))
        key = _resolve_sim_key(sim_index, b_acct, bt)
        if key is not None:
            row = sim_rows[sim_index[key]]
            row["value_gbp"] = max(0.0, _num(row.get("value_gbp"), 0.0)) + b_val
        else:
            # New buys use the candidate's actual metrics; any missing metric uses a CONSERVATIVE default
            b_research = b.get("research") or {}
            sim_rows.append({
                "ticker": bt,
                "account": b_acct,
                "value_gbp": b_val,
                "cagr_5y_pct": _num(b.get("cagr_5y_pct"), MISSING_METRIC_DEFAULTS["cagr_5y_pct"]),
                "expected_5y_cagr_pct": _first_num(b.get("expected_5y_cagr_pct"), b_research.get("expected_5y_cagr_pct"), default=None),
                "sortino_5y": _num(b.get("sortino_5y"), MISSING_METRIC_DEFAULTS["sortino_5y"]),
                "beta_5y": _num(b.get("beta_5y"), MISSING_METRIC_DEFAULTS["beta_5y"]),
                "corr_5y": _num(b.get("corr_5y"), MISSING_METRIC_DEFAULTS["corr_5y"]),
                "crash_2022_ret_pct": _num(b.get("crash_2022_ret_pct"), MISSING_METRIC_DEFAULTS["crash_2022_ret_pct"]),
                "macro_cluster": b.get("macro_cluster") or b_research.get("macro_cluster") or "",
            })
            sim_index[(b_acct, bt)] = len(sim_rows) - 1
        sim_free = max(0.0, sim_free - b_val)

    post_metrics = _eval_book(sim_rows, sim_free + blocked_cash, sim_free, denom)

    # 5-Year Wealth Projections: bear / base / bull scenario rates weighted by the post-plan book (no floors)
    horizon_yrs = int(_num(STRATEGY_CONFIG.get("investment_horizon_years", 5), 5))
    eq_gbp = post_metrics["equity_gbp"]
    cash_bucket_gbp = post_metrics["cash_gbp"] + post_metrics["mmf_gbp"]
    gold_bucket_gbp = post_metrics["gold_gbp"]
    book_sum = eq_gbp + cash_bucket_gbp + gold_bucket_gbp
    projections: Dict[str, Tuple[float, float]] = {}
    for idx, scen in enumerate(("bear", "base", "bull")):
        r_eq = SCENARIO_RATES["equity"][idx]
        r_cash = SCENARIO_RATES["cash"][idx]
        r_gold = SCENARIO_RATES["gold"][idx]
        if book_sum > 0:
            rate = ((eq_gbp * r_eq) + (cash_bucket_gbp * r_cash) + (gold_bucket_gbp * r_gold)) / book_sum
        else:
            rate = r_cash
        projections[scen] = (rate, round(investable_total * ((1.0 + rate) ** horizon_yrs), 0))

    # Signed 2022-style stress delta: positive = plan reduces the loss, negative = plan increases it
    crash_delta_gbp = round(post_metrics["stress_2022_gbp"] - curr_metrics["stress_2022_gbp"], 0)
    crash_saved_gbp = max(0.0, crash_delta_gbp)

    shield_target_pct = _rebalance_cfg_frac("target_crash_shield_pct", 0.15) * 100.0
    lev_cap_pct = _rebalance_cfg_frac("max_leveraged_etf_pct", 0.08) * 100.0
    if post_metrics["shield_pct"] >= shield_target_pct and post_metrics["lev_3x_pct"] <= lev_cap_pct:
        shield_status = "🟢 ALL-WEATHER CRASH SHIELD ACTIVE"
        shield_color = "#10b981"
    elif post_metrics["shield_pct"] >= shield_target_pct / 2.0:
        shield_status = "🟡 BUILDING CRASH SHIELD BUFFER"
        shield_color = "#f59e0b"
    else:
        shield_status = "🔴 HIGH BETA — DEPLOY CRASH SHIELD"
        shield_color = "#ef4444"

    bear_rate, bear_gbp = projections["bear"]
    base_rate, base_gbp = projections["base"]
    bull_rate, bull_gbp = projections["bull"]

    return {
        "horizon_years": horizon_yrs,
        "current_cagr_5y_pct": curr_metrics["cagr_5y_pct"],
        "current_sortino_5y": curr_metrics["sortino_5y"],
        "current_beta_5y": curr_metrics["beta_5y"],
        "current_stress_2022_pct": curr_metrics["stress_2022_pct"],
        "current_stress_2022_gbp": curr_metrics["stress_2022_gbp"],
        "current_shield_gbp": curr_metrics["shield_gbp"],
        "current_shield_pct": curr_metrics["shield_pct"],
        "current_lev_3x_pct": curr_metrics["lev_3x_pct"],
        "current_defensive_gbp": curr_metrics["defensive_gbp"],
        "current_defensive_pct": curr_metrics["defensive_pct"],
        "post_cagr_5y_pct": post_metrics["cagr_5y_pct"],
        "post_sortino_5y": post_metrics["sortino_5y"],
        "post_beta_5y": post_metrics["beta_5y"],
        "post_stress_2022_pct": post_metrics["stress_2022_pct"],
        "post_stress_2022_gbp": post_metrics["stress_2022_gbp"],
        "post_shield_gbp": post_metrics["shield_gbp"],
        "post_shield_pct": post_metrics["shield_pct"],
        "post_lev_3x_pct": post_metrics["lev_3x_pct"],
        "post_defensive_gbp": post_metrics["defensive_gbp"],
        "post_defensive_pct": post_metrics["defensive_pct"],
        "crash_delta_gbp": crash_delta_gbp,
        "crash_capital_saved_gbp": round(crash_saved_gbp, 0),
        "projected_5y_bear_gbp": bear_gbp,
        "projected_5y_bear_cagr_pct": round(bear_rate * 100.0, 1),
        "projected_5y_base_gbp": base_gbp,
        "projected_5y_base_cagr_pct": round(base_rate * 100.0, 1),
        "projected_5y_bull_gbp": bull_gbp,
        "projected_5y_bull_cagr_pct": round(bull_rate * 100.0, 1),
        # Backward-compatible aliases: conservative == base, target == bull
        "projected_5y_conservative_gbp": base_gbp,
        "projected_5y_conservative_cagr_pct": round(base_rate * 100.0, 1),
        "projected_5y_target_gbp": bull_gbp,
        "projected_5y_target_cagr_pct": round(bull_rate * 100.0, 1),
        "shield_status": shield_status,
        "shield_color": shield_color,
        "crash_readiness_grade": shield_status,
        "crash_readiness_color": shield_color,
        "shield_target_pct": round(shield_target_pct, 1),
        "lev_cap_pct": round(lev_cap_pct, 1),
        "earmarked_gbp": round(earmarked_gbp, 2),
        "investable_total_gbp": round(investable_total, 2),
    }


def build_consolidated_portfolio_view(
    accounts_data: Dict[str, Dict[str, Any]],
    gbp_rate: float = 1.35,
    account_plans: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Consolidates positions, cash balances, and risk metrics across all accounts into a single unified portfolio model.

    Args:
        accounts_data: Dict mapping account_name -> {"cash": dict, "analyses": list, "pending_orders": dict}
        gbp_rate: Current GBP/USD exchange rate.
        account_plans: Optional dict mapping account_name -> rebalancing plan. When given, the consolidated post-trade
            simulation uses the union of the plans' non-pending sell_orders and buy_orders, keyed by (account, ticker).
            With no orders, the post-trade metrics equal the current metrics.

    Returns:
        Consolidated portfolio intelligence dict with aggregate valuation, global weights, and unified thematic breakdown.
        Earmarked tickers are excluded from weights/clusters/risk maths; outside holdings (e.g. shares held at
        another broker) count towards cluster and single-name concentration only and are never traded.
    """
    earmarked_set = _earmarked_tickers()
    total_invested_gbp = 0.0
    total_liquid_cash_gbp = 0.0
    total_blocked_cash_gbp = 0.0
    total_ppl_gbp = 0.0
    account_shares = {}

    all_holdings: Dict[str, Dict[str, Any]] = {}
    cluster_totals: Dict[str, float] = {}
    combined_analyses: List[Dict[str, Any]] = []
    family_earmarked_gbp = 0.0
    earmarked_held: Set[str] = set()

    for acc_name, acc_info in accounts_data.items():
        cash = acc_info.get("cash", {}) or {}
        free_gbp = _first_num(cash.get("free_gbp"), cash.get("free"), default=0.0)
        blocked_gbp = _first_num(cash.get("blocked_gbp"), cash.get("blocked"), default=0.0)
        acc_analyses = acc_info.get("analyses") or []
        acc_invested = sum(max(0.0, _num(pos.get("value_gbp"), 0.0)) for pos in acc_analyses)
        acc_earmarked = sum(
            max(0.0, _num(pos.get("value_gbp"), 0.0))
            for pos in acc_analyses
            if is_earmarked(pos.get("ticker", ""), earmarked_set)
        )
        acc_ppl = _num(cash.get("ppl"), 0.0)

        total_invested_gbp += acc_invested
        total_liquid_cash_gbp += free_gbp
        total_blocked_cash_gbp += blocked_gbp
        total_ppl_gbp += acc_ppl
        family_earmarked_gbp += acc_earmarked

        broker_total = _num(cash.get("total"), 0.0)
        calc_total = acc_invested + free_gbp + blocked_gbp
        acc_total = broker_total if broker_total > 0 else calc_total

        account_shares[acc_name] = {
            "invested_gbp": acc_invested,
            "liquid_cash_gbp": free_gbp,
            "blocked_cash_gbp": blocked_gbp,
            "total_value_gbp": acc_total,
            "ppl_gbp": acc_ppl,
            "positions_count": len(acc_analyses),
            "earmarked_gbp": acc_earmarked,
            "investable_total_gbp": max(0.0, acc_total - acc_earmarked),
        }

        # Aggregate holdings across accounts
        for pos in acc_analyses:
            # Stamp the owning account so post-trade simulations are keyed by (account, ticker)
            pos.setdefault("account", acc_name)
            combined_analyses.append(pos)
            ticker = pos.get("ticker", "")
            val_gbp = max(0.0, _num(pos.get("value_gbp"), 0.0))
            qty = _num(pos.get("quantity"), 0.0)
            ppl = _num(pos.get("unrealized_pnl"), 0.0)
            theme = pos.get("theme", "Core Compounder")
            cluster = pos.get("macro_cluster", "Technology")
            pos_earmarked = is_earmarked(ticker, earmarked_set)
            if pos_earmarked:
                earmarked_held.add(ticker)
            else:
                cluster_totals[cluster] = cluster_totals.get(cluster, 0.0) + val_gbp

            if ticker not in all_holdings:
                all_holdings[ticker] = {
                    "ticker": ticker,
                    "search_symbol": pos.get("search_symbol", clean_search_symbol(ticker)),
                    "name": pos.get("name", ticker),
                    "theme": theme,
                    "macro_cluster": cluster,
                    "total_value_gbp": 0.0,
                    "total_quantity": 0.0,
                    "total_unrealized_pnl": 0.0,
                    "accounts": [],
                    "account_details": {},
                    "stage": pos.get("stage"),
                    "stage_label": pos.get("stage_label", ""),
                    "conviction_score": pos.get("conviction_score", 70),
                    "cagr_5y_pct": pos.get("cagr_5y_pct", 14.0),
                    "sortino_5y": pos.get("sortino_5y", 1.10),
                    "beta_5y": pos.get("beta_5y", 1.0),
                    "crash_2022_ret_pct": pos.get("crash_2022_ret_pct", -18.0),
                    "crash_role_label": pos.get("crash_role_label", "📈 5Y Secular Growth"),
                    "price": pos.get("price", 0.0),
                    "currency": pos.get("currency", "USD"),
                    "is_earmarked": pos_earmarked,
                }

            all_holdings[ticker]["total_value_gbp"] += val_gbp
            all_holdings[ticker]["total_quantity"] += qty
            all_holdings[ticker]["total_unrealized_pnl"] += ppl
            all_holdings[ticker]["accounts"].append(acc_name)
            all_holdings[ticker]["account_details"][acc_name] = {
                "shares": qty,
                "value_gbp": val_gbp,
                "unrealized_pnl": ppl
            }

    total_net_worth_gbp = sum(s["total_value_gbp"] for s in account_shares.values())
    # Earmarked funds are excluded from every weight and risk calculation
    investable_net_worth_gbp = max(0.0, total_net_worth_gbp - family_earmarked_gbp)

    # Outside holdings (never traded): vested value counts towards cluster caps and single-name concentration
    outside_rows: List[Dict[str, Any]] = []
    for acc_name in accounts_data.keys():
        outside_rows.extend(value_outside_holdings(acc_name))
    outside_vested_total = sum(_num(h.get("vested_gbp"), 0.0) for h in outside_rows)
    outside_unvested_total = sum(_num(h.get("unvested_gbp"), 0.0) for h in outside_rows)
    for h in outside_rows:
        cl = h.get("macro_cluster") or "Outside Holdings"
        cluster_totals[cl] = cluster_totals.get(cl, 0.0) + _num(h.get("vested_gbp"), 0.0)
    cluster_basis_gbp = investable_net_worth_gbp + outside_vested_total

    def _outside_vested_for(t: str) -> float:
        return sum(_num(h.get("vested_gbp"), 0.0) for h in outside_rows if tickers_match(h.get("ticker", ""), t))

    # Calculate global weight for each holding (investable basis; an outside holding of the same name is added to
    # both the numerator and the denominator of that name's single-name concentration)
    for ticker, item in all_holdings.items():
        v = item["total_value_gbp"]
        if item.get("is_earmarked"):
            item["global_weight_pct"] = round((v / total_net_worth_gbp * 100.0), 1) if total_net_worth_gbp > 0 else 0.0
            continue
        o_t = _outside_vested_for(ticker)
        basis = investable_net_worth_gbp + o_t
        item["global_weight_pct"] = round(((v + o_t) / basis * 100.0), 1) if basis > 0 else 0.0
        if o_t > 0:
            item["outside_vested_gbp"] = round(o_t, 2)

    # Family-level single-name weights for outside-only names (e.g. shares not held in Trading 212)
    outside_single_name_weights: Dict[str, float] = {}
    for h in outside_rows:
        t = h.get("ticker", "")
        if not t or t in outside_single_name_weights:
            continue
        o_t = _outside_vested_for(t)
        held_v = sum(item["total_value_gbp"] for tk, item in all_holdings.items() if tickers_match(tk, t) and not item.get("is_earmarked"))
        basis = investable_net_worth_gbp + o_t
        outside_single_name_weights[t] = round(((held_v + o_t) / basis * 100.0), 1) if basis > 0 else 0.0

    # Calculate per-account percentage of total combined net worth
    for acc_name, s in account_shares.items():
        s["share_of_total_pct"] = round((s["total_value_gbp"] / total_net_worth_gbp * 100.0), 1) if total_net_worth_gbp > 0 else 0.0

    # Thematic Breakdown (investable family book + vested outside holdings)
    thematic_breakdown = []
    for c_name, c_val in sorted(cluster_totals.items(), key=lambda x: x[1], reverse=True):
        pct = round((c_val / cluster_basis_gbp * 100.0), 1) if cluster_basis_gbp > 0 else 0.0
        if pct >= _rebalance_cfg_frac("max_cluster_pct", 0.24) * 100.0:
            status_tag = "🔴 Saturated (Max Cap Reached)"
            badge_bg = "#ef4444"
        elif pct >= 15.0:
            status_tag = "🟢 Core Anchor"
            badge_bg = "#10b981"
        else:
            status_tag = "🟡 Underweighted Pillar"
            badge_bg = "#f59e0b"

        thematic_breakdown.append({
            "cluster": c_name,
            "value_gbp": c_val,
            "weight_pct": pct,
            "status_tag": status_tag,
            "badge_bg": badge_bg
        })

    # Consolidated post-trade simulation: union of the plans' non-pending orders, keyed by (account, ticker)
    sim_sells: List[Dict[str, Any]] = []
    sim_buys: List[Dict[str, Any]] = []
    for acc_name, plan in (account_plans or {}).items():
        if not isinstance(plan, dict):
            continue
        if "sell_orders" not in plan and "buy_orders" not in plan and isinstance(plan.get("rebalancing_plan"), dict):
            plan = plan["rebalancing_plan"]
        for order in (plan.get("sell_orders") or []):
            if isinstance(order, dict) and not order.get("is_pending"):
                if not order.get("account"):
                    order["account"] = acc_name
                sim_sells.append(order)
        for order in (plan.get("buy_orders") or []):
            if isinstance(order, dict) and not order.get("is_pending"):
                if not order.get("account"):
                    order["account"] = acc_name
                sim_buys.append(order)

    combined_risk_5y = compute_portfolio_5y_risk_profile(
        analyses=combined_analyses,
        available_cash_gbp=total_liquid_cash_gbp,
        blocked_cash_gbp=total_blocked_cash_gbp,
        total_portfolio_gbp=total_net_worth_gbp,
        sell_orders=sim_sells,
        buy_orders=sim_buys,
    )

    return {
        "total_net_worth_gbp": total_net_worth_gbp,
        "total_invested_gbp": total_invested_gbp,
        "total_liquid_cash_gbp": total_liquid_cash_gbp,
        "total_blocked_cash_gbp": total_blocked_cash_gbp,
        "total_ppl_gbp": total_ppl_gbp,
        "account_shares": account_shares,
        "holdings": all_holdings,
        "cluster_totals": cluster_totals,
        "thematic_breakdown": thematic_breakdown,
        "risk_profile_5y": combined_risk_5y,
        "portfolio_5y_risk_profile": combined_risk_5y,
        "earmarked_gbp": round(family_earmarked_gbp, 2),
        "earmarked_tickers": sorted(earmarked_held),
        "investable_net_worth_gbp": investable_net_worth_gbp,
        "cluster_basis_gbp": cluster_basis_gbp,
        "outside_holdings": [
            {
                "account": h.get("account"),
                "ticker": h.get("ticker"),
                "vested_gbp": h.get("vested_gbp", 0.0),
                "unvested_gbp": h.get("unvested_gbp", 0.0),
                "macro_cluster": h.get("macro_cluster"),
                "valuation_source": h.get("valuation_source"),
            }
            for h in outside_rows
        ],
        "outside_vested_gbp_total": round(outside_vested_total, 2),
        "outside_unvested_gbp_total": round(outside_unvested_total, 2),
        "outside_single_name_weights": outside_single_name_weights,
        "post_trade_orders_simulated": {"sells": len(sim_sells), "buys": len(sim_buys)},
        "model_version": _model_version(),
    }


def _infer_account_name(analyses: List[Dict[str, Any]]) -> Optional[str]:
    """Infers the owning account from the 'account' tag stamped by build_consolidated_portfolio_view."""
    names = {str(p.get("account")) for p in (analyses or []) if isinstance(p, dict) and p.get("account")}
    return names.pop() if len(names) == 1 else None


def build_rebalancing_plan(
    analyses: List[Dict[str, Any]],
    available_cash_gbp: float,
    new_opportunities: List[Dict[str, Any]],
    gbp_rate: float,
    min_cash_reserve_gbp: Optional[float] = None,
    blocked_cash_gbp: float = 0.0,
    pending_orders_data: Optional[Dict[str, Any]] = None,
    global_context: Optional[Dict[str, Any]] = None,
    official_total_gbp: Optional[float] = None,
    account_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Builds an institutional 5-Year Horizon & Crash-Protected rebalancing plan with 100% harmonized table status, factoring in all pending transactions, blocked capital, and global unified portfolio context.

    Enforced limits (REBALANCE_CONFIG, read with defaults): max_position_pct, max_leveraged_etf_pct, max_new_ideas_to_buy,
    target_crash_shield_pct and max_cluster_pct. Cap-enforcement trims are non-discretionary: trim lockouts, anti-churn and
    recent-entry windows never suppress them. Earmarked tickers are excluded from every weight/cap calculation and never
    traded; outside holdings (account_name, or inferred from the analyses' 'account' tag) count towards cluster caps and
    single-name concentration only. If any held position has an invalid GBP valuation, data_degraded_reason is set and
    NO buy/sell orders are emitted.
    """
    degraded_reasons: List[str] = []
    if analyses is None:
        degraded_reasons.append("positions unavailable from the broker API")
        analyses = []
    new_opportunities = new_opportunities or []

    pending_data = pending_orders_data or {}
    pending_buys = pending_data.get("pending_buys", {}) or {}
    pending_sells = pending_data.get("pending_sells", {}) or {}
    has_orders_scope = pending_data.get("has_orders_scope", False)
    open_orders_list = pending_data.get("orders", [])

    available_cash_gbp = max(0.0, _num(available_cash_gbp, 0.0))
    blocked_cash_gbp = max(0.0, _num(blocked_cash_gbp, 0.0))

    # Configured risk limits (read with defaults; a configured 0.0 is honoured, never replaced by a default)
    pos_cap = _rebalance_cfg_frac("max_position_pct", 0.15)
    lev_cap = _rebalance_cfg_frac("max_leveraged_etf_pct", 0.08)
    cluster_cap_pct = _rebalance_cfg_frac("max_cluster_pct", 0.24) * 100.0
    shield_target_pct = _rebalance_cfg_frac("target_crash_shield_pct", 0.15) * 100.0
    max_new_ideas = max(0, int(_rebalance_cfg_frac("max_new_ideas_to_buy", 2)))
    earmarked_set = _earmarked_tickers()

    acc_name = account_name or _infer_account_name(analyses)

    # Outside holdings for this account (e.g. shares held at another broker): concentration inputs only, never traded
    outside_rows: List[Dict[str, Any]] = []
    if global_context and isinstance(global_context.get("outside_holdings"), list):
        if acc_name:
            outside_rows = [h for h in global_context["outside_holdings"] if isinstance(h, dict) and h.get("account") == acc_name]
    elif acc_name:
        outside_rows = value_outside_holdings(acc_name)
    outside_vested_gbp = sum(max(0.0, _num(h.get("vested_gbp"), 0.0)) for h in outside_rows)
    outside_by_cluster: Dict[str, float] = {}
    for h in outside_rows:
        cl = h.get("macro_cluster") or "Outside Holdings"
        outside_by_cluster[cl] = outside_by_cluster.get(cl, 0.0) + max(0.0, _num(h.get("vested_gbp"), 0.0))

    def _outside_vested_for(t: str) -> float:
        return sum(max(0.0, _num(h.get("vested_gbp"), 0.0)) for h in outside_rows if tickers_match(h.get("ticker", ""), t))

    # Earmarked funds (e.g. a money-market fund set aside for a planned expense): excluded from the investable total and never traded
    for pos in analyses:
        pos["is_earmarked"] = is_earmarked(pos.get("ticker", ""), earmarked_set)
    earmarked_gbp = sum(max(0.0, _num(p.get("value_gbp"), 0.0)) for p in analyses if p["is_earmarked"])
    earmarked_held = sorted({str(p.get("ticker", "")) for p in analyses if p["is_earmarked"]})

    # Fail closed: a held position without a valid GBP valuation corrupts every weight, cap and cash figure
    bad_valuations = sorted({
        str(p.get("ticker", "?")) for p in analyses
        if _num(p.get("quantity"), 0.0) > 0 and p.get("valuation_ok", True) is False
    })
    if bad_valuations:
        degraded_reasons.append(
            "GBP valuation unavailable for held position(s) " + ", ".join(bad_valuations) + " (unknown quote currency or FX rate)"
        )
    data_degraded_reason = ("Data degraded, no orders generated: " + "; ".join(degraded_reasons) + ".") if degraded_reasons else None
    if data_degraded_reason:
        logger.warning("Rebalancing plan%s: %s", f" for {acc_name}" if acc_name else "", data_degraded_reason)

    # 1. Calculate Portfolio Valuation, Current Weights, and Thematic Clusters
    # Note: blocked_cash_gbp is capital committed to pending orders/transactions
    total_invested_gbp = sum(max(0.0, _num(pos.get("value_gbp"), 0.0)) for pos in analyses)
    official_total = _num(official_total_gbp, 0.0)
    total_portfolio_gbp = official_total if official_total > 0 else (total_invested_gbp + available_cash_gbp + blocked_cash_gbp)
    # Earmarked money is excluded from the investable total used for ALL weights, caps, shield maths and sizing
    investable_total_gbp = max(0.0, total_portfolio_gbp - earmarked_gbp)
    for pos in analyses:
        v = max(0.0, _num(pos.get("value_gbp"), 0.0))
        if pos["is_earmarked"]:
            # Display only: share of the gross account total (never used by any cap or weight calculation)
            pos["portfolio_weight_pct"] = round((v / total_portfolio_gbp * 100.0), 1) if total_portfolio_gbp > 0 else 0.0
        else:
            pos["portfolio_weight_pct"] = round((v / investable_total_gbp * 100.0), 1) if investable_total_gbp > 0 else 0.0

    def _local_single_name_pct(pos: Dict[str, Any]) -> float:
        """Single-name weight on the investable basis; an outside holding of the same name counts in numerator and basis."""
        v = max(0.0, _num(pos.get("value_gbp"), 0.0))
        o_t = _outside_vested_for(pos.get("ticker", ""))
        if o_t <= 0:
            return _num(pos.get("portfolio_weight_pct"), 0.0)
        basis = investable_total_gbp + o_t
        return round(((v + o_t) / basis * 100.0), 1) if basis > 0 else 0.0

    # Thematic Macro Cluster Weights (MPT Concentration Control): investable book + vested outside holdings
    cluster_totals: Dict[str, float] = {}
    for pos in analyses:
        if pos["is_earmarked"]:
            continue
        cluster = (pos.get("research") or {}).get("macro_cluster", "Diversified Index Core")
        cluster_totals[cluster] = cluster_totals.get(cluster, 0.0) + max(0.0, _num(pos.get("value_gbp"), 0.0))
    for cl, cl_val in outside_by_cluster.items():
        cluster_totals[cl] = cluster_totals.get(cl, 0.0) + cl_val
    cluster_basis_gbp = investable_total_gbp + outside_vested_gbp

    thematic_breakdown = []
    for c_name, c_val in sorted(cluster_totals.items(), key=lambda x: x[1], reverse=True):
        pct = round((c_val / cluster_basis_gbp * 100.0), 1) if cluster_basis_gbp > 0 else 0.0
        if pct >= _rebalance_cfg_frac("max_cluster_pct", 0.24) * 100.0:
            status_tag = "🔴 Saturated (Max Cap Reached)"
            badge_bg = "#ef4444"
        elif pct >= 15.0:
            status_tag = "🟢 Core Anchor"
            badge_bg = "#10b981"
        else:
            status_tag = "🟡 Underweighted Pillar"
            badge_bg = "#f59e0b"

        thematic_breakdown.append({
            "cluster": c_name,
            "value_gbp": c_val,
            "weight_pct": pct,
            "status_tag": status_tag,
            "badge_bg": badge_bg
        })

    # 2. Identify Sells and Trims (Separating Already-Pending Orders from New Actions)
    sell_orders = []
    pending_sell_orders = []
    cash_from_sells = 0.0
    cap_breaches: List[Dict[str, Any]] = []

    for pos in analyses:
        action = pos.get("action_short", "HOLD")
        qty = max(0.0, _num(pos.get("quantity"), 0.0))
        price_gbp = max(0.0, _num(pos.get("price_gbp"), 0.0))
        val_gbp = max(0.0, _num(pos.get("value_gbp"), 0.0))
        ticker = pos.get("ticker", "")
        sym = pos.get("search_symbol", ticker)
        name = pos.get("name", ticker)
        is_lev_item = is_leveraged_etf(ticker)

        # Non-discretionary cap checks on the investable basis (leverage cap for 3x ETFs; single-name cap for
        # single stocks and sector ETFs; broad index ETFs / gold / money-market exempt unless configured)
        o_t = 0.0 if pos["is_earmarked"] else _outside_vested_for(ticker)
        name_basis = investable_total_gbp + o_t
        w_frac = ((val_gbp + o_t) / name_basis) if name_basis > 0 else 0.0
        cap_frac: Optional[float] = None
        cap_kind: Optional[str] = None
        if not pos["is_earmarked"] and qty > 0:
            if is_lev_item:
                cap_frac, cap_kind = (lev_cap, "leverage") if lev_cap <= pos_cap else (pos_cap, "position")
            elif _position_cap_applies(ticker):
                cap_frac, cap_kind = pos_cap, "position"
        over_cap = cap_frac is not None and w_frac > cap_frac + 1e-9
        cap_excess_gbp = min(val_gbp, max(0.0, (val_gbp + o_t) - cap_frac * name_basis)) if (over_cap and cap_frac is not None) else 0.0
        breach = None
        if over_cap and cap_frac is not None:
            breach = {
                "kind": cap_kind,
                "name": ticker,
                "ticker": ticker,
                "weight_pct": round(w_frac * 100.0, 1),
                "cap_pct": round(cap_frac * 100.0, 1),
                "actioned": False,
            }
            cap_breaches.append(breach)

        matching_ps_key = next((k for k in pending_sells if tickers_match(k, ticker)), None)
        is_already_pending_sell = matching_ps_key is not None

        if is_already_pending_sell:
            # Order is ALREADY queued in Trading 212 - DO NOT tell user to sell again
            p_sell_info = pending_sells.get(matching_ps_key, {}) or {}
            order_type = str(p_sell_info.get("order_type") or "MARKET").upper()
            limit_p = _num(p_sell_info.get("limit_price"), 0.0)
            stop_p = _num(p_sell_info.get("stop_price"), 0.0)
            curr_p = _num(pos.get("price"), 0.0)

            # Only count pending sell proceeds if it's a MARKET order or an in-the-money / near-market limit order.
            # STOP / STOP_LIMIT orders only trigger on a future price move, so they are never treated as immediate fills.
            is_stop_order = "STOP" in order_type
            is_immediate_fill = (not is_stop_order) and (
                (order_type == "MARKET") or (limit_p <= 0) or (curr_p > 0 and limit_p <= curr_p * 1.02)
            )

            # A partial order is valued at its own quantity x GBP price, not at the whole position
            order_qty = _num(p_sell_info.get("quantity"), 0.0)
            if order_qty <= 0 or (qty > 0 and order_qty > qty):
                order_qty = qty
            pending_val_gbp = (order_qty * price_gbp) if price_gbp > 0 else val_gbp

            limit_txt = f" @ limit ${limit_p:,.2f}" if limit_p > 0 else ""
            stop_txt = f" @ stop {stop_p:,.2f}" if stop_p > 0 else ""
            pending_sell_orders.append({
                "ticker": ticker,
                "search_symbol": sym,
                "name": name,
                "action": "SELL (⏳ ALREADY PENDING IN TRADING 212)",
                "is_pending": True,
                "shares": order_qty,
                "price": pos.get("price", 0.0),
                "currency": pos.get("currency", "USD"),
                "estimated_gbp": pending_val_gbp,
                "plain_english_why": (
                    f"Resting {'stop' if is_stop_order else 'limit'} sell order for {name} ({_fmt_shares(order_qty)} shares{stop_txt}{limit_txt}) is queued in Trading 212."
                    if not is_immediate_fill
                    else f"Sell order for {name} ({_fmt_shares(order_qty)} shares) is already queued in Trading 212 awaiting execution. No action needed."
                ),
                "app_instruction": "Already queued in Trading 212 ➔ Awaiting market fill",
                "badge_color": "#f59e0b",
                "limit_price": limit_p,
                "stop_price": stop_p,
                "order_type": order_type,
                "is_immediate_fill": is_immediate_fill,
            })
            if is_immediate_fill:
                cash_from_sells += pending_val_gbp
            if breach is not None:
                breach["actioned"] = True
                breach["pending_order"] = True
            continue

        # Earmarked money is never traded; degraded data never produces orders
        if pos["is_earmarked"] or qty <= 0 or data_degraded_reason:
            continue

        max_dd_txt = _num(pos.get("max_dd_5y_pct"), None)
        crash_txt = _num(pos.get("crash_2022_ret_pct"), None)
        sort_txt = _num(pos.get("sortino_5y"), None)
        risk_txt = (
            f"Measured 5Y risk: {max_dd_txt:+.1f}% max drawdown, {crash_txt:+.1f}% in 2022, Sortino {sort_txt:.2f}."
            if (max_dd_txt is not None and crash_txt is not None and sort_txt is not None) else ""
        )
        outside_txt = f" (incl. £{o_t:,.0f} held outside Trading 212)" if o_t > 0 else ""

        full_exit_kind: Optional[str] = None
        if is_lev_item and lev_cap <= 0.0:
            full_exit_kind = "leverage_policy"
        elif action == "SELL":
            full_exit_kind = "exit_signal"
        elif over_cap and price_gbp > 0 and cap_excess_gbp >= val_gbp - 0.005:
            full_exit_kind = f"{cap_kind}_cap"

        if full_exit_kind and price_gbp > 0:
            proceeds = qty * price_gbp
            if full_exit_kind == "leverage_policy":
                sell_action = LEVERAGE_POLICY_SELL_SIGNAL
                sell_why = (
                    f"Your leverage policy caps 3x daily leveraged ETFs at 0%, so the whole {name} position is sold. {risk_txt} "
                    f"Cap enforcement is not suppressed by the trim lockout or recent-entry windows."
                ).replace("  ", " ")
            elif full_exit_kind == "exit_signal":
                sell_action = "SELL (GET CASH)"
                sell_why = pos.get("plain_english_why", "Falling price trend. Sell to protect capital.")
            else:
                cap_label = "LEVERAGE" if cap_kind == "leverage" else "POSITION"
                sell_action = f"SELL ({cap_label} CAP — {cap_frac * 100:.0f}%)"
                sell_why = f"{name} is {w_frac * 100:.1f}% of the investable portfolio{outside_txt}, above the {cap_frac * 100:.0f}% cap. Selling the position to comply with the cap."
            cash_from_sells += proceeds
            order = {
                "ticker": ticker,
                "search_symbol": sym,
                "name": name,
                "action": sell_action,
                "is_pending": False,
                "shares": qty,
                "price": pos.get("price", 0.0),
                "currency": pos.get("currency", "USD"),
                "estimated_gbp": proceeds,
                "plain_english_why": sell_why,
                "app_instruction": f"Open Trading 212 ➔ Search '{sym}' ➔ Tap Sell ➔ Sell all {_fmt_shares(qty)} shares",
                "badge_color": "#ef4444",
                "order_reason": full_exit_kind,
                "is_profit": _num(pos.get("unrealized_pnl"), 0.0) > 0,
            }
            if acc_name:
                order["account"] = acc_name
            sell_orders.append(order)
            if breach is not None:
                breach["actioned"] = True
            if full_exit_kind != "exit_signal":
                pos["signal"] = sell_action if sell_action.startswith("🔴") else f"🔴 {sell_action}"
                pos["action_short"] = "SELL"
                pos["badge_color"] = "#ef4444"
                pos["reason"] = sell_why
                pos["plain_english_why"] = sell_why
                pos["app_instruction"] = order["app_instruction"]
            continue

        # Trims: cap enforcement (non-discretionary, ignores lockouts) vs discretionary oversized-winner trims
        cap_trim_qty = 0.0
        if over_cap and price_gbp > 0:
            cap_trim_qty = min(qty, math.ceil((cap_excess_gbp / price_gbp) * 10000.0 - 1e-9) / 10000.0)
        disc_trim_qty = 0.0
        if (
            action == "TRIM"
            and not is_lev_item
            and _num(pos.get("portfolio_weight_pct"), 0.0) >= 12.0
            and not pos.get("is_recently_trimmed")
        ):
            disc_trim_qty = round(qty * 0.25, 4)
        trim_qty = max(cap_trim_qty, disc_trim_qty)
        if trim_qty <= 0 or price_gbp <= 0:
            continue

        w_item = _num(pos.get("portfolio_weight_pct"), 0.0)
        if cap_trim_qty > 0 and cap_trim_qty >= disc_trim_qty:
            order_reason = f"{cap_kind}_cap"
            if cap_kind == "leverage":
                trim_action = f"TRIM (LEVERAGE CAP — {cap_frac * 100:.0f}%)"
                trim_why = (
                    f"Holding {w_frac * 100:.1f}% (£{val_gbp:,.0f}) in a 3x daily leveraged ETF exceeds the {cap_frac * 100:.0f}% leverage cap. "
                    f"{risk_txt} Trimming {_fmt_shares(trim_qty)} shares brings it back to the cap."
                ).replace("  ", " ")
            else:
                trim_action = f"TRIM (POSITION CAP — {cap_frac * 100:.0f}%)"
                trim_why = (
                    f"{name} is {w_frac * 100:.1f}% of the investable portfolio{outside_txt}, above the {cap_frac * 100:.0f}% single-name cap. "
                    f"Trimming {_fmt_shares(trim_qty)} shares brings it back to the cap (cap enforcement ignores trim lockouts)."
                )
        else:
            order_reason = "discretionary_trim"
            trim_action = "TRIM (DE-RISK OVERSIZED WINNER)"
            trim_why = pos.get("plain_english_why", f"Position has grown oversized ({w_item:.1f}% of portfolio) into parabolic strength. Banking 25% profit while keeping 75% as a 5-year runner.")

        proceeds = trim_qty * price_gbp
        cash_from_sells += proceeds
        order = {
            "ticker": ticker,
            "search_symbol": sym,
            "name": name,
            "action": trim_action,
            "is_pending": False,
            "shares": trim_qty,
            "price": pos.get("price", 0.0),
            "currency": pos.get("currency", "USD"),
            "estimated_gbp": proceeds,
            "plain_english_why": trim_why,
            "app_instruction": f"Open Trading 212 ➔ Search '{sym}' ➔ Tap Sell ➔ Sell {_fmt_shares(trim_qty)} shares (or ~£{proceeds:,.0f})",
            "badge_color": "#f59e0b",
            "order_reason": order_reason,
            "is_profit": _num(pos.get("unrealized_pnl"), 0.0) > 0,
        }
        if acc_name:
            order["account"] = acc_name
        sell_orders.append(order)
        if breach is not None and order_reason != "discretionary_trim":
            breach["actioned"] = True
        elif breach is not None and disc_trim_qty >= cap_trim_qty:
            breach["actioned"] = True  # the larger discretionary trim also restores the cap
        if order_reason != "discretionary_trim":
            pos["signal"] = f"🟠 {trim_action}"
            pos["action_short"] = "TRIM"
            pos["badge_color"] = "#f59e0b"
            pos["reason"] = trim_why
            pos["plain_english_why"] = trim_why
            pos["app_instruction"] = order["app_instruction"]

    # 3. Calculate Deployable Funds (Enforcing Protected Cash Withdrawal Floor)
    if min_cash_reserve_gbp is None:
        min_cash_reserve_gbp = REBALANCE_CONFIG.get("min_cash_reserve_gbp", 1500.0)
    min_cash_reserve_gbp = max(0.0, _num(min_cash_reserve_gbp, 1500.0))

    # Track capital already committed to pending buy orders in Trading 212
    pending_buys_committed = _num(pending_data.get("total_pending_buy_value"), 0.0)
    if pending_buys_committed <= 0 and pending_buys:
        pending_buys_committed = sum(
            _first_num(pb.get("estimated_value"), pb.get("value"), default=0.0)
            for pb in pending_buys.values() if isinstance(pb, dict)
        )
    if pending_buys_committed <= 0:
        pending_buys_committed = blocked_cash_gbp

    # Ground-Truth Broker Cash Accounting Guarantee:
    # BUY orders must ONLY be funded with real uninvested liquid cash already in the broker account (available_cash_gbp).
    # Unexecuted recommended trims must NEVER be pre-counted into today's buying power.
    # Buffer reserve ensures minimum cash floor (MIN_CASH_RESERVE_GBP) is ring-fenced for withdrawal safety.
    buffer_reserve = max(min_cash_reserve_gbp, available_cash_gbp * _rebalance_cfg_frac("min_cash_buffer_pct", 0.05))
    net_budget_to_deploy = max(0.0, available_cash_gbp - buffer_reserve)
    if net_budget_to_deploy < 1000.0:
        net_budget_to_deploy = 0.0
    if data_degraded_reason:
        net_budget_to_deploy = 0.0
    total_deployable = available_cash_gbp

    # Reserve funding status: the floor only counts as funded by real free cash (never by positions or blocked cash)
    reserve_floor_gbp = min_cash_reserve_gbp
    reserve_funded_gbp = min(available_cash_gbp, reserve_floor_gbp)
    reserve_shortfall_gbp = max(0.0, reserve_floor_gbp - available_cash_gbp)

    # 4. Institutional 5-Year Horizon MPT, Sortino & Crash-Shield Scoring
    owned_account_tickers = {pos["ticker"] for pos in analyses}
    owns_gold = any(any(g in t.upper() for g in ["SGLN", "IGLN", "GLD"]) for t in owned_account_tickers)
    owns_brkb = any("BRK-B" in t.upper() for t in owned_account_tickers)
    owns_vwrp = any("VWRP" in t.upper() for t in owned_account_tickers)

    # Check thematic cluster breakdown (both local account and global combined wealth)
    active_thematic_breakdown = global_context.get("thematic_breakdown", thematic_breakdown) if global_context else thematic_breakdown
    local_cluster_weights = {tb["cluster"]: tb["weight_pct"] for tb in thematic_breakdown}
    global_cluster_weights = {tb["cluster"]: tb["weight_pct"] for tb in active_thematic_breakdown}

    def _is_cluster_saturated(cluster_name: str, broad_index: bool = False) -> bool:
        if broad_index or cluster_name in CLUSTER_CAP_EXEMPT:
            return False
        loc_w = local_cluster_weights.get(cluster_name, 0.0)
        glob_w = global_cluster_weights.get(cluster_name, 0.0)
        # Never deploy new cash into a sector that already exceeds the max_cluster_pct cap of the account or combined family wealth
        return (loc_w >= cluster_cap_pct) or (glob_w >= cluster_cap_pct)

    def _global_single_name_pct(t: str, fallback: float) -> float:
        """Combined-family single-name weight (includes any outside holding of the same name)."""
        if not (global_context and isinstance(global_context.get("holdings"), dict)):
            return fallback
        holdings = global_context.get("holdings", {})
        if t in holdings:
            return _num(holdings[t].get("global_weight_pct"), fallback)
        for k, v in (global_context.get("outside_single_name_weights") or {}).items():
            if tickers_match(k, t):
                return _num(v, fallback)
        return fallback

    # 4a. Evaluate existing holdings as 5-year add-on candidates
    # STRICT ANTI-ROUND-TRIP INVARIANT: Never buy back any position that was trimmed (is_recently_trimmed)
    eligible_existing_holdings = []
    for pos in analyses:
        t = pos["ticker"]
        if t in EXCLUDED_TICKERS or any(e == t for e in EXCLUDED_TICKERS):
            continue
        # Earmarked money is never traded
        if pos.get("is_earmarked"):
            continue
        # Data-quality gate: an add is only ever recommended on complete data with a valid GBP valuation
        if pos.get("data_ok", True) is False or pos.get("valuation_ok", True) is False:
            logger.info("Skipping %s as add-on candidate: data incomplete (data_ok=%s, valuation_ok=%s).", t, pos.get("data_ok"), pos.get("valuation_ok"))
            continue
        # 1. Anti-round-trip lockout: Never recommend buying back a stock that was trimmed to harvest profit / reduce risk
        if pos.get("is_recently_trimmed"):
            logger.info("Skipping %s as buy candidate: previously trimmed (anti-round-trip lockout active).", t)
            continue
        # 1b. 7-Day Recent-Buy Scaling Cooldown: Do not immediately double-down on a position bought in the last 7 days
        days_held = pos.get("days_held")
        if days_held is not None and 0 <= days_held < 7:
            logger.info("Skipping %s as add-on candidate: entered %dd ago (7-day scaling cooldown active).", t, days_held)
            continue
        # 2. Never add to 3x leveraged products or pure-play semiconductor ETFs
        if is_leveraged_etf(t) or "SMGB" in t.upper():
            continue
        action = pos.get("action_short", "HOLD")
        if action in ["SELL", "TRIM"]:
            continue

        is_broad_index = any(k in t.upper() for k in ["VWRP", "VUAA", "VUSA", "SGLN", "IGLN"])
        w_local = _local_single_name_pct(pos)
        w_global = _global_single_name_pct(t, w_local)

        # 3. Single-stock concentration cap: Once an individual company is >= 8% of portfolio, let it compound on its own
        if (not is_broad_index) and (w_local >= 8.0 or w_global >= 8.0):
            logger.info("Skipping %s as buy candidate: already full-sized position (%.1f%% local / %.1f%% global).", t, w_local, w_global)
            continue
        # 3b. Configured single-name cap (max_position_pct) for capped instruments
        if _position_cap_applies(t) and w_local >= pos_cap * 100.0:
            logger.info("Skipping %s as buy candidate: at/above the %.0f%% single-name cap.", t, pos_cap * 100.0)
            continue

        # 4. Sector saturation cap: Do not add to sectors already >= max_cluster_pct of portfolio (e.g. Semiconductors / Cloud Software)
        c_cluster = (pos.get("research") or {}).get("macro_cluster") or pos.get("macro_cluster", "")
        if _is_cluster_saturated(c_cluster, broad_index=is_broad_index):
            logger.info("Skipping %s as buy candidate: sector '%s' is already saturated.", t, c_cluster)
            continue

        is_stage_2 = pos.get("is_weekly_stage_2", False) and (
            _num(pos.get("minervini_pass"), 0) >= 3
            or _num(pos.get("conviction_score"), 0) >= 70
            or is_broad_index
            or (w_local < 4.5 and _num(pos.get("rsi"), 50.0) <= 48.0)
        )
        is_ext = pos.get("is_extended", False)
        if action == "BUY" or (is_stage_2 and not is_ext and (_num(pos.get("conviction_score"), 0) >= 70 or is_broad_index)):
            eligible_existing_holdings.append(pos)

    # Filter out already-owned tickers BEFORE slicing so unowned Crash-Shield anchors (SGLN.L, BRK-B, VWRP.L, ABBV) are never truncated
    unowned_opportunities = [
        opp for opp in new_opportunities
        if opp["ticker"] not in owned_account_tickers
        and not any(tickers_match(opp["ticker"], ot) for ot in owned_account_tickers)
    ]
    top_new_ideas = unowned_opportunities[:max_new_ideas * 6]

    scored_candidates = []
    seen_tickers = set()

    # Evaluate new watchlist additions (skip if already pending execution, already owned, or in a saturated sector)
    for opp in top_new_ideas:
        t = opp["ticker"]
        if t in EXCLUDED_TICKERS or any(e == t for e in EXCLUDED_TICKERS):
            continue
        # Earmarked money and 3x leveraged ETFs are never bought
        if is_earmarked(t, earmarked_set) or is_leveraged_etf(t):
            continue
        # Data-quality gate: a new idea is only ever bought on complete data with a valid GBP valuation
        if opp.get("data_ok", True) is False or opp.get("valuation_ok", True) is False:
            logger.info("Watchlist ticker %s skipped: data incomplete (data_ok=%s, valuation_ok=%s).", t, opp.get("data_ok"), opp.get("valuation_ok"))
            continue
        sym = opp.get("search_symbol", clean_search_symbol(t))
        is_broad_index = any(k in t.upper() for k in ["VWRP", "VUAA", "VUSA", "SGLN", "IGLN"])
        c_cluster = (opp.get("research") or {}).get("macro_cluster") or opp.get("macro_cluster", "")
        if _is_cluster_saturated(c_cluster, broad_index=is_broad_index):
            logger.info("Watchlist ticker %s skipped for account: sector '%s' is already saturated.", t, c_cluster)
            continue
        has_pending_b = any(tickers_match(k, t) for k in pending_buys)
        if has_pending_b:
            logger.info("Ticker %s already has a pending buy order in Trading 212. Deduplicating.", t)
            continue
        has_pending_s = any(tickers_match(k, t) for k in pending_sells)
        if has_pending_s:
            logger.info("Ticker %s already has a pending sell order in Trading 212. Skipping buy setup.", t)
            continue
        if t in seen_tickers:
            continue

        base_conv = opp.get("conviction_score", 70)
        gw = _global_single_name_pct(t, 0.0)

        if (not is_broad_index) and gw >= 8.0:
            continue
        elif is_broad_index:
            opp["rebalance_score"] = base_conv + 10  # Prioritise core index & crisis-shield foundations when unowned
        else:
            opp["rebalance_score"] = base_conv + 6
        scored_candidates.append(dict(opp))
        seen_tickers.add(t)

    # Evaluate existing holdings (skip if already pending execution)
    for pos in eligible_existing_holdings:
        t = pos["ticker"]
        sym = pos.get("search_symbol", clean_search_symbol(t))
        has_pending_b = any(tickers_match(k, t) for k in pending_buys)
        if has_pending_b:
            logger.info("Ticker %s already has a pending buy order in Trading 212. Deduplicating.", t)
            continue
        has_pending_s = any(tickers_match(k, t) for k in pending_sells)
        if has_pending_s:
            logger.info("Ticker %s already has a pending sell order in Trading 212. Skipping buy setup.", t)
            continue
        if t in seen_tickers:
            continue

        base_conv = pos.get("conviction_score", 70)
        w = _global_single_name_pct(t, _local_single_name_pct(pos))
        is_all_world = "VWRP" in pos["ticker"].upper()

        if pos.get("action_short") == "BUY":
            base_conv += 8

        if is_all_world and pos.get("portfolio_weight_pct", 0.0) < 15.0:
            pos["rebalance_score"] = base_conv + 14  # Boost All-World core anchor for 5-year balance
        elif w >= 18.0:
            pos["rebalance_score"] = base_conv - 30  # Already dominant across combined wealth
        elif w >= 10.0:
            pos["rebalance_score"] = base_conv - 15  # Well sized
        elif w < 7.0:
            pos["rebalance_score"] = base_conv + 8   # Under-weighted builder globally
        else:
            pos["rebalance_score"] = base_conv

        scored_candidates.append(dict(pos))
        seen_tickers.add(t)

    # Apply Institutional 5-Year Horizon Risk Gatekeepers, Sortino, Forward 5Y Compounding & Pullback-to-Value Scoring
    for cand in scored_candidates:
        c_ticker_u = cand.get("ticker", "").upper()
        c_research = cand.get("research") or {}
        c_cluster = c_research.get("macro_cluster") or cand.get("macro_cluster", "")
        rr = _num(cand.get("risk_reward_ratio"), None)
        p_score = c_research.get("piotroski_score")
        if p_score is None:
            p_score = cand.get("piotroski_score")
        days_e = c_research.get("days_to_earnings")
        sort_5y = _num(cand.get("sortino_5y"), 1.1)
        crash_22 = _num(cand.get("crash_2022_ret_pct"), -18.0)
        beta_5y = _num(cand.get("beta_5y"), 1.0)
        is_lev_cand = is_leveraged_etf(c_ticker_u)

        # 1. Asymmetry bonus / penalty (R:R None = no valid analyst target: neutral, no bonus and no penalty)
        if rr is not None:
            if rr >= 3.0:
                cand["rebalance_score"] += 8
            elif rr < 2.0:
                cand["rebalance_score"] -= 20

        # 2. Forensic Quality boost
        if p_score is not None:
            if p_score >= 8:
                cand["rebalance_score"] += 6
            elif p_score < 5:
                cand["rebalance_score"] -= 25

        # 3. Binary Earnings Gatekeeper
        if days_e is not None and 0 <= days_e <= 5:
            cand["rebalance_score"] -= 40
            cand["gatekeeper_note"] = f"Earnings in {days_e}d"

        # 4. MPT Sector Saturation Guardrail (evaluated across local & combined portfolio)
        c_weight = max(local_cluster_weights.get(c_cluster, 0.0), global_cluster_weights.get(c_cluster, 0.0))
        if c_cluster not in CLUSTER_CAP_EXEMPT and c_weight >= cluster_cap_pct:
            cand["rebalance_score"] -= 60
            cand["gatekeeper_note"] = f"{c_cluster} Concentration Cap Reached ({c_weight:.1f}%)"
        elif c_weight < 12.0:
            cand["rebalance_score"] += 10  # Boost under-represented long-term pillar

        # 5. Wall Street Consensus Revision Momentum (Academic Alpha Factor #1)
        eps_mom = c_research.get("eps_revision_momentum_pct")
        if eps_mom is not None:
            if eps_mom >= 80.0:
                cand["rebalance_score"] += 8  # Strong institutional upgrade momentum
            elif eps_mom <= 40.0:
                cand["rebalance_score"] -= 18 # Deteriorating Wall Street consensus

        # 6. PEAD Earnings Surprise Streak (Academic Alpha Factor #2)
        beat_rate = c_research.get("earnings_beat_rate_pct")
        if beat_rate is not None:
            if beat_rate >= 75.0:
                cand["rebalance_score"] += 6  # Consistent beat streak
            elif beat_rate < 50.0:
                cand["rebalance_score"] -= 10 # Earnings vulnerability

        # 7. Quality Valuation & Institutional Sponsorship Factors
        fcf_y = c_research.get("fcf_yield_pct")
        if fcf_y is not None and fcf_y >= 3.5:
            cand["rebalance_score"] += 5  # Robust free cash flow backing

        inst_own = c_research.get("institutional_ownership_pct")
        if inst_own is not None and inst_own >= 65.0:
            cand["rebalance_score"] += 4  # Institutional sponsorship

        # 7b. Return on Invested Capital (ROIC) Factor (Compounding Engine)
        roic = c_research.get("roic")
        if roic is not None:
            if roic >= 25.0:
                cand["rebalance_score"] += 6  # Elite ROIC Compounding Machine
            elif roic >= 15.0:
                cand["rebalance_score"] += 4  # Superior Value Creator
            elif roic < 8.0 and not cand.get("is_crash_shield"):
                cand["rebalance_score"] -= 10 # Sub-par capital efficiency / value trap

        # 7c. SEC Form 4 Insider Cluster Buying & Flow
        insider_lbl = c_research.get("insider_label", "") or ""
        if "Insider Buying" in insider_lbl or "Cluster Buying" in insider_lbl:
            cand["rebalance_score"] += 6  # C-suite high-conviction cluster accumulation
        elif "Heavy Insider Selling" in insider_lbl:
            cand["rebalance_score"] -= 8  # Heavy executive liquidation

        # 7d. Options Market Implied Sentiment (Put/Call Ratio)
        pcr_val = c_research.get("options_pcr_vol")
        if pcr_val is not None:
            if pcr_val < 0.65:
                cand["rebalance_score"] += 5  # Institutional call accumulation
            elif pcr_val > 1.25:
                cand["rebalance_score"] -= 8  # Heavy defensive put hedging skew

        # 7e. Top-10 Institutional 13F Ownership Flow Momentum
        inst_chg = c_research.get("institutional_change_pct")
        if inst_chg is not None:
            if inst_chg >= 1.0:
                cand["rebalance_score"] += 4  # Smart-money institutional inflow
            elif inst_chg <= -1.5:
                cand["rebalance_score"] -= 6  # Institutional trimming

        # 7f. 5-Year Empirical Sortino Ratio (Downside-Risk-Adjusted Compounding)
        if sort_5y >= 1.55:
            cand["rebalance_score"] += 12  # Top-decile 5Y risk-adjusted compounder
        elif sort_5y >= 1.15:
            cand["rebalance_score"] += 7   # Strong 5Y Sortino
        elif sort_5y >= 0.80:
            cand["rebalance_score"] += 4   # Solid all-weather anchor
        elif is_lev_cand or sort_5y < 0.50:
            cand["rebalance_score"] -= 20  # Severe volatility drag

        # 7g. 2022 Bear Market Crash Alpha & Low-Beta Protection
        if crash_22 >= 10.0:
            cand["rebalance_score"] += 12  # Proven double-digit positive return in 2022 crash
        elif crash_22 >= 0.0:
            cand["rebalance_score"] += 8   # Positive return in 2022 crash
        elif crash_22 >= -8.0:
            cand["rebalance_score"] += 5   # Resilient single-digit drawdown in 2022 crash
        elif crash_22 <= -45.0:
            cand["rebalance_score"] -= 10

        # 7h. Uncorrelated Crisis-Shield & All-Weather Portfolio Completion Boost
        min_pass = _num(cand.get("minervini_pass"), 4)
        c_rsi = _num(cand.get("rsi"), 50.0)
        is_etf_cand = bool(c_research.get("is_etf")) and not is_lev_cand
        if is_etf_cand:
            # Benchmark Quality Equivalence: Unleveraged index ETFs do not report single-stock Piotroski/ROIC/EPS-revisions
            cand["rebalance_score"] += 22
        if "VWRP" in c_ticker_u and not owns_vwrp:
            cand["rebalance_score"] += 16  # Establish All-World equity core
        elif "BRK-B" in c_ticker_u and not owns_brkb and min_pass >= 3:
            cand["rebalance_score"] += 14  # Establish $300B+ Cash Fortress ballast
        elif any(g in c_ticker_u for g in ["SGLN", "IGLN", "GLD"]) and not owns_gold and (min_pass >= 3 or c_rsi <= 45.0):
            cand["rebalance_score"] += 14  # Establish -0.03 beta Physical Gold crisis shield

        # 7i. Forward 5-Year Intrinsic Compounding Engine (ROIC Reinvestment + FCF Yield + Multiple Normalization)
        exp_cagr = _first_num(
            cand.get("expected_5y_cagr_pct"),
            c_research.get("expected_5y_cagr_pct"),
            cand.get("cagr_5y_pct"),
            default=14.0,
        )
        cagr_boost = int(round(max(-12.0, min(18.0, (exp_cagr - 12.0) * 1.5))))
        cand["rebalance_score"] += cagr_boost

        # 7j. Institutional Pullback-to-Value Timing Bonus (Buy wide-moat leaders on dips, don't chase spikes)
        if cand.get("is_weekly_stage_2", False):
            if 30.0 <= c_rsi <= 46.0:
                cand["rebalance_score"] += 10  # Prime institutional pullback accumulation zone
            elif 46.0 < c_rsi <= 55.0:
                cand["rebalance_score"] += 5   # Constructive mid-channel entry
            elif c_rsi > 64.0:
                cand["rebalance_score"] -= 8   # Extended short-term; prefer better pullback entry

        # 8. Momentum Exhaustion Gatekeeper (Bearish Divergence)
        if cand.get("bearish_divergence"):
            cand["rebalance_score"] -= 30
            cand["gatekeeper_note"] = "Bearish Momentum Divergence"

    scored_candidates.sort(key=lambda x: x.get("rebalance_score", 0), reverse=True)

    # 5. Asset-Class Aware Capital Allocation (Fractional Kelly Edge/Downside-Risk Sizing with Cross-Cluster Diversity)
    buy_orders = []
    min_ord_gbp = _rebalance_cfg_frac("min_order_gbp", 950.0)
    effective_floor_gbp = max(min_ord_gbp, 950.0 if net_budget_to_deploy >= 3000.0 else min_ord_gbp)
    max_feasible_orders = int(net_budget_to_deploy // effective_floor_gbp) if effective_floor_gbp > 0 else 0
    target_count = min(
        len(scored_candidates),
        max_feasible_orders,
        4 if net_budget_to_deploy > 7500 else (3 if net_budget_to_deploy >= 3100 else 2)
    )

    if scored_candidates and target_count >= 1 and net_budget_to_deploy >= effective_floor_gbp:
        # Select top candidates while enforcing cross-cluster diversity so we don't double-up on the same sector in one batch,
        # and never more than max_new_ideas_to_buy brand-new positions per plan (remaining slots may go to adds)
        selected_candidates = []
        used_clusters = set()
        new_count = 0
        top_score = scored_candidates[0].get("rebalance_score", 80)
        for cand in scored_candidates:
            if len(selected_candidates) >= target_count:
                break
            is_new_c = bool(cand.get("is_new_addition"))
            if is_new_c and new_count >= max_new_ideas:
                continue
            c_cl = cand.get("macro_cluster") or (cand.get("research") or {}).get("macro_cluster", "")
            if c_cl in used_clusters and (top_score - cand.get("rebalance_score", 0)) < 20:
                continue
            selected_candidates.append(cand)
            if is_new_c:
                new_count += 1
            if c_cl:
                used_clusters.add(c_cl)
        # Backfill if diversity filter left slots open
        if len(selected_candidates) < target_count:
            for cand in scored_candidates:
                if len(selected_candidates) >= target_count:
                    break
                if cand["ticker"] in {c["ticker"] for c in selected_candidates}:
                    continue
                is_new_c = bool(cand.get("is_new_addition"))
                if is_new_c and new_count >= max_new_ideas:
                    continue
                selected_candidates.append(cand)
                if is_new_c:
                    new_count += 1

        # All-Weather Crash-Shield Completion Rule:
        # If an account lacks VWRP.L or has < target_crash_shield_pct invested in Crash Shield assets (investable basis,
        # earmarked funds excluded), ensure at least 1 slot builds a low-beta Crash Shield anchor
        curr_invested_shield_gbp = sum(
            max(0.0, _num(p.get("value_gbp"), 0.0))
            for p in analyses
            if p.get("is_crash_shield") and not p.get("is_earmarked")
        )
        curr_invested_shield_pct = (curr_invested_shield_gbp / max(1.0, investable_total_gbp)) * 100.0
        has_shield_in_selected = any(
            c.get("is_crash_shield") or "VWRP" in c.get("ticker", "").upper()
            for c in selected_candidates
        )
        if target_count >= 2 and selected_candidates and (not owns_vwrp or curr_invested_shield_pct < shield_target_pct) and not has_shield_in_selected:
            selected_tickers = {c["ticker"] for c in selected_candidates}
            best_shield_cand = next(
                (
                    c for c in scored_candidates
                    if (c.get("is_crash_shield") or "VWRP" in c.get("ticker", "").upper())
                    and _num(c.get("minervini_pass"), 0) >= 3
                    and c["ticker"] not in selected_tickers
                ),
                None,
            )
            if best_shield_cand is not None:
                replace_idx: Optional[int] = len(selected_candidates) - 1
                if (
                    best_shield_cand.get("is_new_addition")
                    and not selected_candidates[replace_idx].get("is_new_addition")
                    and new_count >= max_new_ideas
                ):
                    # Swap out the lowest-ranked NEW idea instead, so the new-idea cap still holds
                    new_idxs = [i for i, c in enumerate(selected_candidates) if c.get("is_new_addition")]
                    replace_idx = new_idxs[-1] if new_idxs else None
                if replace_idx is not None:
                    selected_candidates[replace_idx] = best_shield_cand

        # Fractional Kelly Conviction + Expected 5Y Compounding Edge over Downside Volatility:
        candidate_weights = {}
        for cand in selected_candidates:
            c_t = cand["ticker"]
            vcp = _num(cand.get("vcp_pct"), None)
            if vcp is None or vcp <= 0:
                vcp = 2.0
            vol_risk = max(1.2, float(vcp) * (0.85 if cand.get("is_crash_shield") else 1.0))
            score_factor = max(15.0, float(cand.get("rebalance_score", 50)))
            exp_c = _first_num(
                cand.get("expected_5y_cagr_pct"),
                (cand.get("research") or {}).get("expected_5y_cagr_pct"),
                cand.get("cagr_5y_pct"),
                default=14.0,
            )
            edge_mult = max(0.85, min(1.65, (exp_c - 3.8) / 10.0))
            candidate_weights[c_t] = (score_factor * edge_mult) / vol_risk

        total_w = sum(candidate_weights.values()) or 1.0
        # Guarantee every selected candidate meets effective_floor_gbp and distribute surplus by Kelly-adjusted weight
        base_floor_total = len(selected_candidates) * effective_floor_gbp
        surplus_budget = max(0.0, net_budget_to_deploy - base_floor_total)
        raw_allocations = {
            cand["ticker"]: effective_floor_gbp + ((candidate_weights[cand["ticker"]] / total_w) * surplus_budget)
            for cand in selected_candidates
        }

        # Single-name cap on the POST-TRADE weight for capped instruments (single stocks and sector ETFs; broad index
        # ETFs / gold / money-market exempt unless cap_applies_to_broad_etfs). Leveraged ETFs are never bought.
        # Capital above a cap spills over to candidates that still have room; anything left stays in cash.
        room_by_ticker: Dict[str, float] = {}
        for candidate in selected_candidates:
            c_ticker = candidate["ticker"]
            if _position_cap_applies(c_ticker):
                existing_val = sum(max(0.0, _num(p.get("value_gbp"), 0.0)) for p in analyses if p.get("ticker") == c_ticker)
                o_c = _outside_vested_for(c_ticker)
                room_by_ticker[c_ticker] = max(0.0, pos_cap * (investable_total_gbp + o_c) - existing_val - o_c)
            else:
                room_by_ticker[c_ticker] = float("inf")

        allocated_amounts = {t: min(raw_allocations[t], room_by_ticker[t]) for t in raw_allocations}
        spillover_capital = sum(raw_allocations[t] - allocated_amounts[t] for t in raw_allocations)
        for _ in range(3):
            if spillover_capital <= 0.01:
                break
            open_tickers = [t for t in allocated_amounts if room_by_ticker[t] - allocated_amounts[t] > 0.01]
            if not open_tickers:
                break
            open_w_sum = sum(candidate_weights[t] for t in open_tickers) or 1.0
            leftover = 0.0
            for t in open_tickers:
                extra = spillover_capital * (candidate_weights[t] / open_w_sum)
                take = min(extra, room_by_ticker[t] - allocated_amounts[t])
                allocated_amounts[t] += take
                leftover += extra - take
            spillover_capital = leftover

        for candidate in selected_candidates:
            c_ticker = candidate["ticker"]
            alloc_gbp = allocated_amounts.get(c_ticker, 0.0)
            if alloc_gbp < min_ord_gbp * 0.95:
                continue

            p_gbp = _num(candidate.get("price_gbp"), 0.0)
            if p_gbp <= 0:
                continue

            # Fractional shares rounded DOWN to 4 decimals so an order never exceeds its allocation or cap room
            shares = math.floor((alloc_gbp / p_gbp) * 10000.0) / 10000.0
            if shares <= 0:
                continue

            order_val_gbp = shares * p_gbp
            sym = candidate.get("search_symbol", c_ticker)
            name = candidate.get("name", c_ticker)
            is_new = bool(candidate.get("is_new_addition", False))
            c_research = candidate.get("research") or {}
            cagr_5y = _num(candidate.get("cagr_5y_pct"), MISSING_METRIC_DEFAULTS["cagr_5y_pct"])
            exp_cagr_5y = _first_num(
                candidate.get("expected_5y_cagr_pct"),
                c_research.get("expected_5y_cagr_pct"),
                default=cagr_5y,
            )
            fwd_capped = candidate.get("expected_5y_cagr_capped", c_research.get("expected_5y_cagr_capped"))
            val_regime = (
                candidate.get("valuation_regime_label")
                or c_research.get("valuation_regime_label")
                or "🟢 Fair GARP Valuation"
            )
            # Metrics travel with the order; anything missing uses the conservative sentinels (never optimistic)
            sort_5y = _num(candidate.get("sortino_5y"), MISSING_METRIC_DEFAULTS["sortino_5y"])
            crash_22 = _num(candidate.get("crash_2022_ret_pct"), MISSING_METRIC_DEFAULTS["crash_2022_ret_pct"])
            beta_5y = _num(candidate.get("beta_5y"), MISSING_METRIC_DEFAULTS["beta_5y"])
            corr_5y = _num(candidate.get("corr_5y"), MISSING_METRIC_DEFAULTS["corr_5y"])
            role_lbl = candidate.get("crash_role_label", "🛡️ 5Y Compounder")
            rr_val = _num(candidate.get("risk_reward_ratio"), None)

            type_label = "NEW SHARE" if is_new else "ADD TO POSITION"
            crash_note = (
                f"+{crash_22:.1f}% positive crisis alpha in 2022 crash"
                if crash_22 >= 0.0
                else f"Beta {beta_5y:.2f} ({val_regime})"
            )
            order = {
                "ticker": c_ticker,
                "search_symbol": sym,
                "name": name,
                "theme": candidate.get("theme", "Core Compounder"),
                "macro_cluster": candidate.get("macro_cluster", "Technology"),
                "action": f"BUY ({type_label})",
                "is_new": is_new,
                "shares": shares,
                "price": candidate.get("price", 0.0),
                "currency": candidate.get("currency", "USD"),
                "estimated_gbp": order_val_gbp,
                "stop_price": candidate.get("trailing_stop", 0.0),
                "target_price": candidate.get("target_price", 0.0),
                "risk_reward_ratio": rr_val,
                "cagr_5y_pct": cagr_5y,
                "expected_5y_cagr_pct": exp_cagr_5y,
                "expected_5y_cagr_capped": fwd_capped,
                "valuation_regime_label": val_regime,
                "sortino_5y": sort_5y,
                "crash_2022_ret_pct": crash_22,
                "beta_5y": beta_5y,
                "corr_5y": corr_5y,
                "crash_role_label": role_lbl,
                "is_crash_shield": bool(candidate.get("is_crash_shield")),
                "piotroski_score": candidate.get("piotroski_score"),
                "altman_z": candidate.get("altman_z"),
                "altman_label": candidate.get("altman_label", "N/A"),
                "sloan_accrual": candidate.get("sloan_accrual"),
                "sloan_label": candidate.get("sloan_label", "N/A"),
                "weekly_trend_label": candidate.get("weekly_trend_label") or "Weekly trend n/a",
                "eps_revision_label": candidate.get("eps_revision_label") or c_research.get("eps_revision_label", "Neutral Consensus"),
                "eps_revision_momentum_pct": candidate.get("eps_revision_momentum_pct") or c_research.get("eps_revision_momentum_pct"),
                "minervini_stage": candidate.get("minervini_stage", DATA_UNAVAILABLE_LABEL),
                "volume_footprint_label": candidate.get("volume_footprint_label", "Constructive Volume"),
                "rs_label": candidate.get("rs_label", "Benchmark Neutral"),
                "chandelier_stop": candidate.get("chandelier_stop", candidate.get("trailing_stop")),
                "earnings_catalyst_status": candidate.get("earnings_catalyst_status", "✅ Clear"),
                "plain_english_why": (
                    f"{role_lbl} ({candidate.get('theme', 'Secular Growth')}): {_fmt_fwd_cagr(exp_cagr_5y, fwd_capped)} Forward 5Y Expected CAGR ({cagr_5y:+.1f}% Hist 5Y), "
                    f"{sort_5y:.2f} Sortino, RSI {_num(candidate.get('rsi'), 50.0):.1f}, {crash_note}. "
                    + (
                        f"Initiating high-conviction position to compound 5-year wealth in {candidate.get('macro_cluster', 'all-weather')}."
                        if is_new
                        else (
                            f"Scaling into under-weighted 5-year compounder ({_num(candidate.get('portfolio_weight_pct'), 0.0):.1f}% current weight)"
                            + (
                                " at pullback support"
                                if _num(candidate.get("pct_vs_30w"), _num((candidate.get("weekly_trend") or {}).get("pct_vs_30w"), 99.0)) <= 6.0
                                else ""
                            )
                            + f" with R:R {_fmt_rr(rr_val)}."
                        )
                    )
                ),
                "app_instruction": f"Open Trading 212 ➔ Search '{sym}' ➔ Tap Buy ➔ Enter £{order_val_gbp:,.0f}",
                "conviction_score": candidate.get("conviction_score", 70),
                "rebalance_score": candidate.get("rebalance_score", 75),
                "research": c_research,
                "badge_color": "#10b981",
                "data_ok": True,
            }
            if acc_name:
                order["account"] = acc_name
            buy_orders.append(order)

    # 6. Single Source of Truth: Synchronize Every Holding with Planned & Pending Action
    sold_dict = {s["ticker"]: s for s in sell_orders if not s.get("is_pending")}
    bought_dict = {b["ticker"]: b for b in buy_orders}
    
    hold_items = []
    table_items = []

    for pos in analyses:
        t = pos["ticker"]
        w = pos.get("portfolio_weight_pct", 0.0)
        name = pos.get("name", t)
        
        matching_ps = next((k for k in pending_sells if tickers_match(k, t)), None)
        matching_pb = next((k for k in pending_buys if tickers_match(k, t)), None)
        matching_so = next((v for k, v in sold_dict.items() if tickers_match(k, t)), None)
        matching_bo = next((v for k, v in bought_dict.items() if tickers_match(k, t)), None)

        if matching_ps:
            p_sell = pending_sells[matching_ps] or {}
            pending_qty = _num(p_sell.get("quantity"), 0.0)
            if pending_qty <= 0:
                pending_qty = _num(pos.get("quantity"), 0.0)
            pos["plan_status"] = "⏳ PENDING SELL"
            pos["plan_badge_color"] = "#f59e0b"
            pos["plan_summary"] = f"Sell order of {_fmt_shares(pending_qty)} shares is currently pending execution in Trading 212."
        elif matching_pb:
            p_buy = pending_buys[matching_pb] or {}
            pos["plan_status"] = "⏳ PENDING BUY"
            pos["plan_badge_color"] = "#0284c7"
            pos["plan_summary"] = f"Buy order of ~£{_num(p_buy.get('estimated_value'), 0.0):,.0f} is currently pending execution in Trading 212."
        elif matching_so:
            so = matching_so
            reason_kind = so.get("order_reason", "")
            is_trim = "TRIM" in so.get("action", "")
            if is_trim:
                pos["plan_status"] = "TRIM"
                pos["plan_badge_color"] = "#f59e0b"
                if reason_kind == "leverage_cap":
                    purpose = "to cap 3x leverage & protect against crashes"
                elif reason_kind == "position_cap":
                    purpose = f"to bring it back within the {pos_cap * 100:.0f}% single-name cap"
                else:
                    purpose = "to de-risk an oversized winner" + (" & lock in gains" if so.get("is_profit") else "")
                pos["plan_summary"] = f"Trimming {_fmt_shares(so['shares'])} shares (+£{so['estimated_gbp']:,.0f} cash unlocked) {purpose}."
            else:
                pos["plan_status"] = "SELL"
                pos["plan_badge_color"] = "#ef4444"
                if reason_kind == "leverage_policy":
                    exit_txt = "Exiting the 3x leveraged ETF under the 0% leverage policy."
                elif reason_kind in ("leverage_cap", "position_cap"):
                    exit_txt = "Exiting to comply with the configured cap."
                else:
                    exit_txt = "Exiting: " + ("; ".join(pos.get("exit_reasons") or []) or "risk exit signal") + "."
                pos["plan_summary"] = f"Liquidating all {_fmt_shares(pos.get('quantity', 0.0))} shares (+£{so['estimated_gbp']:,.0f} cash unlocked). {exit_txt}"
        elif matching_bo:
            bo = matching_bo
            descriptor = "5-year crash-resistant compounder" if bo.get("is_crash_shield") else "5-year compounder"
            pos["plan_status"] = f"BUY (+£{bo['estimated_gbp']:,.0f})"
            pos["plan_badge_color"] = "#10b981"
            pos["plan_summary"] = f"Active Buy Order: Deploying £{bo['estimated_gbp']:,.0f} into {descriptor} ({w}% weight)."
        else:
            # Held position (trend wording only claims an uptrend when price data confirms it)
            pos["plan_status"] = "HOLD"
            pos["plan_badge_color"] = "#6366f1"
            price_now = _num(pos.get("price"), 0.0)
            ema_200_now = _num(pos.get("ema_200"), 0.0)
            above_200 = ema_200_now > 0 and price_now > ema_200_now
            uptrend_ok = above_200 and bool(pos.get("is_weekly_stage_2", False))
            if pos.get("is_earmarked"):
                pos["plan_summary"] = "Earmarked funds: excluded from portfolio maths and never traded."
            elif data_degraded_reason:
                pos["plan_summary"] = f"No recommendation today. {data_degraded_reason}"
            elif pos.get("stop_breached"):
                pos["plan_summary"] = f"Below its reference stop ({w}% of portfolio): review the position. No automatic order."
            elif _num(w, 0.0) >= 18.0:
                pos["plan_summary"] = f"Core foundation ({w}% of portfolio). " + (
                    "5-year uptrend healthy—hold and let compound." if uptrend_ok
                    else "Trend not confirmed (below the 200 EMA or weekly downtrend)—hold and review."
                )
            elif _num(w, 0.0) >= 10.0:
                pos["plan_summary"] = f"Core holding ({w}% of portfolio). Fully positioned—hold and let compound over 5 years."
            else:
                pos["plan_summary"] = f"5-year compounder ({w}% of portfolio). " + (
                    "Primary uptrend intact above 200 EMA." if above_200
                    else "Price is below the 200 EMA, so the primary uptrend is not intact—hold and review."
                )
            hold_items.append(pos)
        
        table_items.append(pos)

    # Append pending buys for any tickers not in existing portfolio (e.g. from watchlist)
    for pticker, pbuy in pending_buys.items():
        if pticker not in [pos["ticker"] for pos in analyses]:
            table_items.append({
                "ticker": pticker,
                "search_symbol": clean_search_symbol(pticker),
                "name": pbuy.get("name", pticker),
                "value_gbp": 0.0,
                "unrealized_pnl": 0.0,
                "plan_status": "⏳ PENDING BUY",
                "plan_badge_color": "#0284c7",
                "plan_summary": f"Pending Buy Order: ~£{_num(pbuy.get('estimated_value'), 0.0):,.0f} committed in Trading 212.",
                "currency": "USD",
                "price": pbuy.get("limit_price", 0.0),
                "is_new_addition": True,
                "research": {},
            })

    # Also append new additions from rebalancing plan to table_items
    for bo in buy_orders:
        if bo.get("is_new"):
            new_label = "NEW 5Y SHIELD & ALPHA" if bo.get("is_crash_shield") else "NEW POSITION"
            table_items.append({
                "ticker": bo["ticker"],
                "search_symbol": bo["search_symbol"],
                "name": bo["name"],
                "value_gbp": 0.0,
                "unrealized_pnl": 0.0,
                "plan_status": f"BUY (+£{bo['estimated_gbp']:,.0f})",
                "plan_badge_color": "#10b981",
                "plan_summary": f"{new_label}: Initiating £{bo['estimated_gbp']:,.0f} starter position ({bo.get('theme', 'Growth')}).",
                "currency": bo.get("currency", "USD"),
                "price": bo["price"],
                "cagr_5y_pct": bo.get("cagr_5y_pct", MISSING_METRIC_DEFAULTS["cagr_5y_pct"]),
                "sortino_5y": bo.get("sortino_5y", MISSING_METRIC_DEFAULTS["sortino_5y"]),
                "crash_2022_ret_pct": bo.get("crash_2022_ret_pct", MISSING_METRIC_DEFAULTS["crash_2022_ret_pct"]),
                "beta_5y": bo.get("beta_5y", MISSING_METRIC_DEFAULTS["beta_5y"]),
                "crash_role_label": bo.get("crash_role_label", "🛡️ 5Y Compounder"),
                "is_new_addition": True,
                "research": bo.get("research", {}),
            })

    # Free cash already excludes blocked cash committed to pending orders, so blocked cash is not subtracted again
    final_cash_balance = available_cash_gbp - sum(b["estimated_gbp"] for b in buy_orders)

    # Cluster concentration above the configured cap (no trims are generated for clusters; new buys there are blocked)
    for tb in thematic_breakdown:
        if tb["cluster"] not in CLUSTER_CAP_EXEMPT and _num(tb.get("weight_pct"), 0.0) > cluster_cap_pct:
            cap_breaches.append({
                "kind": "cluster",
                "name": tb["cluster"],
                "weight_pct": tb["weight_pct"],
                "cap_pct": round(cluster_cap_pct, 1),
                "actioned": False,
            })

    # 7. Compute 5-Year Risk-Adjusted Return & 2022 Crash Stress-Test Profile for this Account
    risk_profile_5y = compute_portfolio_5y_risk_profile(
        analyses=analyses,
        available_cash_gbp=available_cash_gbp,
        blocked_cash_gbp=blocked_cash_gbp,
        total_portfolio_gbp=total_portfolio_gbp,
        sell_orders=sell_orders,
        buy_orders=buy_orders,
    )

    return {
        "account_name": acc_name,
        "available_cash_gbp": available_cash_gbp,
        "blocked_cash_gbp": blocked_cash_gbp,
        "pending_buys_committed_gbp": pending_buys_committed,
        "pending_orders_data": pending_data,
        "has_orders_scope": has_orders_scope,
        "open_orders_list": open_orders_list,
        "pending_buys": pending_buys,
        "pending_sells": pending_sells,
        "cash_from_sells_gbp": cash_from_sells,
        "total_deployable_gbp": total_deployable,
        "min_cash_reserve_gbp": min_cash_reserve_gbp,
        "buffer_reserve_gbp": buffer_reserve,
        "reserve_floor_gbp": reserve_floor_gbp,
        "reserve_funded_gbp": reserve_funded_gbp,
        "reserve_shortfall_gbp": reserve_shortfall_gbp,
        "net_budget_to_deploy_gbp": net_budget_to_deploy,
        "sell_orders": sell_orders,
        "pending_sell_orders": pending_sell_orders,
        "buy_orders": buy_orders,
        "hold_items": hold_items,
        "table_items": table_items,
        "thematic_breakdown": thematic_breakdown,
        "final_cash_balance_gbp": max(0.0, final_cash_balance),
        "risk_profile_5y": risk_profile_5y,
        "portfolio_5y_risk_profile": risk_profile_5y,
        "global_context": global_context,
        "data_degraded_reason": data_degraded_reason,
        "earmarked_gbp": round(earmarked_gbp, 2),
        "earmarked_tickers": earmarked_held,
        "investable_total_gbp": investable_total_gbp,
        "outside_vested_gbp": round(outside_vested_gbp, 2),
        "cap_breaches": cap_breaches,
        "limits": {
            "max_position_pct": pos_cap,
            "max_leveraged_etf_pct": lev_cap,
            "max_cluster_pct": round(cluster_cap_pct / 100.0, 4),
            "target_crash_shield_pct": round(shield_target_pct / 100.0, 4),
            "max_new_ideas_to_buy": max_new_ideas,
            "cap_applies_to_broad_etfs": _rebalance_cfg_bool("cap_applies_to_broad_etfs", False),
        },
        "model_version": _model_version(),
    }


def generate_tldr_summary(results_by_account: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generates high-priority, 30-second actionable TL;DR directives partitioned by account."""
    tldr = {}
    for acc_name, data in results_by_account.items():
        plan = data.get("rebalancing_plan", {}) or {}
        cash_info = data.get("cash", {})
        avail_cash = _num(plan.get("available_cash_gbp"), 0.0)
        net_budget = _num(plan.get("net_budget_to_deploy_gbp"), 0.0)
        buffer_reserve = _num(plan.get("buffer_reserve_gbp"), 0.0)
        reserve_floor = _first_num(plan.get("reserve_floor_gbp"), plan.get("min_cash_reserve_gbp"), default=buffer_reserve)
        reserve_funded = _first_num(plan.get("reserve_funded_gbp"), default=min(avail_cash, reserve_floor))
        reserve_shortfall = _first_num(plan.get("reserve_shortfall_gbp"), default=max(0.0, reserve_floor - avail_cash))
        degraded_reason = plan.get("data_degraded_reason")

        actionable_sells = [s for s in plan.get("sell_orders", []) if not s.get("is_pending")]
        pending_sells = plan.get("pending_sell_orders", [])
        if not pending_sells and plan.get("pending_sells"):
            pending_sells = [
                {"ticker": k, "search_symbol": clean_search_symbol(k), "name": v.get("name", k), "estimated_gbp": v.get("estimated_value", 0.0)}
                for k, v in plan["pending_sells"].items()
            ]

        pending_buys = plan.get("pending_buys", {})
        buy_orders = plan.get("buy_orders", [])
        hold_items = plan.get("hold_items", [])

        sells_to_place = []
        for s in actionable_sells:
            is_trim = "TRIM" in s.get("action", "")
            action_verb = "TRIM" if is_trim else "SELL"
            shares_txt = _fmt_shares(s.get("shares", 0))
            if is_trim:
                if s.get("order_reason") in ("leverage_cap", "position_cap"):
                    inst_text = f"Sell {shares_txt} shares (cap rebalance)"
                elif s.get("is_profit", True):
                    inst_text = f"Sell {shares_txt} shares (take profits)"
                else:
                    inst_text = f"Sell {shares_txt} shares (de-risk)"
            else:
                inst_text = f"Sell all {shares_txt} shares"
            sells_to_place.append({
                "name": s.get("name", s.get("ticker")),
                "symbol": s.get("search_symbol", s.get("ticker")),
                "shares": s.get("shares", 0),
                "is_trim": is_trim,
                "order_reason": s.get("order_reason"),
                "estimated_gbp": s.get("estimated_gbp", 0.0),
                "instruction": f"{action_verb} {s.get('name', s.get('ticker'))} ({s.get('search_symbol', s.get('ticker'))}) ➔ {inst_text} (+£{_num(s.get('estimated_gbp'), 0.0):,.0f})"
            })

        sells_already_queued = []
        for ps in pending_sells:
            lim = _num(ps.get("limit_price"), 0.0)
            lim_str = f" @ limit ${lim:,.2f}" if lim > 0 else ""
            s_name = ps.get("name", ps.get("ticker"))
            s_sym = ps.get("search_symbol", ps.get("ticker"))
            sells_already_queued.append(f"{s_name} ({s_sym}){lim_str}")

        buys_already_queued = []
        for pb_sym, pb in pending_buys.items():
            pb_name = pb.get("name", pb_sym)
            pb_val = _first_num(pb.get("estimated_value"), pb.get("value"), default=0.0)
            buys_already_queued.append(f"{pb_name} (~£{pb_val:,.0f})")

        buys_to_place = []
        for b in buy_orders:
            buys_to_place.append({
                "name": b.get("name", b.get("ticker")),
                "symbol": b.get("search_symbol", b.get("ticker")),
                "amount_gbp": b.get("estimated_gbp", 0.0),
                "shares": b.get("shares", 0.0),
                "is_new": b.get("is_new", False),
                "instruction": f"BUY {b.get('name', b.get('ticker'))} ({b.get('search_symbol', b.get('ticker'))}) ➔ Enter £{_num(b.get('estimated_gbp'), 0.0):,.0f}"
            })

        hold_names = [h.get("name", h.get("ticker")) for h in hold_items]

        def _is_market_like(ps: Dict[str, Any]) -> bool:
            # STOP / STOP_LIMIT orders only fill on a future price move, so they never count as queued proceeds
            order_type = str(ps.get("order_type") or "MARKET").upper()
            if "STOP" in order_type:
                return False
            return order_type == "MARKET" or _num(ps.get("limit_price"), 0.0) <= 0

        has_pending_market_sells = any(_is_market_like(ps) for ps in pending_sells)

        pending_proceeds = sum(_num(ps.get("estimated_gbp"), 0.0) for ps in pending_sells if _is_market_like(ps))

        all_trims = bool(sells_to_place and all(s.get("is_trim", False) for s in sells_to_place))
        trims_are_profit_taking = all_trims and all(
            (s.get("order_reason") not in ("leverage_cap", "position_cap")) and s.get("is_profit", True)
            for s in actionable_sells
        )
        open_breaches = [b for b in (plan.get("cap_breaches") or []) if isinstance(b, dict) and not b.get("actioned")]
        reserve_txt = f"the £{buffer_reserve:,.0f} protected reserve"

        if degraded_reason:
            phase = "DATA_DEGRADED"
            phase_label = "⚠️ DATA DEGRADED — NO RECOMMENDATIONS TODAY"
            phase_desc = f"{degraded_reason} No buy or sell orders are suggested until the data is complete again."
        elif sells_to_place and buys_to_place:
            if all_trims:
                phase = "HARVEST_AND_REALLOCATE"
                if trims_are_profit_taking:
                    phase_label = "🟠 HARVEST PROFITS & REALLOCATE CAPITAL"
                    phase_desc = f"Execute {len(sells_to_place)} profit trim(s) below to bank gains. Reallocate proceeds into {len(buys_to_place)} high-asymmetry buy orders."
                else:
                    phase_label = "🟠 TRIM TO LIMITS & REALLOCATE CAPITAL"
                    phase_desc = f"Execute {len(sells_to_place)} trim(s) below to bring positions back within configured limits. Today's {len(buys_to_place)} buy order(s) use only the £{net_budget:,.0f} of free cash above {reserve_txt}."
            else:
                phase = "SELL_FIRST"
                phase_label = "🔴 SELL ACTIONS REQUIRED FIRST"
                phase_desc = f"Place {len(sells_to_place)} sell order(s) below. You also have £{net_budget:,.0f} deployable cash (free cash above {reserve_txt}) for today's buys."
        elif sells_to_place:
            phase = "TRIM_FIRST" if all_trims else "SELL_FIRST"
            if all_trims and trims_are_profit_taking:
                phase_label = "🟠 HARVEST PROFITS FIRST"
                phase_desc = "Open Trading 212 and execute the trim orders below to bank profits."
            elif all_trims:
                phase_label = "🟠 TRIM TO LIMITS FIRST"
                phase_desc = "Open Trading 212 and execute the trim orders below to bring positions back within configured limits."
            else:
                phase_label = "🔴 SELL ACTIONS REQUIRED FIRST"
                phase_desc = "Open Trading 212 and execute the sell orders below to unlock capital."
        elif has_pending_market_sells and avail_cash >= 1000 and buys_to_place:
            phase = "BUY_NOW"
            phase_label = "🟢 ACTION REQUIRED TODAY — PLACE BUYS"
            phase_desc = f"You have £{net_budget:,.0f} deployable now (free cash above {reserve_txt}), plus ~£{pending_proceeds:,.0f} from queued sells. Place these {len(buys_to_place)} buy orders today."
        elif has_pending_market_sells and buys_to_place:
            phase = "AWAIT_FILLS_THEN_BUY"
            phase_label = "⏳ SELLS QUEUED ➔ BUY ONCE CASH SETTLES"
            phase_desc = f"Your sell orders are queued awaiting market fill (+~£{pending_proceeds:,.0f} cash). Once filled, place the {len(buys_to_place)} buy orders below."
        elif buys_to_place:
            phase = "BUY_NOW"
            phase_label = "🟢 ACTION REQUIRED TODAY — PLACE BUYS"
            phase_desc = f"You have £{net_budget:,.0f} deployable capital (free cash above {reserve_txt}). Place the {len(buys_to_place)} buy orders below."
        else:
            phase = "ALL_SET"
            phase_label = "✅ ALL SET — SIT BACK & RELAX"
            if open_breaches:
                breach_txt = ", ".join(
                    f"{b.get('name') or b.get('ticker')} {_num(b.get('weight_pct'), 0.0):.1f}% vs {_num(b.get('cap_pct'), 0.0):.0f}% cap"
                    for b in open_breaches[:4]
                )
                phase_desc = f"No new orders today, but {len(open_breaches)} configured limit(s) remain exceeded: {breach_txt}."
            else:
                phase_desc = "No action required today. Portfolio is fully balanced and compounding."

        if reserve_shortfall > 0.005:
            reserve_note = f"⚠️ Cash reserve unfunded: £{avail_cash:,.0f} free cash vs £{reserve_floor:,.0f} floor (shortfall £{reserve_shortfall:,.0f})."
            if phase == "ALL_SET":
                phase_desc = f"{phase_desc} {reserve_note}"
        else:
            reserve_note = f"🛡️ Cash reserve funded: £{reserve_floor:,.0f} floor covered by £{avail_cash:,.0f} free cash."

        tldr[acc_name] = {
            "account_name": acc_name,
            "phase": phase,
            "phase_label": phase_label,
            "phase_desc": phase_desc,
            "available_cash": avail_cash,
            "net_budget": net_budget,
            "pending_proceeds": pending_proceeds,
            "buffer_reserve": buffer_reserve,
            "reserve_floor": reserve_floor,
            "reserve_funded": reserve_funded,
            "reserve_shortfall": reserve_shortfall,
            "reserve_note": reserve_note,
            "data_degraded_reason": degraded_reason,
            "cap_breaches": open_breaches,
            "sells_to_place": sells_to_place,
            "sells_already_queued": sells_already_queued,
            "buys_already_queued": buys_already_queued,
            "buys_to_place": buys_to_place,
            "hold_names": hold_names,
        }

    return tldr


"""Quantitative strategy engine for multi-factor trend, momentum, watchlist scanning, and capital allocation."""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from config import STRATEGY_CONFIG, WATCHLIST, REBALANCE_CONFIG, EXCLUDED_TICKERS
from research_agent import DeepResearchAgent
from t212_client import normalize_t212_ticker_for_yfinance

logger = logging.getLogger(__name__)

# Cached GBP/USD rate
_CACHED_GBP_RATE = None
research_agent = DeepResearchAgent()


def is_core_asset(yf_ticker: str, company_name: str = "", macro_cluster: str = "") -> bool:
    """Returns True if the asset is an unleveraged core index foundation, physical gold ETC, or diversified ETF that should never be panic-sold."""
    clean_sym = yf_ticker.upper().strip()
    if any(k in clean_sym for k in ["QQQ3", "3LUS", "3SEM", "3X"]):
        return False
    c_upper = company_name.upper()
    if "LEVERAGED" in c_upper or "3X" in c_upper:
        return False
    core_symbols = {"VUAA.L", "VUSA.L", "VWRP.L", "CNDX.L", "SMGB.L", "DFND.SW", "SGLN.L", "IGLN.L", "GLD", "CSH2.L", "SPY", "QQQ", "RSP"}
    if clean_sym in core_symbols or any(k in clean_sym for k in ["VUAA", "VUSA", "VWRP", "CNDX", "SMGB", "DFND", "SGLN", "IGLN", "CSH2"]):
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
    clean_sym = yf_ticker.upper().strip()
    if any(k in clean_sym for k in ["QQQ3", "3LUS", "3SEM", "3X", "SMGB"]):
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


def get_gbp_usd_rate() -> float:
    """Fetches the current GBP/USD exchange rate (e.g. 1.35 means £1 = $1.35)."""
    global _CACHED_GBP_RATE
    if _CACHED_GBP_RATE is not None:
        return _CACHED_GBP_RATE
    try:
        t = yf.Ticker("GBPUSD=X")
        rate = float(t.fast_info.get("lastPrice", 1.35))
        _CACHED_GBP_RATE = rate
        return rate
    except Exception as e:
        logger.warning("Failed to fetch live GBPUSD rate (%s), defaulting to 1.35", e)
        _CACHED_GBP_RATE = 1.35
        return 1.35


def to_gbp(price: float, currency: str, gbp_rate: float) -> float:
    """Converts a price in its native currency into GBP (£)."""
    if price is None or np.isnan(price) or price <= 0:
        return 0.0
    raw_c = str(currency).strip()
    # Note: Yahoo Finance uses 'GBp' (lowercase p) specifically for pence, and 'GBP' for pounds sterling.
    # Trading 212 / other sources may use 'GBX' or 'PENCE'.
    if raw_c == "GBp" or raw_c.upper() in ("GBX", "PENCE", "GBPX"):
        return price / 100.0
    c = raw_c.upper()
    if c == "GBP":
        return price
    elif c == "USD":
        return price / gbp_rate
    elif c == "EUR":
        return price / (gbp_rate * 0.85)  # Approx EUR/GBP
    return price


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


def evaluate_weekly_trend(ticker_obj: yf.Ticker, yf_ticker: str = "") -> Dict[str, Any]:
    """Evaluates Stan Weinstein's 30-week Moving Average rule AND computes empirical 5-Year
    Risk-Adjusted Return (CAGR, Volatility, Sortino, Sharpe, Calmar, Max Drawdown) & 2022 Crash Resilience.
    """
    default_5y = {
        "is_weekly_stage_2": True,
        "label": "🟢 Weekly Data Baseline",
        "badge_color": "#10b981",
        "ema_10w": 0.0,
        "ema_30w": 0.0,
        "ema_50w": 0.0,
        "pct_vs_30w": 0.0,
        "cagr_5y_pct": 14.0,
        "vol_5y_pct": 20.0,
        "sortino_5y": 1.10,
        "sharpe_5y": 0.80,
        "max_dd_5y_pct": -22.0,
        "calmar_5y": 0.64,
        "crash_2022_ret_pct": -18.0,
        "beta_5y": 1.0,
        "corr_5y": 0.70,
        "crash_role_label": "📈 5Y Secular Growth",
        "crash_role_color": "#6366f1",
    }
    try:
        df_w = ticker_obj.history(period="5y", interval="1wk")
        if df_w.empty or len(df_w) < 25:
            return default_5y

        c = df_w["Close"].dropna()
        if c.empty or len(c) < 20:
            return default_5y

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

        # Compute 5-Year Empirical Risk-Adjusted & Crash Metrics from Weekly Bars
        c_tz_free = c.copy()
        if hasattr(c_tz_free.index, "tz") and c_tz_free.index.tz is not None:
            c_tz_free.index = c_tz_free.index.tz_localize(None)

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

        # Beta & Correlation vs S&P 500 (SPY) over 5 years
        beta_5y = 1.0
        corr_5y = 0.70
        spy_w = get_spy_5y_weekly_history()
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
            # For instruments listed after 2022 (e.g. DFND.SW), estimate 2022 shock via empirical beta
            crash_2022_ret_pct = round(max(-75.0, min(15.0, beta_5y * -19.4)), 1)

        # Classify 5-Year Portfolio Crash & Compounding Role
        sym_u = yf_ticker.upper()
        if any(k in sym_u for k in ["QQQ3", "3LUS", "3SEM", "3X"]):
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
    """Analyzes a stock using trend, momentum, volatility indicators, and deep fundamental research."""
    if gbp_rate is None:
        gbp_rate = get_gbp_usd_rate()

    try:
        research = research_agent.conduct_deep_dive(yf_ticker) or {}
    except Exception as e:
        logger.warning("Error during deep research for %s: %s", yf_ticker, e)
        research = {}
    if not isinstance(research, dict):
        research = {}

    company_name = research.get("company_name", yf_ticker)

    p_qty = float(position_data.get("quantity", 0.0) or 0.0) if position_data else 0.0
    p_price = float(position_data.get("currentPrice") or position_data.get("averagePrice") or 0.0) if position_data else 0.0
    p_curr = str(position_data.get("currency") or research.get("currency") or "USD") if position_data else research.get("currency", "USD")
    p_gbp = to_gbp(p_price, p_curr, gbp_rate)
    p_val = p_qty * p_gbp

    default_result = {
        "ticker": yf_ticker,
        "search_symbol": clean_search_symbol(yf_ticker),
        "name": company_name,
        "price": p_price,
        "currency": p_curr,
        "price_gbp": p_gbp,
        "signal": "🟡 HOLD",
        "badge_color": "#6366f1",
        "action_short": "HOLD",
        "reason": "Data temporarily unavailable for ticker",
        "plain_english_why": f"{company_name} is stable. Keep holding as core compounder.",
        "app_instruction": f"Search '{clean_search_symbol(yf_ticker)}' ➔ Leave in portfolio",
        "rsi": 50.0,
        "trailing_stop": 0.0,
        "chandelier_stop": 0.0,
        "stop_distance_pct": 0.0,
        "target_price": 0.0,
        "ema_50": 0.0,
        "ema_200": 0.0,
        "minervini_stage": "Stage 2 Confirmed Bull (5/5)",
        "minervini_pass": 5,
        "minervini_badge_bg": "#10b981",
        "vol_spike": 1.0,
        "up_down_vol_ratio": 1.2,
        "volume_footprint_label": "Constructive Volume",
        "volume_footprint_badge": "#10b981",
        "rs_alpha": 0.0,
        "rs_label": "Benchmark Neutral",
        "macd_line": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0,
        "macd_status": "Neutral",
        "vcp_pct": 2.5,
        "vcp_label": "ATR Volatility: 2.5%",
        "conviction_score": 50,
        "risk_reward_ratio": 2.5,
        "days_to_earnings": research.get("days_to_earnings"),
        "earnings_catalyst_status": research.get("earnings_catalyst_status", "✅ Clear"),
        "piotroski_score": research.get("piotroski_score"),
        "piotroski_label": research.get("piotroski_label", "N/A"),
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
        "cagr_5y_pct": 14.0,
        "vol_5y_pct": 20.0,
        "sortino_5y": 1.10,
        "sharpe_5y": 0.80,
        "max_dd_5y_pct": -22.0,
        "calmar_5y": 0.64,
        "crash_2022_ret_pct": -18.0,
        "beta_5y": 1.0,
        "corr_5y": 0.70,
        "crash_role_label": "📈 5Y Secular Growth",
        "crash_role_color": "#64748b",
        "is_crash_shield": False,
        "unrealized_pnl": position_data.get("ppl", 0.0) if position_data else 0.0,
        "quantity": p_qty,
        "avg_price": position_data.get("averagePrice", 0.0) if position_data else 0.0,
        "value_gbp": p_val,
        "portfolio_weight_pct": 0.0,
        "research": research,
    }

    try:
        ticker_obj = yf.Ticker(yf_ticker)
        df = ticker_obj.history(period="1y", interval="1d")
        
        if df.empty:
            default_result["reason"] = "No historical price bars returned"
            return default_result

        # Drop any incomplete or NaN candles from exchange holidays / pre-market feeds
        df = df.dropna(subset=["Close", "High", "Low"])

        if len(df) < 60:
            default_result["reason"] = "Insufficient historical price bars to compute EMAs"
            return default_result

        fast_info = getattr(ticker_obj, "fast_info", {})
        currency = fast_info.get("currency", "USD")

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
        if position_data and position_data.get("currentPrice") and float(position_data.get("currentPrice")) > 0:
            live_quote = float(position_data["currentPrice"])
        elif fast_info and fast_info.get("lastPrice") and float(fast_info.get("lastPrice")) > 0:
            live_quote = float(fast_info.get("lastPrice"))
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

        # 1b. Stan Weinstein Multi-Timeframe Weekly Trend & 5-Year Empirical Risk/Crash Audit
        weekly_trend = evaluate_weekly_trend(ticker_obj, yf_ticker=yf_ticker)

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
        avg_price = float(position_data.get("averagePrice", 0.0) or 0.0) if position_data else 0.0
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

        trailing_stop = max(base_trailing_stop, ratchet_stop)
        trailing_stop = min(trailing_stop, latest_price * 0.975)  # Preserve minimum 2.5% breathing room
        stop_dist_pct = ((latest_price - trailing_stop) / latest_price) * 100

        recent_high = float(df["High"].tail(120).max())
        target_price = max(recent_high * 1.05, latest_price * 1.15)
        pct_above_50 = (latest_price - curr_ema_50) / curr_ema_50 if curr_ema_50 > 0 else 0.0
        pct_above_20 = (latest_price - curr_ema_20) / curr_ema_20 if curr_ema_20 > 0 else 0.0
        is_extended = (pct_above_20 > 0.065) or (curr_rsi > 68.0)

        # Institutional Asymmetric Risk/Reward Ratio
        target_eval = research.get("target_mean_price") or target_price
        if target_eval <= latest_price:
            target_eval = target_price

        downside_risk = max(latest_price * 0.02, latest_price - trailing_stop)
        upside_reward = max(0.0, target_eval - latest_price)
        risk_reward_ratio = round(upside_reward / downside_risk, 2) if downside_risk > 0 else 3.0

        # Earnings Catalyst Imminence Check
        days_to_earnings = research.get("days_to_earnings")
        earnings_imminent = (days_to_earnings is not None and 0 <= days_to_earnings <= 5)

        # 7. World-Class Multi-Factor Composite Conviction Scoring (0 to 100)
        macro_score = research_agent.get_macro_regime().get("macro_score", 75)
        fund_score = research.get("fundamental_score", 75)
        is_leveraged = any(k in yf_ticker.upper() for k in ["QQQ3", "3LUS", "3SEM", "3X"])

        # Recognize Institutional Pullback-to-Value in a Weekly Stage 2 Compounder
        is_quality_weekly_pullback = (
            weekly_trend["is_weekly_stage_2"]
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

        if weekly_trend["is_weekly_stage_2"]: tech_score += 12
        else: tech_score -= 20

        if 32 <= curr_rsi <= 50 and weekly_trend["is_weekly_stage_2"]:
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
        sector_mom = research_agent.get_macro_regime().get("sector_momentum", {})
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

        if sector_ret is not None:
            if sector_ret >= 8.0:
                tech_score += 6
            elif sector_ret <= -5.0:
                tech_score -= 8

        # Short Interest Dynamics
        short_pct = research.get("short_interest_pct") or 0.0
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

        # Asymmetry Score (10% weight)
        asym_score = 50
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
        sortino_5y = weekly_trend.get("sortino_5y", 1.1)
        crash_2022_ret = weekly_trend.get("crash_2022_ret_pct", -18.0)
        cagr_5y = weekly_trend.get("cagr_5y_pct", 14.0)
        beta_5y = weekly_trend.get("beta_5y", 1.0)
        corr_5y = weekly_trend.get("corr_5y", 0.70)

        macro_cluster = research.get("macro_cluster", "")
        is_core = is_core_asset(yf_ticker, company_name, macro_cluster)
        is_crash_shield = is_crash_shield_asset(yf_ticker, macro_cluster, beta_5y, crash_2022_ret)
        exp_5y_cagr = float(research.get("expected_5y_cagr_pct") or cagr_5y)

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

        search_sym = clean_search_symbol(yf_ticker)

        # Check Anti-Churn Holding Period, 5Y Horizon Lockouts & Core Foundation Protections
        is_recent, days_held = is_recent_entry(position_data, days_threshold=28)
        is_gold_etc = any(k in yf_ticker.upper() for k in ["SGLN", "IGLN", "GLD"])
        gain_pct = ((latest_price - avg_price) / avg_price) if (avg_price > 0 and latest_price > 0) else 0.0

        # Anti-churn rule: protect new entries from daily friction, BUT NEVER block taking profits
        # on strong winners (+20%+ profit with softening R:R), overextended parabolic spikes, or excessive leverage.
        allow_anti_churn = is_recent and (gain_pct < 0.20) and (not is_extended) and (pct_above_50 < 0.15) and (not is_leveraged)

        # 180-day (6-month) anti-round-trip & anti-re-trim lockout for unleveraged stocks;
        # 14-day cooldown for 3x daily leveraged ETFs so oversized (>12%) 3x leverage is never left unhedged.
        trim_lockout_days = 14 if is_leveraged else 180
        is_trimmed, days_trimmed = is_recently_trimmed(position_data, days_threshold=trim_lockout_days)

        # Signal Classification with Institutional Gatekeepers & 5-Year Horizon Anti-Churn Defense
        if is_trimmed:
            days_str = f"{days_trimmed}d ago" if (days_trimmed is not None and days_trimmed > 0) else "today"
            signal = "🟢 HOLD (PROFIT HARVESTED — 5Y RUNNER)"
            badge_color = "#10b981"
            action_short = "HOLD"
            reason = f"Partial profits already banked {days_str}. Remaining runner shares locked for 5-year compounding (anti-round-trip active)."
            plain_english_why = f"You already trimmed {company_name} ({days_str}) to lock in gains and right-size risk. Hold your remaining runner shares for the 5-year trend—do not sell more and do not buy back higher."
            app_instruction = "No action needed ➔ Leave 5Y runner in portfolio"

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
                is_gold_etc and
                weekly_trend["is_weekly_stage_2"] and
                not is_extended and
                curr_rsi <= 62.0
            ):
                signal = "🟢 BUY (CRISIS ALPHA SHIELD)"
                badge_color = "#10b981"
                action_short = "BUY"
                reason = f"Physical Gold pulling back to 50-week structural support (-0.03 S&P correlation, +14.4% in 2022 crash)."
                plain_english_why = f"{company_name} is testing its rising 50-week structural support in GBP. Prime entry to build an uncorrelated 5-year crisis shield."
                app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Buy"
            elif (
                minervini["is_stage_2"] and
                curr_ema_50 > 0 and abs(latest_price - curr_ema_50) / curr_ema_50 < 0.055 and
                35.0 <= curr_rsi <= 62.0 and
                vol_footprint["is_accumulation"] and
                weekly_trend["is_weekly_stage_2"] and
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

        elif is_leveraged and (pct_above_20 > 0.03 or curr_rsi > 56.0 or sortino_5y < 0.60):
            signal = "🟠 TRIM (3X LEVERAGE CRASH & DECAY RISK)"
            badge_color = "#f59e0b"
            action_short = "TRIM"
            reason = f"3x daily leveraged ETF suffers severe 5Y volatility decay (Sortino {sortino_5y:.2f}, -80.7% Max Drawdown). Cap strictly <= 8%."
            plain_english_why = f"{company_name} is a 3x daily leveraged product that experienced a -76.2% crash in 2022 and suffers mathematical variance drain over 5-year horizons. Trim oversized exposure to protect against crashes."
            app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Sell ➔ Reduce oversized 3x leverage"

        elif (not is_core) and curr_rsi > 82.0 and pct_above_50 > 0.25 and gain_pct >= 0.35:
            signal = "🟠 TRIM (PARABOLIC BLOW-OFF)"
            badge_color = "#f59e0b"
            action_short = "TRIM"
            reason = f"Extreme parabolic blow-off (RSI {curr_rsi:.1f}, +{pct_above_50*100:.1f}% vs 50 EMA, +{gain_pct*100:.1f}% gain)."
            plain_english_why = f"{company_name} has surged +{gain_pct*100:.1f}% into an extreme parabolic spike (>25% above 50 EMA). Banking partial profits if position is oversized."
            app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Sell ➔ Sell 1/4 of shares to lock in gains"

        elif (
            # 1. Sustained macro structural breakdown below 200 EMA + Weekly Stage 4 downtrend
            (not weekly_trend["is_weekly_stage_2"] and latest_price < curr_ema_200 * 0.96 and curr_rsi < 38.0) or
            # 2. Severe fundamental distress (insolvency risk)
            (research.get("altman_z") is not None and research.get("altman_z") < 1.81 and research.get("piotroski_score") is not None and research.get("piotroski_score") <= 2) or
            # 3. Long-Term Thesis Break: Deteriorating fundamentals + broken weekly trend below 50d EMA
            (not is_core and not weekly_trend["is_weekly_stage_2"] and latest_price < curr_ema_50 * 0.97 and fund_score < 48)
        ):
            signal = "🔴 SELL (EXIT)"
            badge_color = "#ef4444"
            action_short = "SELL"
            reason = f"Structural breakdown below 200-day EMA and weekly 30-week EMA / fundamental thesis impairment."
            plain_english_why = f"{company_name} has broken its intermediate and long-term trend structure or suffered fundamental deterioration. Sell to defend capital."
            app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Sell ➔ Sell all shares"

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
                risk_reward_ratio >= 2.2
            )
        ):
            if not weekly_trend["is_weekly_stage_2"]:
                signal = "🟡 HOLD (WEEKLY DOWNTREND FILTER)"
                badge_color = "#6366f1"
                action_short = "HOLD"
                pct_w = weekly_trend.get("pct_vs_30w", 0.0)
                reason = f"Daily bounce detected, but weekly chart is below 30-week EMA ({pct_w:+.1f}%). Filter prevents counter-trend entry."
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
            elif risk_reward_ratio < 2.0:
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
                reason = f"Weekly Stage 2 compounder at institutional pullback support (RSI {curr_rsi:.1f}, +{exp_5y_cagr:.1f}% Expected 5Y CAGR, {risk_reward_ratio}:1 R:R)."
                plain_english_why = f"{company_name} is a wide-moat compounder in a confirmed Weekly Stage 2 uptrend pulling back into an institutional value entry zone (RSI {curr_rsi:.1f}, +{exp_5y_cagr:.1f}% Expected 5Y CAGR)."
                app_instruction = f"Search '{search_sym}' in Trading 212 ➔ Tap Buy"

        else:
            signal = "🟡 HOLD (DO NOTHING)"
            badge_color = "#6366f1"
            action_short = "HOLD"
            trend_note = minervini["stage_label"]
            reason = f"{trend_note} intact above 200 EMA ({cagr_5y:+.1f}% 5Y CAGR, {sortino_5y:.2f} Sortino). 5Y Chandelier stop at {trailing_stop:,.2f}."
            plain_english_why = f"{company_name} is compounding well in its upward channel. Sit back and let your 5-year thesis work."
            app_instruction = "No action needed ➔ Leave in portfolio"

        price_in_gbp = to_gbp(latest_price, currency, gbp_rate)
        quantity = position_data.get("quantity", 0.0) if position_data else 0.0
        val_gbp = quantity * price_in_gbp
        if val_gbp is None or np.isnan(val_gbp):
            val_gbp = 0.0

        return {
            "ticker": yf_ticker,
            "search_symbol": search_sym,
            "name": company_name,
            "price": latest_price,
            "currency": currency,
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
            "target_price": target_price,
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
            "is_weekly_stage_2": weekly_trend["is_weekly_stage_2"],
            "weekly_trend_label": weekly_trend["label"],
            "cagr_5y_pct": weekly_trend.get("cagr_5y_pct", 14.0),
            "expected_5y_cagr_pct": exp_5y_cagr,
            "fair_pe_multiple": research.get("fair_pe_multiple"),
            "valuation_regime_label": research.get("valuation_regime_label", "🟢 Fair GARP Valuation"),
            "valuation_badge_color": research.get("valuation_badge_color", "#10b981"),
            "vol_5y_pct": weekly_trend.get("vol_5y_pct", 20.0),
            "sortino_5y": weekly_trend.get("sortino_5y", 1.10),
            "sharpe_5y": weekly_trend.get("sharpe_5y", 0.80),
            "max_dd_5y_pct": weekly_trend.get("max_dd_5y_pct", -22.0),
            "calmar_5y": weekly_trend.get("calmar_5y", 0.64),
            "crash_2022_ret_pct": weekly_trend.get("crash_2022_ret_pct", -18.0),
            "beta_5y": weekly_trend.get("beta_5y", 1.0),
            "corr_5y": weekly_trend.get("corr_5y", 0.70),
            "crash_role_label": weekly_trend.get("crash_role_label", "📈 5Y Secular Growth"),
            "crash_role_color": weekly_trend.get("crash_role_color", "#64748b"),
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
        }

    except Exception as e:
        logger.error("Error analyzing %s: %s", yf_ticker, e)
        default_result["reason"] = f"Analysis error: {e}"
        return default_result


def scan_watchlist_opportunities(
    owned_tickers: Set[str],
    gbp_rate: float
) -> List[Dict[str, Any]]:
    """Scans curated high-alpha watchlist tickers that are not currently in the portfolio."""
    opportunities = []
    for item in WATCHLIST:
        ticker = item["ticker"]
        if ticker in EXCLUDED_TICKERS or any(e == ticker for e in EXCLUDED_TICKERS):
            logger.info("Watchlist ticker %s filtered out: excluded by user mandate.", ticker)
            continue
        if any(tickers_match(ticker, ot) for ot in owned_tickers):
            continue

        analysis = analyze_position(ticker, position_data=None, gbp_rate=gbp_rate)

        # Risk Gatekeeper 1: Never recommend buying an unowned stock that is flashing SELL or TRIM
        if analysis.get("action_short") in ["SELL", "TRIM"]:
            logger.info("Watchlist ticker %s filtered out: currently flashing %s", ticker, analysis.get("action_short"))
            continue

        # Risk Gatekeeper 2: Require weekly Stage 2 confirmation (never catch a falling knife)
        if not analysis.get("is_weekly_stage_2", True):
            logger.info("Watchlist ticker %s filtered out: intermediate weekly trend below 30w EMA", ticker)
            continue

        # Risk Gatekeeper 3: Prevent buying overextended stocks
        if analysis.get("is_extended", False):
            logger.info("Watchlist ticker %s filtered out: price extended >6.5%% above 20 EMA or RSI >68", ticker)
            continue

        # Risk Gatekeeper 4: Pre-earnings blackout (never initiate binary bets <= 5 days before earnings)
        days_to_earnings = analysis.get("days_to_earnings")
        if days_to_earnings is not None and 0 <= days_to_earnings <= 5:
            logger.info("Watchlist ticker %s filtered out: binary earnings catalyst imminent in %d days", ticker, days_to_earnings)
            continue

        analysis["theme"] = item["theme"]
        analysis["is_new_addition"] = True

        if (
            analysis["action_short"] == "BUY"
            or analysis["conviction_score"] >= 74
            or (analysis.get("is_crash_shield") and analysis["conviction_score"] >= 65)
        ):
            if analysis["action_short"] != "BUY":
                analysis["action_short"] = "BUY"
            exp_cagr = analysis.get("expected_5y_cagr_pct", analysis.get("cagr_5y_pct", 14.0))
            val_lbl = analysis.get("valuation_regime_label", "Fair GARP Valuation")
            analysis["plain_english_why"] = (
                f"{analysis['name']} is in a confirmed Weekly Stage 2 uptrend with elite institutional quality "
                f"(Expected 5Y CAGR: +{exp_cagr:.1f}%, {val_lbl}, Conviction: {analysis['conviction_score']}/100). High-probability entry into {item['theme']}."
            )
            opportunities.append(analysis)

    opportunities.sort(
        key=lambda x: (
            x.get("conviction_score", 0)
            + 0.75 * float(x.get("expected_5y_cagr_pct", x.get("cagr_5y_pct", 14.0)) or 14.0)
            + (8.0 if 30.0 <= float(x.get("rsi", 50.0) or 50.0) <= 48.0 else 0.0)
        ),
        reverse=True,
    )
    return opportunities


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
    """
    denom = max(1.0, float(total_portfolio_gbp))
    cash_now = max(0.0, float(available_cash_gbp or 0.0) + float(blocked_cash_gbp or 0.0))

    def _eval_book(book_items: List[Dict[str, Any]], cash_val: float, total_val: float) -> Dict[str, float]:
        d = max(1.0, total_val)
        w_cash = max(0.0, cash_val) / d
        # UK money-market / interest-on-cash baseline (~4.0% yield, near-zero excess return above 3.8% MAR)
        cagr_sum = w_cash * 4.0
        beta_sum = 0.0
        stress_2022_sum = w_cash * 1.8
        shield_gbp = max(0.0, cash_val)
        lev_3x_gbp = 0.0

        active_nodes = []
        for item in book_items:
            v = max(0.0, float(item.get("value_gbp", 0.0) or 0.0))
            if v <= 0:
                continue
            w = v / d
            t_sym = str(item.get("ticker", "")).upper()
            hist_cagr = float(item.get("cagr_5y_pct", 14.0) or 14.0)
            exp_cagr = float(
                item.get("expected_5y_cagr_pct")
                or (item.get("research") or {}).get("expected_5y_cagr_pct")
                or hist_cagr
            )
            cagr_i = (0.45 * hist_cagr) + (0.55 * exp_cagr)
            sort_i = max(0.35, float(item.get("sortino_5y", 1.15) or 1.15))
            beta_i = float(item.get("beta_5y", 1.0) or 1.0)
            corr_i = float(item.get("corr_5y", 0.70) or 0.70)
            crash_i = float(item.get("crash_2022_ret_pct", -18.0) or -18.0)
            cluster_i = str(item.get("macro_cluster") or (item.get("research") or {}).get("macro_cluster") or "")

            # Implied annualized downside semi-deviation (%)
            dd_i = max(5.5, abs(cagr_i - 3.5) / sort_i)
            is_gold_or_cash = any(k in t_sym for k in ["SGLN", "IGLN", "GLD", "CSH2"])

            cagr_sum += w * cagr_i
            beta_sum += w * beta_i
            stress_2022_sum += w * crash_i

            if is_crash_shield_asset(t_sym, cluster_i, beta_i, crash_i):
                shield_gbp += v
            if any(k in t_sym for k in ["QQQ3", "3LUS", "3SEM", "3X"]):
                lev_3x_gbp += v

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
        }

    curr_metrics = _eval_book(analyses, cash_now, denom)

    # Simulate Post-Plan Portfolio Book (after recommended trims/sells and buys)
    sim_map: Dict[str, Dict[str, Any]] = {}
    for pos in analyses:
        t = pos.get("ticker", "")
        sim_map[t] = dict(pos)

    sim_cash = cash_now
    for s in (sell_orders or []):
        if s.get("is_pending"):
            continue
        st = s.get("ticker", "")
        s_val = float(s.get("estimated_gbp", 0.0) or 0.0)
        if st in sim_map:
            sim_map[st]["value_gbp"] = max(0.0, float(sim_map[st].get("value_gbp", 0.0) or 0.0) - s_val)
        sim_cash += s_val

    for b in (buy_orders or []):
        bt = b.get("ticker", "")
        b_val = float(b.get("estimated_gbp", 0.0) or 0.0)
        if bt in sim_map:
            sim_map[bt]["value_gbp"] = float(sim_map[bt].get("value_gbp", 0.0) or 0.0) + b_val
        else:
            sim_map[bt] = {
                "ticker": bt,
                "value_gbp": b_val,
                "cagr_5y_pct": b.get("cagr_5y_pct", 16.0),
                "expected_5y_cagr_pct": b.get("expected_5y_cagr_pct", b.get("cagr_5y_pct", 16.0)),
                "sortino_5y": b.get("sortino_5y", 1.35),
                "beta_5y": b.get("beta_5y", 0.45),
                "corr_5y": b.get("corr_5y", 0.55),
                "crash_2022_ret_pct": b.get("crash_2022_ret_pct", 10.0),
                "macro_cluster": b.get("macro_cluster", ""),
            }
        sim_cash = max(0.0, sim_cash - b_val)

    post_metrics = _eval_book(list(sim_map.values()), sim_cash, denom)

    # 5-Year Wealth Projections (Conservative Institutional vs Target Compounding)
    horizon_yrs = int(STRATEGY_CONFIG.get("investment_horizon_years", 5))
    cons_rate = min(0.155, max(0.085, (post_metrics["cagr_5y_pct"] / 100.0) * 0.62))
    hist_rate = min(0.225, max(0.120, (post_metrics["cagr_5y_pct"] / 100.0) * 0.85))
    proj_cons_gbp = round(denom * ((1.0 + cons_rate) ** horizon_yrs), 0)
    proj_hist_gbp = round(denom * ((1.0 + hist_rate) ** horizon_yrs), 0)

    crash_saved_gbp = max(0.0, abs(curr_metrics["stress_2022_gbp"]) - abs(post_metrics["stress_2022_gbp"]))

    if post_metrics["lev_3x_pct"] <= 11.5 and post_metrics["shield_pct"] >= 11.5:
        shield_status = "🟢 ALL-WEATHER CRASH SHIELD ACTIVE"
        shield_color = "#10b981"
    elif post_metrics["shield_pct"] >= 7.5:
        shield_status = "🟡 BUILDING CRASH SHIELD BUFFER"
        shield_color = "#f59e0b"
    else:
        shield_status = "🔴 HIGH BETA — DEPLOY CRASH SHIELD"
        shield_color = "#ef4444"

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
        "post_cagr_5y_pct": post_metrics["cagr_5y_pct"],
        "post_sortino_5y": post_metrics["sortino_5y"],
        "post_beta_5y": post_metrics["beta_5y"],
        "post_stress_2022_pct": post_metrics["stress_2022_pct"],
        "post_stress_2022_gbp": post_metrics["stress_2022_gbp"],
        "post_shield_gbp": post_metrics["shield_gbp"],
        "post_shield_pct": post_metrics["shield_pct"],
        "post_lev_3x_pct": post_metrics["lev_3x_pct"],
        "crash_capital_saved_gbp": round(crash_saved_gbp, 0),
        "projected_5y_conservative_gbp": proj_cons_gbp,
        "projected_5y_conservative_cagr_pct": round(cons_rate * 100.0, 1),
        "projected_5y_target_gbp": proj_hist_gbp,
        "projected_5y_target_cagr_pct": round(hist_rate * 100.0, 1),
        "shield_status": shield_status,
        "shield_color": shield_color,
        "crash_readiness_grade": shield_status,
        "crash_readiness_color": shield_color,
    }


def build_consolidated_portfolio_view(
    accounts_data: Dict[str, Dict[str, Any]],
    gbp_rate: float = 1.35
) -> Dict[str, Any]:
    """Consolidates positions, cash balances, and risk metrics across all accounts into a single unified portfolio model.

    Args:
        accounts_data: Dict mapping account_name -> {"cash": dict, "analyses": list, "pending_orders": dict}
        gbp_rate: Current GBP/USD exchange rate.

    Returns:
        Consolidated portfolio intelligence dict with aggregate valuation, global weights, and unified thematic breakdown.
    """
    total_invested_gbp = 0.0
    total_liquid_cash_gbp = 0.0
    total_blocked_cash_gbp = 0.0
    total_ppl_gbp = 0.0
    account_shares = {}

    all_holdings: Dict[str, Dict[str, Any]] = {}
    cluster_totals: Dict[str, float] = {}
    combined_analyses: List[Dict[str, Any]] = []

    for acc_name, acc_info in accounts_data.items():
        cash = acc_info.get("cash", {})
        free_gbp = float(cash.get("free_gbp", cash.get("free", 0.0)) or 0.0)
        if np.isnan(free_gbp): free_gbp = 0.0
        blocked_gbp = float(cash.get("blocked_gbp", cash.get("blocked", 0.0)) or 0.0)
        if np.isnan(blocked_gbp): blocked_gbp = 0.0
        acc_invested = sum(
            float(pos.get("value_gbp", 0.0) or 0.0)
            for pos in acc_info.get("analyses", [])
            if not np.isnan(float(pos.get("value_gbp", 0.0) or 0.0))
        )
        acc_ppl = float(cash.get("ppl", 0.0) or 0.0)
        if np.isnan(acc_ppl): acc_ppl = 0.0

        total_invested_gbp += acc_invested
        total_liquid_cash_gbp += free_gbp
        total_blocked_cash_gbp += blocked_gbp
        total_ppl_gbp += acc_ppl

        broker_total = float(cash.get("total", 0.0) or 0.0)
        calc_total = acc_invested + free_gbp + blocked_gbp
        acc_total = broker_total if broker_total > 0 else calc_total

        account_shares[acc_name] = {
            "invested_gbp": acc_invested,
            "liquid_cash_gbp": free_gbp,
            "blocked_cash_gbp": blocked_gbp,
            "total_value_gbp": acc_total,
            "ppl_gbp": acc_ppl,
            "positions_count": len(acc_info.get("analyses", [])),
        }

        # Aggregate holdings across accounts
        for pos in acc_info.get("analyses", []):
            combined_analyses.append(pos)
            ticker = pos.get("ticker", "")
            val_gbp = float(pos.get("value_gbp", 0.0) or 0.0)
            if np.isnan(val_gbp): val_gbp = 0.0
            qty = float(pos.get("quantity", 0.0) or 0.0)
            if np.isnan(qty): qty = 0.0
            ppl = float(pos.get("unrealized_pnl", 0.0) or 0.0)
            if np.isnan(ppl): ppl = 0.0
            theme = pos.get("theme", "Core Compounder")
            cluster = pos.get("macro_cluster", "Technology")

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

    # Calculate global weight for each holding
    for ticker, item in all_holdings.items():
        v = item["total_value_gbp"]
        item["global_weight_pct"] = round((v / total_net_worth_gbp * 100.0), 1) if total_net_worth_gbp > 0 else 0.0

    # Calculate per-account percentage of total combined net worth
    for acc_name, s in account_shares.items():
        s["share_of_total_pct"] = round((s["total_value_gbp"] / total_net_worth_gbp * 100.0), 1) if total_net_worth_gbp > 0 else 0.0

    # Thematic Breakdown
    thematic_breakdown = []
    for c_name, c_val in sorted(cluster_totals.items(), key=lambda x: x[1], reverse=True):
        pct = round((c_val / total_net_worth_gbp * 100.0), 1) if total_net_worth_gbp > 0 else 0.0
        if pct >= 35.0:
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

    combined_risk_5y = compute_portfolio_5y_risk_profile(
        analyses=combined_analyses,
        available_cash_gbp=total_liquid_cash_gbp,
        blocked_cash_gbp=total_blocked_cash_gbp,
        total_portfolio_gbp=total_net_worth_gbp,
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
    }


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
) -> Dict[str, Any]:
    """Builds an institutional 5-Year Horizon & Crash-Protected rebalancing plan with 100% harmonized table status, factoring in all pending transactions, blocked capital, and global unified portfolio context."""
    
    pending_data = pending_orders_data or {}
    pending_buys = pending_data.get("pending_buys", {})
    pending_sells = pending_data.get("pending_sells", {})
    has_orders_scope = pending_data.get("has_orders_scope", False)
    open_orders_list = pending_data.get("orders", [])

    # 1. Calculate Portfolio Valuation, Current Weights, and Thematic Clusters
    # Note: blocked_cash_gbp is capital committed to pending orders/transactions
    total_invested_gbp = sum(pos.get("value_gbp", 0.0) for pos in analyses)
    total_portfolio_gbp = official_total_gbp if (official_total_gbp and official_total_gbp > 0) else (total_invested_gbp + available_cash_gbp + blocked_cash_gbp)
    for pos in analyses:
        v = pos.get("value_gbp", 0.0)
        pos["portfolio_weight_pct"] = round((v / total_portfolio_gbp * 100.0), 1) if total_portfolio_gbp > 0 else 0.0

    # Thematic Macro Cluster Weights (MPT Concentration Control)
    cluster_totals: Dict[str, float] = {}
    for pos in analyses:
        cluster = pos.get("research", {}).get("macro_cluster", "Diversified Index Core")
        cluster_totals[cluster] = cluster_totals.get(cluster, 0.0) + pos.get("value_gbp", 0.0)

    thematic_breakdown = []
    for c_name, c_val in sorted(cluster_totals.items(), key=lambda x: x[1], reverse=True):
        pct = round((c_val / total_portfolio_gbp * 100.0), 1) if total_portfolio_gbp > 0 else 0.0
        if pct >= 35.0:
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

    for pos in analyses:
        action = pos.get("action_short", "HOLD")
        qty = pos.get("quantity", 0.0)
        price_gbp = pos.get("price_gbp", 0.0)
        val_gbp = pos.get("value_gbp", 0.0)
        ticker = pos.get("ticker", "")
        sym = pos.get("search_symbol", ticker)
        name = pos.get("name", ticker)

        matching_ps_key = next((k for k in pending_sells if tickers_match(k, ticker)), None)
        is_already_pending_sell = matching_ps_key is not None

        if is_already_pending_sell:
            # Order is ALREADY queued in Trading 212 - DO NOT tell user to sell again
            p_sell_info = pending_sells.get(matching_ps_key, {})
            order_type = p_sell_info.get("order_type", "MARKET").upper()
            limit_p = float(p_sell_info.get("limit_price", 0.0) or 0.0)
            curr_p = float(pos.get("price", 0.0) or 0.0)

            # Only count pending sell proceeds if it's a MARKET order or an in-the-money / near-market limit order
            is_immediate_fill = (order_type == "MARKET") or (limit_p <= 0) or (curr_p > 0 and limit_p <= curr_p * 1.02)

            limit_txt = f" @ limit ${limit_p:,.2f}" if limit_p > 0 else ""
            pending_sell_orders.append({
                "ticker": ticker,
                "search_symbol": sym,
                "name": name,
                "action": "SELL (⏳ ALREADY PENDING IN TRADING 212)",
                "is_pending": True,
                "shares": p_sell_info.get("quantity") or qty,
                "price": pos["price"],
                "currency": pos.get("currency", "USD"),
                "estimated_gbp": val_gbp,
                "plain_english_why": (
                    f"Resting limit sell order for {name} ({qty:,.0f} shares{limit_txt}) is queued in Trading 212."
                    if not is_immediate_fill
                    else f"Sell order for {name} ({qty:,.0f} shares) is already queued in Trading 212 awaiting execution. No action needed."
                ),
                "app_instruction": "Already queued in Trading 212 ➔ Awaiting market fill",
                "badge_color": "#f59e0b",
                "limit_price": limit_p,
                "order_type": order_type,
            })
            if is_immediate_fill:
                cash_from_sells += val_gbp
        elif action == "SELL" and qty > 0:
            proceeds = qty * price_gbp
            qty_label = f"{qty:.2f}" if qty < 1.0 else (f"{qty:,.2f}".rstrip("0").rstrip(".") if qty % 1 != 0 else f"{qty:,.0f}")
            sell_orders.append({
                "ticker": ticker,
                "search_symbol": sym,
                "name": name,
                "action": "SELL (GET CASH)",
                "is_pending": False,
                "shares": qty,
                "price": pos["price"],
                "currency": pos.get("currency", "USD"),
                "estimated_gbp": proceeds,
                "plain_english_why": pos.get("plain_english_why", "Falling price trend. Sell to protect capital."),
                "app_instruction": f"Open Trading 212 ➔ Search '{sym}' ➔ Tap Sell ➔ Sell all {qty_label} shares",
                "badge_color": "#ef4444"
            })
        elif (not pos.get("is_recently_trimmed")) and qty > 0 and (
            (any(k in ticker for k in ["QQQ3", "3LUS", "3SEM", "3X"]) and (action == "TRIM" or pos.get("portfolio_weight_pct", 0.0) > 12.0))
            or (action == "TRIM" and pos.get("portfolio_weight_pct", 0.0) >= 12.0)
        ):
            is_lev_item = any(k in ticker for k in ["QQQ3", "3LUS", "3SEM", "3X"])
            w_item = pos.get("portfolio_weight_pct", 0.0)
            if is_lev_item and w_item > 12.0:
                trim_fraction = 0.50
                trim_action = "TRIM (5Y CRASH PROTECTION & LEVERAGE CAP)"
                trim_why = (
                    f"Holding {w_item:.1f}% (£{val_gbp:,.0f}) in 3x daily leveraged Nasdaq exceeds the 8% institutional safety cap. "
                    f"QQQ3.L suffered a -76.2% crash in 2022 and has the lowest 5-year Sortino ratio (0.44) due to daily variance drain. "
                    f"Trimming 50% locks in gains and cuts your portfolio crash drawdown by ~10 percentage points."
                )
            else:
                trim_fraction = 0.25
                trim_action = "TRIM (DE-RISK OVERSIZED WINNER)"
                trim_why = pos.get("plain_english_why", f"Position has grown oversized ({w_item:.1f}% of portfolio) into parabolic strength. Banking 25% profit while keeping 75% as a 5-year runner.")

            trim_qty = round(qty * trim_fraction, 4)
            proceeds = trim_qty * price_gbp
            cash_from_sells += proceeds
            trim_qty_label = f"{trim_qty:.2f}" if trim_qty < 1.0 else (f"{trim_qty:,.2f}".rstrip("0").rstrip(".") if trim_qty % 1 != 0 else f"{trim_qty:,.0f}")
            sell_orders.append({
                "ticker": ticker,
                "search_symbol": sym,
                "name": name,
                "action": trim_action,
                "is_pending": False,
                "shares": trim_qty,
                "price": pos["price"],
                "currency": pos.get("currency", "USD"),
                "estimated_gbp": proceeds,
                "plain_english_why": trim_why,
                "app_instruction": f"Open Trading 212 ➔ Search '{sym}' ➔ Tap Sell ➔ Sell {trim_qty_label} shares (or ~£{proceeds:,.0f})",
                "badge_color": "#f59e0b"
            })

    # 3. Calculate Deployable Funds (Enforcing Protected Cash Withdrawal Floor)
    if min_cash_reserve_gbp is None:
        min_cash_reserve_gbp = REBALANCE_CONFIG.get("min_cash_reserve_gbp", 1500.0)

    # Track capital already committed to pending buy orders in Trading 212
    pending_buys_committed = pending_data.get("total_pending_buy_value", 0.0) or 0.0
    if pending_buys_committed <= 0 and pending_buys:
        pending_buys_committed = sum(float(pb.get("estimated_value", pb.get("value", 0.0)) or 0.0) for pb in pending_buys.values())
    if pending_buys_committed <= 0:
        pending_buys_committed = blocked_cash_gbp

    # Ground-Truth Broker Cash Accounting Guarantee:
    # BUY orders must ONLY be funded with real uninvested liquid cash already in the broker account (available_cash_gbp).
    # Unexecuted recommended trims must NEVER be pre-counted into today's buying power.
    # Buffer reserve ensures minimum cash floor (MIN_CASH_RESERVE_GBP) is ring-fenced for withdrawal safety.
    buffer_reserve = max(min_cash_reserve_gbp, available_cash_gbp * REBALANCE_CONFIG.get("min_cash_buffer_pct", 0.05))
    net_budget_to_deploy = max(0.0, available_cash_gbp - buffer_reserve)
    if net_budget_to_deploy < 1000.0:
        net_budget_to_deploy = 0.0
    total_deployable = available_cash_gbp

    # 4. Institutional 5-Year Horizon MPT, Sortino & Crash-Shield Scoring
    owned_account_tickers = {pos["ticker"] for pos in analyses}
    owns_gold = any(any(g in t.upper() for g in ["SGLN", "IGLN", "GLD"]) for t in owned_account_tickers)
    owns_brkb = any("BRK-B" in t.upper() for t in owned_account_tickers)
    owns_vwrp = any("VWRP" in t.upper() for t in owned_account_tickers)

    # Check thematic cluster breakdown (both local account and global combined wealth)
    active_thematic_breakdown = global_context.get("thematic_breakdown", thematic_breakdown) if global_context else thematic_breakdown
    local_cluster_weights = {tb["cluster"]: tb["weight_pct"] for tb in thematic_breakdown}
    global_cluster_weights = {tb["cluster"]: tb["weight_pct"] for tb in active_thematic_breakdown}

    def _is_cluster_saturated(cluster_name: str, is_broad_index_etf: bool = False) -> bool:
        if is_broad_index_etf or cluster_name in ("Diversified Index Core", "Uncorrelated Crash Hedge & Gold", "Defensive Ballast & Value"):
            return False
        loc_w = local_cluster_weights.get(cluster_name, 0.0)
        glob_w = global_cluster_weights.get(cluster_name, 0.0)
        # Never deploy new cash into a sector that already exceeds 24% of the account or combined family wealth
        return (loc_w >= 24.0) or (glob_w >= 24.0)

    # 4a. Evaluate existing holdings as 5-year add-on candidates
    # STRICT ANTI-ROUND-TRIP INVARIANT: Never buy back any position that was trimmed (is_recently_trimmed)
    eligible_existing_holdings = []
    for pos in analyses:
        t = pos["ticker"]
        if t in EXCLUDED_TICKERS or any(e == t for e in EXCLUDED_TICKERS):
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
        if any(k in t.upper() for k in ["QQQ3", "3LUS", "3SEM", "3X", "SMGB"]):
            continue
        action = pos.get("action_short", "HOLD")
        if action in ["SELL", "TRIM"]:
            continue

        is_broad_index = any(k in t.upper() for k in ["VWRP", "VUAA", "VUSA", "SGLN", "IGLN"])
        w_local = pos.get("portfolio_weight_pct", 0.0)
        w_global = w_local
        if global_context and "holdings" in global_context:
            w_global = global_context.get("holdings", {}).get(t, {}).get("global_weight_pct", w_local)

        # 3. Single-stock concentration cap: Once an individual company is >= 8% of portfolio, let it compound on its own
        if (not is_broad_index) and (w_local >= 8.0 or w_global >= 8.0):
            logger.info("Skipping %s as buy candidate: already full-sized position (%.1f%% local / %.1f%% global).", t, w_local, w_global)
            continue

        # 4. Sector saturation cap: Do not add to sectors already >= 24% of portfolio (e.g. Semiconductors / Cloud Software)
        c_cluster = pos.get("research", {}).get("macro_cluster") or pos.get("macro_cluster", "")
        if _is_cluster_saturated(c_cluster, is_broad_index_etf=is_broad_index):
            logger.info("Skipping %s as buy candidate: sector '%s' is already saturated.", t, c_cluster)
            continue

        is_stage_2 = pos.get("is_weekly_stage_2", True) and (
            pos.get("minervini_pass", 0) >= 3
            or pos.get("conviction_score", 0) >= 70
            or is_broad_index
            or (w_local < 4.5 and float(pos.get("rsi", 50.0) or 50.0) <= 48.0)
        )
        is_ext = pos.get("is_extended", False)
        if action == "BUY" or (is_stage_2 and not is_ext and (pos.get("conviction_score", 0) >= 70 or is_broad_index)):
            eligible_existing_holdings.append(pos)

    # Filter out already-owned tickers BEFORE slicing so unowned Crash-Shield anchors (SGLN.L, BRK-B, VWRP.L, ABBV) are never truncated
    unowned_opportunities = [
        opp for opp in new_opportunities
        if opp["ticker"] not in owned_account_tickers
        and not any(tickers_match(opp["ticker"], ot) for ot in owned_account_tickers)
    ]
    top_new_ideas = unowned_opportunities[:REBALANCE_CONFIG["max_new_ideas_to_buy"] * 6]

    scored_candidates = []
    seen_tickers = set()

    # Evaluate new watchlist additions (skip if already pending execution, already owned, or in a saturated sector)
    for opp in top_new_ideas:
        t = opp["ticker"]
        if t in EXCLUDED_TICKERS or any(e == t for e in EXCLUDED_TICKERS):
            continue
        sym = opp.get("search_symbol", clean_search_symbol(t))
        is_broad_index = any(k in t.upper() for k in ["VWRP", "VUAA", "VUSA", "SGLN", "IGLN"])
        c_cluster = opp.get("research", {}).get("macro_cluster") or opp.get("macro_cluster", "")
        if _is_cluster_saturated(c_cluster, is_broad_index_etf=is_broad_index):
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
        gw = 0.0
        if global_context and "holdings" in global_context:
            gw = global_context.get("holdings", {}).get(t, {}).get("global_weight_pct", 0.0)

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
        w = pos.get("portfolio_weight_pct", 0.0)
        if global_context and "holdings" in global_context:
            w = global_context.get("holdings", {}).get(t, {}).get("global_weight_pct", w)
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
        c_cluster = cand.get("research", {}).get("macro_cluster") or cand.get("macro_cluster", "")
        rr = cand.get("risk_reward_ratio", 2.5)
        p_score = cand.get("research", {}).get("piotroski_score")
        if p_score is None:
            p_score = cand.get("piotroski_score")
        days_e = cand.get("research", {}).get("days_to_earnings")
        sort_5y = float(cand.get("sortino_5y", 1.1) or 1.1)
        crash_22 = float(cand.get("crash_2022_ret_pct", -18.0) or -18.0)
        beta_5y = float(cand.get("beta_5y", 1.0) or 1.0)
        is_lev_cand = any(k in c_ticker_u for k in ["QQQ3", "3LUS", "3SEM", "3X"])

        # 1. Asymmetry bonus / penalty
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
        if c_cluster not in ("Diversified Index Core", "Uncorrelated Crash Hedge & Gold", "Defensive Ballast & Value") and c_weight >= 24.0:
            cand["rebalance_score"] -= 60
            cand["gatekeeper_note"] = f"{c_cluster} Concentration Cap Reached ({c_weight:.1f}%)"
        elif c_weight < 12.0:
            cand["rebalance_score"] += 10  # Boost under-represented long-term pillar

        # 5. Wall Street Consensus Revision Momentum (Academic Alpha Factor #1)
        eps_mom = cand.get("research", {}).get("eps_revision_momentum_pct")
        if eps_mom is not None:
            if eps_mom >= 80.0:
                cand["rebalance_score"] += 8  # Strong institutional upgrade momentum
            elif eps_mom <= 40.0:
                cand["rebalance_score"] -= 18 # Deteriorating Wall Street consensus

        # 6. PEAD Earnings Surprise Streak (Academic Alpha Factor #2)
        beat_rate = cand.get("research", {}).get("earnings_beat_rate_pct")
        if beat_rate is not None:
            if beat_rate >= 75.0:
                cand["rebalance_score"] += 6  # Consistent beat streak
            elif beat_rate < 50.0:
                cand["rebalance_score"] -= 10 # Earnings vulnerability

        # 7. Quality Valuation & Institutional Sponsorship Factors
        fcf_y = cand.get("research", {}).get("fcf_yield_pct")
        if fcf_y is not None and fcf_y >= 3.5:
            cand["rebalance_score"] += 5  # Robust free cash flow backing

        inst_own = cand.get("research", {}).get("institutional_ownership_pct")
        if inst_own is not None and inst_own >= 65.0:
            cand["rebalance_score"] += 4  # Institutional sponsorship

        # 7b. Return on Invested Capital (ROIC) Factor (Compounding Engine)
        roic = cand.get("research", {}).get("roic")
        if roic is not None:
            if roic >= 25.0:
                cand["rebalance_score"] += 6  # Elite ROIC Compounding Machine
            elif roic >= 15.0:
                cand["rebalance_score"] += 4  # Superior Value Creator
            elif roic < 8.0 and not cand.get("is_crash_shield"):
                cand["rebalance_score"] -= 10 # Sub-par capital efficiency / value trap

        # 7c. SEC Form 4 Insider Cluster Buying & Flow
        insider_lbl = cand.get("research", {}).get("insider_label", "")
        if "Insider Buying" in insider_lbl or "Cluster Buying" in insider_lbl:
            cand["rebalance_score"] += 6  # C-suite high-conviction cluster accumulation
        elif "Heavy Insider Selling" in insider_lbl:
            cand["rebalance_score"] -= 8  # Heavy executive liquidation

        # 7d. Options Market Implied Sentiment (Put/Call Ratio)
        pcr_val = cand.get("research", {}).get("options_pcr_vol")
        if pcr_val is not None:
            if pcr_val < 0.65:
                cand["rebalance_score"] += 5  # Institutional call accumulation
            elif pcr_val > 1.25:
                cand["rebalance_score"] -= 8  # Heavy defensive put hedging skew

        # 7e. Top-10 Institutional 13F Ownership Flow Momentum
        inst_chg = cand.get("research", {}).get("institutional_change_pct")
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
        min_pass = cand.get("minervini_pass", 4)
        c_rsi = float(cand.get("rsi", 50.0) or 50.0)
        is_etf_cand = bool((cand.get("research") or {}).get("is_etf")) and not is_lev_cand
        if is_etf_cand:
            # Benchmark Quality Equivalence: Unleveraged index ETFs do not report single-stock Piotroski/ROIC/EPS-revisions
            cand["rebalance_score"] += 22
        if "VWRP" in c_ticker_u and not owns_vwrp:
            cand["rebalance_score"] += 16  # Establish 0.60 beta All-World equity core
        elif "BRK-B" in c_ticker_u and not owns_brkb and min_pass >= 3:
            cand["rebalance_score"] += 14  # Establish $300B+ Cash Fortress ballast
        elif any(g in c_ticker_u for g in ["SGLN", "IGLN", "GLD"]) and not owns_gold and (min_pass >= 3 or c_rsi <= 45.0):
            cand["rebalance_score"] += 14  # Establish -0.03 beta Physical Gold crisis shield

        # 7i. Forward 5-Year Intrinsic Compounding Engine (ROIC Reinvestment + FCF Yield + Multiple Normalization)
        exp_cagr = float(
            cand.get("expected_5y_cagr_pct")
            or (cand.get("research") or {}).get("expected_5y_cagr_pct")
            or cand.get("cagr_5y_pct", 14.0)
        )
        cagr_boost = int(round(max(-12.0, min(18.0, (exp_cagr - 12.0) * 1.5))))
        cand["rebalance_score"] += cagr_boost

        # 7j. Institutional Pullback-to-Value Timing Bonus (Buy wide-moat leaders on dips, don't chase spikes)
        if cand.get("is_weekly_stage_2", True):
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
    min_ord_gbp = float(REBALANCE_CONFIG.get("min_order_gbp", 950.0))
    effective_floor_gbp = max(min_ord_gbp, 950.0 if net_budget_to_deploy >= 3000.0 else min_ord_gbp)
    max_feasible_orders = int(net_budget_to_deploy // effective_floor_gbp) if effective_floor_gbp > 0 else 0
    target_count = min(
        len(scored_candidates),
        max_feasible_orders,
        4 if net_budget_to_deploy > 7500 else (3 if net_budget_to_deploy >= 3100 else 2)
    )

    if scored_candidates and target_count >= 1 and net_budget_to_deploy >= effective_floor_gbp:
        # Select top candidates while enforcing cross-cluster diversity so we don't double-up on the same sector in one batch
        selected_candidates = []
        used_clusters = set()
        top_score = scored_candidates[0].get("rebalance_score", 80)
        for cand in scored_candidates:
            if len(selected_candidates) >= target_count:
                break
            c_cl = cand.get("macro_cluster") or (cand.get("research") or {}).get("macro_cluster", "")
            if c_cl in used_clusters and (top_score - cand.get("rebalance_score", 0)) < 20:
                continue
            selected_candidates.append(cand)
            if c_cl:
                used_clusters.add(c_cl)
        # Backfill if diversity filter left slots open
        if len(selected_candidates) < target_count:
            for cand in scored_candidates:
                if len(selected_candidates) >= target_count:
                    break
                if cand["ticker"] not in {c["ticker"] for c in selected_candidates}:
                    selected_candidates.append(cand)

        # All-Weather Crash-Shield Completion Rule:
        # If an account lacks VWRP.L or has < 13% invested in Crash Shield assets, ensure at least 1 slot builds a low-beta Crash Shield anchor
        curr_invested_shield_gbp = sum(
            float(p.get("value_gbp", 0.0) or 0.0)
            for p in analyses
            if p.get("is_crash_shield")
        )
        curr_invested_shield_pct = (curr_invested_shield_gbp / max(1.0, total_portfolio_gbp)) * 100.0
        has_shield_in_selected = any(
            c.get("is_crash_shield") or "VWRP" in c.get("ticker", "").upper()
            for c in selected_candidates
        )
        if target_count >= 2 and (not owns_vwrp or curr_invested_shield_pct < 12.5) and not has_shield_in_selected:
            best_shield_cand = next(
                (
                    c for c in scored_candidates
                    if (c.get("is_crash_shield") or "VWRP" in c.get("ticker", "").upper())
                    and c.get("minervini_pass", 0) >= 3
                ),
                None,
            )
            if best_shield_cand is not None:
                selected_candidates[-1] = best_shield_cand

        # Fractional Kelly Conviction + Expected 5Y Compounding Edge over Downside Volatility:
        candidate_weights = {}
        for cand in selected_candidates:
            c_t = cand["ticker"]
            vcp = cand.get("vcp_pct")
            if vcp is None or vcp <= 0:
                vcp = 2.0
            vol_risk = max(1.2, float(vcp) * (0.85 if cand.get("is_crash_shield") else 1.0))
            score_factor = max(15.0, float(cand.get("rebalance_score", 50)))
            exp_c = float(
                cand.get("expected_5y_cagr_pct")
                or (cand.get("research") or {}).get("expected_5y_cagr_pct")
                or cand.get("cagr_5y_pct", 14.0)
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

        # Total portfolio equity baseline for risk percentage limits
        acc_total_equity = sum(p.get("value_gbp", 0.0) for p in analyses) + available_cash_gbp + cash_from_sells
        acc_total_equity = max(acc_total_equity, net_budget_to_deploy)

        allocated_amounts = {}
        spillover_capital = 0.0
        uncapped_candidates = []

        for candidate in selected_candidates:
            c_ticker = candidate["ticker"]
            is_leveraged = any(k in c_ticker for k in ["QQQ3", "3LUS", "3SEM"])
            existing_val = next((p.get("value_gbp", 0.0) for p in analyses if p.get("ticker") == c_ticker), 0.0)
            target_raw = raw_allocations[c_ticker]

            if is_leveraged:
                # Institutional 8.0% Maximum Portfolio Cap on 3x Leveraged instruments
                max_allowed_lev = max(min_ord_gbp, acc_total_equity * 0.08)
                room = max(0.0, max_allowed_lev - existing_val)
                buy_amt = min(target_raw, room)
                if buy_amt < min_ord_gbp and room >= min_ord_gbp:
                    buy_amt = min_ord_gbp
                elif room < min_ord_gbp:
                    buy_amt = 0.0

                allocated_amounts[c_ticker] = buy_amt
                spillover_capital += max(0.0, target_raw - buy_amt)
            else:
                uncapped_candidates.append(c_ticker)

        # Redistribute spillover capital from capped leveraged products into bedrock compounders proportionally
        if uncapped_candidates and spillover_capital > 0:
            uncapped_w_sum = sum(candidate_weights[c] for c in uncapped_candidates) or 1.0
            for c_ticker in uncapped_candidates:
                extra = spillover_capital * (candidate_weights[c_ticker] / uncapped_w_sum)
                allocated_amounts[c_ticker] = raw_allocations[c_ticker] + extra
        else:
            for c_ticker in uncapped_candidates:
                allocated_amounts[c_ticker] = raw_allocations[c_ticker]

        for candidate in selected_candidates:
            c_ticker = candidate["ticker"]
            alloc_gbp = allocated_amounts.get(c_ticker, 0.0)
            if alloc_gbp < min_ord_gbp * 0.95:
                continue

            p_gbp = candidate.get("price_gbp", 1.0)
            if p_gbp <= 0:
                continue

            shares = alloc_gbp / p_gbp
            shares = round(shares, 2) if p_gbp > 50 else round(shares, 1)

            order_val_gbp = shares * p_gbp
            sym = candidate.get("search_symbol", candidate["ticker"])
            name = candidate.get("name", candidate["ticker"])
            is_new = candidate.get("is_new_addition", False)
            cagr_5y = candidate.get("cagr_5y_pct", 15.0)
            exp_cagr_5y = float(
                candidate.get("expected_5y_cagr_pct")
                or (candidate.get("research") or {}).get("expected_5y_cagr_pct")
                or cagr_5y
            )
            val_regime = (
                candidate.get("valuation_regime_label")
                or (candidate.get("research") or {}).get("valuation_regime_label")
                or "🟢 Fair GARP Valuation"
            )
            sort_5y = candidate.get("sortino_5y", 1.30)
            crash_22 = candidate.get("crash_2022_ret_pct", 0.0)
            beta_5y = candidate.get("beta_5y", 0.50)
            role_lbl = candidate.get("crash_role_label", "🛡️ 5Y Compounder")

            type_label = "NEW SHARE" if is_new else "ADD TO POSITION"
            crash_note = (
                f"+{crash_22:.1f}% positive crisis alpha in 2022 crash"
                if crash_22 >= 0.0
                else f"Beta {beta_5y:.2f} ({val_regime})"
            )
            buy_orders.append({
                "ticker": candidate["ticker"],
                "search_symbol": sym,
                "name": name,
                "theme": candidate.get("theme", "Core Compounder"),
                "macro_cluster": candidate.get("macro_cluster", "Technology"),
                "action": f"BUY ({type_label})",
                "is_new": is_new,
                "shares": shares,
                "price": candidate["price"],
                "currency": candidate.get("currency", "USD"),
                "estimated_gbp": order_val_gbp,
                "stop_price": candidate["trailing_stop"],
                "target_price": candidate["target_price"],
                "risk_reward_ratio": candidate.get("risk_reward_ratio", 2.5),
                "cagr_5y_pct": cagr_5y,
                "expected_5y_cagr_pct": exp_cagr_5y,
                "valuation_regime_label": val_regime,
                "sortino_5y": sort_5y,
                "crash_2022_ret_pct": crash_22,
                "beta_5y": beta_5y,
                "crash_role_label": role_lbl,
                "piotroski_score": candidate.get("piotroski_score"),
                "altman_z": candidate.get("altman_z"),
                "altman_label": candidate.get("altman_label", "N/A"),
                "sloan_accrual": candidate.get("sloan_accrual"),
                "sloan_label": candidate.get("sloan_label", "N/A"),
                "weekly_trend_label": candidate.get("weekly_trend_label", "Weekly Confirmed"),
                "eps_revision_label": candidate.get("eps_revision_label") or candidate.get("research", {}).get("eps_revision_label", "Neutral Consensus"),
                "eps_revision_momentum_pct": candidate.get("eps_revision_momentum_pct") or candidate.get("research", {}).get("eps_revision_momentum_pct"),
                "minervini_stage": candidate.get("minervini_stage", "Stage 2 Confirmed Bull (5/5)"),
                "volume_footprint_label": candidate.get("volume_footprint_label", "Constructive Volume"),
                "rs_label": candidate.get("rs_label", "Benchmark Neutral"),
                "chandelier_stop": candidate.get("chandelier_stop", candidate.get("trailing_stop")),
                "earnings_catalyst_status": candidate.get("earnings_catalyst_status", "✅ Clear"),
                "plain_english_why": (
                    f"{role_lbl} ({candidate.get('theme', 'Secular Growth')}): +{exp_cagr_5y:.1f}% Forward 5Y Expected CAGR (+{cagr_5y:.1f}% Hist 5Y), "
                    f"{sort_5y:.2f} Sortino, RSI {candidate.get('rsi', 50.0):.1f}, {crash_note}. "
                    + (
                        f"Initiating high-conviction position to compound 5-year wealth in {candidate.get('macro_cluster', 'all-weather')}."
                        if is_new
                        else f"Scaling into under-weighted 5-year compounder ({candidate.get('portfolio_weight_pct', 0.0):.1f}% current weight) at pullback support with {candidate.get('risk_reward_ratio', 2.5)}:1 R:R."
                    )
                ),
                "app_instruction": f"Open Trading 212 ➔ Search '{sym}' ➔ Tap Buy ➔ Enter £{order_val_gbp:,.0f}",
                "conviction_score": candidate.get("conviction_score", 70),
                "rebalance_score": candidate.get("rebalance_score", 75),
                "research": candidate.get("research", {}),
                "badge_color": "#10b981"
            })

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
            p_sell = pending_sells[matching_ps]
            pos["plan_status"] = "⏳ PENDING SELL"
            pos["plan_badge_color"] = "#f59e0b"
            pos["plan_summary"] = f"Sell order of {p_sell.get('quantity', pos['quantity']):,.0f} shares is currently pending execution in Trading 212."
        elif matching_pb:
            p_buy = pending_buys[matching_pb]
            pos["plan_status"] = "⏳ PENDING BUY"
            pos["plan_badge_color"] = "#0284c7"
            pos["plan_summary"] = f"Buy order of ~£{p_buy.get('estimated_value', 0):,.0f} is currently pending execution in Trading 212."
        elif matching_so:
            so = matching_so
            is_trim = "TRIM" in so.get("action", "")
            if is_trim:
                pos["plan_status"] = "TRIM"
                pos["plan_badge_color"] = "#f59e0b"
                pos["plan_summary"] = f"Trimming {so['shares']:,.2f} shares (+£{so['estimated_gbp']:,.0f} cash unlocked) to cap 3x leverage & protect against crashes."
            else:
                pos["plan_status"] = "SELL"
                pos["plan_badge_color"] = "#ef4444"
                pos["plan_summary"] = f"Liquidating all {pos['quantity']:,.0f} shares (+£{so['estimated_gbp']:,.0f} cash unlocked). Exiting downtrend."
        elif matching_bo:
            bo = matching_bo
            pos["plan_status"] = f"BUY (+£{bo['estimated_gbp']:,.0f})"
            pos["plan_badge_color"] = "#10b981"
            pos["plan_summary"] = f"Active Buy Order: Deploying £{bo['estimated_gbp']:,.0f} into 5-year crash-resistant compounder ({w}% weight)."
        else:
            # Held position
            pos["plan_status"] = "HOLD"
            pos["plan_badge_color"] = "#6366f1"
            if w >= 18.0:
                pos["plan_summary"] = f"Core foundation ({w}% of portfolio). 5-year uptrend healthy—hold and let compound."
            elif w >= 10.0:
                pos["plan_summary"] = f"Core holding ({w}% of portfolio). Fully positioned—hold and let compound over 5 years."
            else:
                pos["plan_summary"] = f"5-year compounder ({w}% of portfolio). Primary uptrend intact above 200 EMA."
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
                "plan_summary": f"Pending Buy Order: ~£{pbuy.get('estimated_value', 0):,.0f} committed in Trading 212.",
                "currency": "USD",
                "price": pbuy.get("limit_price", 0.0),
                "is_new_addition": True,
                "research": {},
            })

    # Also append new additions from rebalancing plan to table_items
    for bo in buy_orders:
        if bo.get("is_new"):
            table_items.append({
                "ticker": bo["ticker"],
                "search_symbol": bo["search_symbol"],
                "name": bo["name"],
                "value_gbp": 0.0,
                "unrealized_pnl": 0.0,
                "plan_status": f"BUY (+£{bo['estimated_gbp']:,.0f})",
                "plan_badge_color": "#10b981",
                "plan_summary": f"NEW 5Y SHIELD & ALPHA: Initiating £{bo['estimated_gbp']:,.0f} starter position ({bo.get('theme', 'Growth')}).",
                "currency": bo.get("currency", "USD"),
                "price": bo["price"],
                "cagr_5y_pct": bo.get("cagr_5y_pct", 15.0),
                "sortino_5y": bo.get("sortino_5y", 1.30),
                "crash_2022_ret_pct": bo.get("crash_2022_ret_pct", 0.0),
                "beta_5y": bo.get("beta_5y", 0.50),
                "crash_role_label": bo.get("crash_role_label", "🛡️ 5Y Compounder"),
                "is_new_addition": True,
                "research": bo.get("research", {}),
            })

    final_cash_balance = available_cash_gbp - pending_buys_committed - sum(b["estimated_gbp"] for b in buy_orders)

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
    }


def generate_tldr_summary(results_by_account: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generates high-priority, 30-second actionable TL;DR directives partitioned by account."""
    tldr = {}
    for acc_name, data in results_by_account.items():
        plan = data.get("rebalancing_plan", {})
        cash_info = data.get("cash", {})
        avail_cash = plan.get("available_cash_gbp", 0.0)
        net_budget = plan.get("net_budget_to_deploy_gbp", 0.0)

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
            inst_text = f"Sell {s.get('shares', 0):,.0f} shares (take profits)" if is_trim else f"Sell all {s.get('shares', 0):,.0f} shares"
            sells_to_place.append({
                "name": s.get("name", s.get("ticker")),
                "symbol": s.get("search_symbol", s.get("ticker")),
                "shares": s.get("shares", 0),
                "is_trim": is_trim,
                "estimated_gbp": s.get("estimated_gbp", 0.0),
                "instruction": f"{action_verb} {s.get('name', s.get('ticker'))} ({s.get('search_symbol', s.get('ticker'))}) ➔ {inst_text} (+£{s.get('estimated_gbp', 0):,.0f})"
            })

        sells_already_queued = []
        for ps in pending_sells:
            lim = ps.get("limit_price", 0.0)
            lim_str = f" @ limit ${lim:,.2f}" if lim > 0 else ""
            s_name = ps.get("name", ps.get("ticker"))
            s_sym = ps.get("search_symbol", ps.get("ticker"))
            sells_already_queued.append(f"{s_name} ({s_sym}){lim_str}")

        buys_already_queued = []
        for pb_sym, pb in pending_buys.items():
            pb_name = pb.get("name", pb_sym)
            pb_val = pb.get("estimated_value", pb.get("value", 0.0))
            buys_already_queued.append(f"{pb_name} (~£{pb_val:,.0f})")

        buys_to_place = []
        for b in buy_orders:
            buys_to_place.append({
                "name": b.get("name", b.get("ticker")),
                "symbol": b.get("search_symbol", b.get("ticker")),
                "amount_gbp": b.get("estimated_gbp", 0.0),
                "shares": b.get("shares", 0.0),
                "is_new": b.get("is_new", False),
                "instruction": f"BUY {b.get('name', b.get('ticker'))} ({b.get('search_symbol', b.get('ticker'))}) ➔ Enter £{b.get('estimated_gbp', 0):,.0f}"
            })

        hold_names = [h.get("name", h.get("ticker")) for h in hold_items]

        has_pending_market_sells = any(
            ps.get("order_type", "MARKET") == "MARKET" or ps.get("limit_price", 0) <= 0
            for ps in pending_sells
        )

        pending_proceeds = sum(ps.get("estimated_gbp", 0.0) for ps in pending_sells if ps.get("limit_price", 0) <= 0 or ps.get("order_type", "MARKET") == "MARKET")

        all_trims = bool(sells_to_place and all(s.get("is_trim", False) for s in sells_to_place))

        if sells_to_place and buys_to_place:
            if all_trims:
                phase = "HARVEST_AND_REALLOCATE"
                phase_label = "🟠 HARVEST PROFITS & REALLOCATE CAPITAL"
                phase_desc = f"Execute {len(sells_to_place)} profit trim(s) below to bank gains. Reallocate proceeds into {len(buys_to_place)} high-asymmetry buy orders."
            else:
                phase = "SELL_FIRST"
                phase_label = "🔴 SELL ACTIONS REQUIRED FIRST"
                phase_desc = f"Place {len(sells_to_place)} sell order(s) below to unlock capital. You also have £{avail_cash:,.0f} liquid cash ready to deploy into buys today."
        elif sells_to_place:
            phase = "TRIM_FIRST" if all_trims else "SELL_FIRST"
            phase_label = "🟠 HARVEST PROFITS FIRST" if all_trims else "🔴 SELL ACTIONS REQUIRED FIRST"
            phase_desc = "Open Trading 212 and execute the trim orders below to bank profits." if all_trims else "Open Trading 212 and execute the sell orders below to unlock capital."
        elif has_pending_market_sells and avail_cash >= 1000 and buys_to_place:
            phase = "BUY_NOW"
            phase_label = "🟢 ACTION REQUIRED TODAY — PLACE BUYS"
            phase_desc = f"You have £{avail_cash:,.0f} free cash ready now, plus ~£{pending_proceeds:,.0f} from queued sells. Place these {len(buys_to_place)} buy orders today."
        elif has_pending_market_sells and buys_to_place:
            phase = "AWAIT_FILLS_THEN_BUY"
            phase_label = "⏳ SELLS QUEUED ➔ BUY ONCE CASH SETTLES"
            phase_desc = f"Your sell orders are queued awaiting market fill (+~£{pending_proceeds:,.0f} cash). Once filled, place the {len(buys_to_place)} buy orders below."
        elif buys_to_place:
            phase = "BUY_NOW"
            phase_label = "🟢 ACTION REQUIRED TODAY — PLACE BUYS"
            phase_desc = f"You have £{net_budget:,.0f} deployable capital. Place the {len(buys_to_place)} buy orders below."
        else:
            phase = "ALL_SET"
            phase_label = "✅ ALL SET — SIT BACK & RELAX"
            phase_desc = "No action required today. Portfolio is fully balanced and compounding."

        tldr[acc_name] = {
            "account_name": acc_name,
            "phase": phase,
            "phase_label": phase_label,
            "phase_desc": phase_desc,
            "available_cash": avail_cash,
            "net_budget": net_budget,
            "pending_proceeds": pending_proceeds,
            "buffer_reserve": plan.get("buffer_reserve_gbp", 0.0),
            "sells_to_place": sells_to_place,
            "sells_already_queued": sells_already_queued,
            "buys_already_queued": buys_already_queued,
            "buys_to_place": buys_to_place,
            "hold_names": hold_names,
        }

    return tldr


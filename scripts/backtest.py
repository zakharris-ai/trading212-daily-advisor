#!/usr/bin/env python3
"""Phase 2 evidence: do simple rules-based strategies beat just holding a global index fund (VWRP)?

A monthly, GBP-based, after-cost backtest for a UK ISA investor (no tax), using free Yahoo Finance data.

Strategies (fixed textbook parameters; nothing is fitted or optimised, so the whole window is out-of-sample
with respect to the parameters):
  S0      100% global equity (ACWI converted to GBP: the proxy for VWRP).
  S1      85% global equity + 7.5% gold + 7.5% UK short gilts. Sleeves are only reset to target when one drifts
          more than +/-5 percentage points from its target (then all sleeves are reset).
  S2      65% global equity + 20% satellite + 7.5% gold + 7.5% gilts. Satellite = top 3 of 9 US sector ETFs by
          12-1 momentum (return from month-end t-12 to t-1, skipping the latest month; Jegadeesh & Titman 1993),
          equal-weighted, re-selected every month; sleeve-level +/-5pp bands as in S1.
  S2b     Control for S2: identical, but the satellite holds all 9 sectors equally (no momentum selection).
  S3      Faber (2007) trend filter: global equity while its month-end GBP price > its 10-month SMA, else gilts.
  S4      The tool's own trend rule at index level: global equity while strategy_engine.evaluate_weekly_trend()
          would report is_weekly_stage_2 on weekly (W-FRI) GBP closes, else gilts. Signals at a week's close apply
          from the next week. The tool's rule for VWRP includes its '50-week crisis shield' exception, so S4 does
          too; S4-pure is the 30-week EMA test alone (what the tool applies to any non-exempt ticker).

Method:
  * Daily auto-adjusted closes (total return via adjusted close) put on one calendar (forward-filled), USD assets
    converted to GBP (GBP value = USD price / GBPUSD), then sampled at month-end; the current month is dropped.
  * No look-ahead: signals at month-end t see only data up to and including t (simulate() passes the strategy a
    history truncated at t); weights decided at t earn the t -> t+1 return; weights drift between rebalances.
    S4 does the same with weekly signals and daily returns, then its equity curve is sampled at month-end.
  * Costs on every trade: 0.05% x |weight change| x portfolio value, plus a 0.15% FX fee on sector-ETF trades
    only (the core and defensive sleeves would be held through GBP lines VWRP.L / SGLN.L / IGLS.L). No extra fund
    fees: proxy prices already include their own (ACWI's 0.32% is above VWRP's 0.22%, which is conservative).
    The initial purchase is free for every strategy.
  * Metrics: CAGR, volatility, Sortino (annualised mean monthly return / downside deviation, MAR 0), max drawdown,
    Calmar, one-way turnover, costs, % of calendar years beating S0, rolling 5-year CAGR, worst 12 months and the
    CAGR gap vs S0, for the full window and 2008-2014 / 2015-2020 / 2021-latest.

Caveats:
  * Price-only rules: free data has no point-in-time fundamentals, so the tool's fundamental screens are untested.
  * Sector ETFs, not stocks: deliberate (it avoids the survivorship bias of back-testing today's hand-picked
    watchlist), but it means this does not test single-stock picking.
  * UK short gilts: IGLS.L only starts in April 2009; before that SHY (US 1-3y Treasuries) converted to GBP is
    spliced in. Unhedged SHY carries the 2008-09 sterling collapse, which flatters anything holding 'gilts' then,
    so a sensitivity run uses SHY's USD return instead (roughly what a currency-hedged holding earned).
  * ACWI is the VWRP proxy. Before ACWI launched (March 2008) the trend signals are warmed up on the MSCI ACWI
    price index plus 2%/yr for dividends (fallback: IOO); this is used for signals only, never for returns.
  * One historical path; month-end sampling understates intra-month drawdowns; trades are assumed at the signal
    close; the strategy menu itself was chosen in 2026 with broad hindsight.

Usage (from the repo root; needs network for Yahoo Finance):
    ./.venv/bin/python3 scripts/backtest.py --out results.md [--json results.json]
Nothing is written unless --out and/or --json are given (a short summary is printed either way).
"""

import argparse
import json
import math
import sys
import time
from datetime import date
from functools import reduce
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------
# Universe and fixed parameters
# ----------------------------------------------------------------------------------------------------------------
DOWNLOAD_START = "1998-01-01"
EQUITY = "ACWI"                  # global equity (USD), proxy for VWRP
GOLD = "GLD"                     # gold (USD), cross-check SGLN.L
IGLS = "IGLS.L"                  # UK 0-5y gilts (GBP), from April 2009
SHY = "SHY"                      # US 1-3y Treasuries (USD), spliced in before IGLS.L
FX = "GBPUSD=X"                  # USD per 1 GBP
SECTORS = ("XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB")
CROSS_CHECK_TICKERS = ("VWRL.L", "SGLN.L")
# (ticker, annual drift added for missing dividends, description); first one available is used for signal warm-up
SIGNAL_PROXIES = (
    ("^892400-USD-STRD", 0.02, "MSCI ACWI price index + 2%/yr for dividends"),
    ("IOO", 0.0, "iShares Global 100 ETF"),
)
REQUIRED_TICKERS = (EQUITY, GOLD, IGLS, SHY, FX) + SECTORS
USD_TICKERS = (EQUITY, GOLD, SHY) + SECTORS

# Derived series names
GILTS = "GILTS"                  # IGLS.L, with SHY converted to GBP before it exists (the instructed splice)
GILTS_HEDGED = "GILTS_HEDGED"    # IGLS.L, with SHY's USD return before it exists (sensitivity)
EQUITY_SIGNAL = "ACWI_SIGNAL"    # ACWI in GBP, back-filled with the proxy before launch (signals only)
SHY_USD = "SHY_USD"

SPREAD = 0.0005                  # 0.05% on every trade
FX_FEE = 0.0015                  # +0.15% on sector-ETF trades only
BAND = 0.05                      # +/-5 percentage-point sleeve bands
SMA_MONTHS = 10                  # Faber (2007)
MOM_TOP_N = 3
S1_SLEEVES = {"core": 0.85, "gold": 0.075, "gilts": 0.075}
S2_SLEEVES = {"core": 0.65, "satellite": 0.20, "gold": 0.075, "gilts": 0.075}
# Tickers the tool exempts via its 50-week 'crisis shield' (strategy_engine.evaluate_weekly_trend)
CRISIS_HEDGE_SYMBOLS = ("SGLN", "IGLN", "GLD", "BRK-B", "VWRP", "CSH2", "ABBV")
GLITCH_MAX_JUMP = 0.25           # a .L ETF never genuinely jumps 25% in a day and straight back
MONTHLY_RED_FLAG = 0.50          # a monthly ETF return beyond +/-50% is treated as a data red flag

STRATEGY_ORDER = ("S0", "S1", "S2", "S2b", "S3", "S4", "S4-pure")
STRATEGY_LABELS = {
    "S0": "100% global equity (benchmark)",
    "S1": "85% equity / 7.5% gold / 7.5% gilts, bands",
    "S2": "65% equity / 20% momentum sectors / 15% defensive",
    "S2b": "S2 control: all 9 sectors, no momentum",
    "S3": "10-month SMA trend filter (equity or gilts)",
    "S4": "Tool's weekly Stage 2 rule (equity or gilts)",
    "S4-pure": "S4 without the VWRP 50-week exception",
}
PERIODS = (("Full period", None, None), ("2008-2014", 2008, 2014), ("2015-2020", 2015, 2020), ("2021-latest", 2021, None))


# ----------------------------------------------------------------------------------------------------------------
# Data loading (network; not unit-tested)
# ----------------------------------------------------------------------------------------------------------------
def _extract_close(df: Optional[pd.DataFrame], ticker: str) -> Optional[pd.Series]:
    """Close column of a yf.download() frame as a clean, tz-naive, sorted float Series (None when unusable)."""
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close[ticker] if ticker in close.columns else close.iloc[:, 0]
    else:
        if "Close" not in df.columns:
            return None
        close = df["Close"]
    close = pd.to_numeric(close, errors="coerce").dropna()
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.name = ticker
    return close


def download_daily_closes(tickers: Iterable[str], start: str = DOWNLOAD_START, retries: int = 4,
                          backoff: float = 5.0, log: Callable[[str], None] = print) -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    """Daily auto-adjusted closes from Yahoo Finance, one ticker at a time, with exponential-backoff retries."""
    import yfinance as yf  # imported lazily so the offline tests never touch it

    closes: Dict[str, pd.Series] = {}
    failures: Dict[str, str] = {}
    for ticker in tickers:
        for attempt in range(1, retries + 1):
            try:
                df = yf.download(ticker, start=start, interval="1d", auto_adjust=True, progress=False, threads=False)
                series = _extract_close(df, ticker)
                if series is None or len(series) < 30:
                    raise RuntimeError("no usable data returned")
                closes[ticker] = series
                failures.pop(ticker, None)
                break
            except Exception as exc:  # noqa: BLE001 - network/API errors are retried, then reported
                failures[ticker] = str(exc)
                if attempt < retries:
                    wait = backoff * 2 ** (attempt - 1)
                    log(f"  {ticker}: attempt {attempt} failed ({exc}); retrying in {wait:.0f}s")
                    time.sleep(wait)
    return closes, failures


def all_tickers() -> List[str]:
    return list(REQUIRED_TICKERS) + list(CROSS_CHECK_TICKERS) + [p[0] for p in SIGNAL_PROXIES]


def load_monthly_prices(today: Optional[Any] = None, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Convenience wrapper (network): downloads everything and returns the month-end GBP price panel."""
    closes, _ = download_daily_closes(all_tickers(), log=log)
    return prepare_data(closes, today=today)["monthly"]


# ----------------------------------------------------------------------------------------------------------------
# Pure data helpers
# ----------------------------------------------------------------------------------------------------------------
def repair_price_glitches(prices: pd.Series, max_jump: float = GLITCH_MAX_JUMP,
                          max_run: int = 30) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """Repairs Yahoo unit glitches in a daily ETF close series, walking backwards from the (most reliable) latest data.

    * A price ~100x or ~1/100x the running reference is a pence/pound mix-up and is rescaled.
    * Any other jump beyond +/-max_jump that later reverts is dropped (NaN), e.g. SGLN.L's 2011 days quoted in US cents.
    * More than max_run consecutive rejected prices are accepted as a genuine level change.
    Returns (repaired series with NaN for dropped points, list of events).
    """
    s = pd.to_numeric(pd.Series(prices), errors="coerce")
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    events: List[Dict[str, Any]] = []
    good = vals[np.isfinite(vals) & (vals > 0)]
    if len(good) == 0:
        return pd.Series(out, index=s.index, name=s.name), events
    ref = float(np.median(good[-10:]))
    limit = math.log1p(max_jump)
    pending: List[int] = []
    ratios: Dict[int, float] = {}  # rejected index -> price / trusted reference

    def _confirm_dropped() -> None:
        for j in pending:
            events.append({"date": s.index[j], "raw": float(vals[j]), "fixed": None, "action": "dropped",
                           "ratio": ratios.get(j, float("nan"))})
        pending.clear()

    for i in range(len(vals) - 1, -1, -1):
        p = vals[i]
        if not (np.isfinite(p) and p > 0):
            continue
        ratio = p / ref
        cand, action = p, None
        if 50.0 <= ratio <= 200.0:
            cand, action = p / 100.0, "rescaled /100"
        elif 1.0 / 200.0 <= ratio <= 1.0 / 50.0:
            cand, action = p * 100.0, "rescaled x100"
        if abs(math.log(cand / ref)) <= limit:
            _confirm_dropped()
            out[i] = cand
            ref = cand
            if action:
                events.append({"date": s.index[i], "raw": float(p), "fixed": float(cand), "action": action})
        else:
            pending.append(i)
            ratios[i] = float(ratio)
            if len(pending) > max_run:  # persistent: a genuine level, not a glitch
                for j in pending:
                    out[j] = vals[j]
                ref = float(vals[pending[-1]])
                pending.clear()
    _confirm_dropped()
    events.sort(key=lambda e: e["date"])
    return pd.Series(out, index=s.index, name=s.name), events


def flag_monthly_outliers(monthly_prices: pd.Series, limit: float = MONTHLY_RED_FLAG) -> List[Tuple[pd.Timestamp, float]]:
    """Month-end returns beyond +/-limit (a red flag for unit glitches on an ETF)."""
    r = monthly_prices.dropna().pct_change(fill_method=None).dropna()
    return [(d, float(v)) for d, v in r[r.abs() > limit].items()]


def usd_to_gbp(prices_usd: Any, gbpusd: pd.Series) -> Any:
    """GBP value of USD-priced assets: USD price / GBPUSD (GBPUSD = USD per 1 GBP), FX forward-filled onto the price dates."""
    fx = gbpusd.reindex(gbpusd.index.union(prices_usd.index)).sort_index().ffill().reindex(prices_usd.index)
    if isinstance(prices_usd, pd.DataFrame):
        return prices_usd.div(fx, axis=0)
    return prices_usd / fx


def splice_series(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    """Chain-linked index: fallback's returns until primary's first price, primary's returns afterwards.

    Both series must share one (calendar) index. The level starts at 1.0 on fallback's first valid date.
    """
    idx = primary.index.union(fallback.index)
    p = primary.reindex(idx)
    f = fallback.reindex(idx)
    first_primary = p.first_valid_index()
    first_fallback = f.first_valid_index()
    if first_fallback is None:
        raise ValueError("fallback series is empty")
    rp = p.ffill().pct_change(fill_method=None)
    rf = f.ffill().pct_change(fill_method=None)
    r = rf.copy() if first_primary is None else rf.where(idx <= first_primary, rp)
    r = r[idx >= first_fallback].fillna(0.0)
    r.iloc[0] = 0.0
    return (1.0 + r).cumprod().reindex(idx)


def extend_with_proxy(primary: pd.Series, proxy: pd.Series, annual_drift: float = 0.0) -> pd.Series:
    """Back-fills primary before its first price with proxy's path, joined at primary's first date.

    annual_drift adds missing growth to the proxy (e.g. dividends for a price-only index): going back t years the
    proxy segment is scaled by exp(-annual_drift * t). Values from primary's first date onwards are primary's own.
    """
    p = primary.dropna()
    x = proxy.dropna()
    if p.empty:
        raise ValueError("primary series is empty")
    start = p.index[0]
    x_before = x[x.index <= start]
    if x_before.empty:
        return p.copy()
    anchor = float(x_before.iloc[-1])
    years_back = (start - x_before.index).days.to_numpy(dtype=float) / 365.25
    back = float(p.iloc[0]) * (x_before / anchor) * np.exp(-annual_drift * years_back)
    back = back[back.index < start]
    return pd.concat([back, p]).sort_index()


# ----------------------------------------------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------------------------------------------
def momentum_12_1(prices: pd.DataFrame, t: Optional[Any] = None) -> pd.Series:
    """12-1 momentum at month-end t: total return from month-end t-12 to month-end t-1 (the latest month is skipped).

    Uses only rows up to and including t. All NaN when fewer than 13 month-ends are available.
    """
    hist = prices if t is None else prices.loc[:t]
    if len(hist) < 13:
        return pd.Series(np.nan, index=prices.columns, dtype=float)
    return hist.iloc[-2] / hist.iloc[-13] - 1.0


def top_by_momentum(prices: pd.DataFrame, n: int = MOM_TOP_N, t: Optional[Any] = None) -> List[str]:
    """Names of the n highest 12-1 momentum assets (ties keep column order)."""
    mom = momentum_12_1(prices, t).dropna()
    if len(mom) < n:
        raise ValueError("not enough momentum history to select the satellite")
    return list(mom.sort_values(ascending=False, kind="mergesort").index[:n])


def sma_signal(prices: pd.Series, months: int = SMA_MONTHS) -> bool:
    """Faber trend test on month-end prices: True when the latest price is strictly above its `months`-month SMA."""
    px = prices.dropna()
    if len(px) < months:
        raise ValueError(f"need {months} month-ends for the SMA, have {len(px)}")
    return bool(px.iloc[-1] > px.iloc[-months:].mean())


def _tool_cagr_5y_pct(c: pd.Series) -> float:
    """cagr_5y_pct exactly as evaluate_weekly_trend computes it (5y CAGR blended 55/45 with the 3y CAGR)."""
    latest = float(c.iloc[-1])
    days_span = max(365.0, float((c.index[-1] - c.index[0]).days))
    years_span = days_span / 365.25
    first = float(c.iloc[0]) if float(c.iloc[0]) > 0 else latest
    raw_5y = ((latest / first) ** (1.0 / years_span)) - 1.0 if first > 0 else 0.12
    if len(c) >= 150:
        idx_3y = max(0, len(c) - 156)
        w_3y = float(c.iloc[idx_3y])
        days_3y = max(365.0, float((c.index[-1] - c.index[idx_3y]).days))
        cagr_3y = ((latest / w_3y) ** (365.25 / days_3y)) - 1.0 if w_3y > 0 else raw_5y
        cagr = 0.55 * raw_5y + 0.45 * cagr_3y
    else:
        cagr = raw_5y
    return round(cagr * 100.0, 1)


def tool_weekly_stage2(closes: pd.Series, symbol: str = "") -> bool:
    """Replica of the is_weekly_stage_2 decision in strategy_engine.evaluate_weekly_trend().

    `closes` is the window of weekly GBP closes the tool would see (it requests 5 years of weekly bars):
      Stage 2 = latest >= 0.985 x EMA30 and EMA30 >= 0.995 x EMA30 four weeks earlier (EMAs: span, adjust=False).
      For the tool's crisis-shield tickers (VWRP among them) a miss is overridden when cagr_5y_pct >= 10,
      EMA50 >= 0.995 x EMA50 seven weeks earlier and latest >= 0.965 x EMA50.
      Fewer than 25 weekly bars -> False (the tool's 'data unavailable' default).
    """
    c = pd.Series(closes, dtype=float).dropna()
    c = c[c > 0]
    if len(c) < 25:
        return False
    latest = float(c.iloc[-1])
    ema30 = c.ewm(span=30, adjust=False).mean()
    ema_30w, ema_30w_prev = float(ema30.iloc[-1]), float(ema30.iloc[-5])
    stage2 = (latest >= ema_30w * 0.985) and (ema_30w >= ema_30w_prev * 0.995)
    if not stage2 and any(g in symbol.upper() for g in CRISIS_HEDGE_SYMBOLS):
        ema50 = c.ewm(span=50, adjust=False).mean()
        ema_50w, ema_50w_prev = float(ema50.iloc[-1]), float(ema50.iloc[-8])
        if _tool_cagr_5y_pct(c) >= 10.0 and ema_50w >= ema_50w_prev * 0.995 and latest >= ema_50w * 0.965:
            stage2 = True
    return bool(stage2)


def weekly_stage2_series(weekly_closes: pd.Series, symbol: str = "", start: Optional[Any] = None,
                         lookback_years: int = 5) -> pd.Series:
    """Weekly Stage 2 state for each week, each computed only from the trailing `lookback_years` of closes up to it."""
    w = pd.Series(weekly_closes, dtype=float).dropna()
    w = w[w > 0]
    idx = w.index
    start_ts = pd.Timestamp(start) if start is not None else None
    out: Dict[pd.Timestamp, bool] = {}
    for i, label in enumerate(idx):
        if start_ts is not None and label < start_ts:
            continue
        lo = idx.searchsorted(label - pd.DateOffset(years=lookback_years), side="right")
        out[label] = tool_weekly_stage2(w.iloc[lo:i + 1], symbol)
    return pd.Series(out, dtype=bool)


# ----------------------------------------------------------------------------------------------------------------
# Portfolio mechanics
# ----------------------------------------------------------------------------------------------------------------
def band_rebalance(current_w: pd.Series, target_w: pd.Series, band: float = BAND) -> Tuple[pd.Series, bool]:
    """Sleeve bands: if any sleeve is MORE than `band` away from its target, every sleeve goes back to target.

    Returns (new sleeve weights, triggered). Within the bands the drifted weights are kept (no trade).
    """
    cur = current_w.reindex(target_w.index).fillna(0.0).astype(float)
    triggered = bool(((cur - target_w).abs() > band + 1e-12).any())
    return (target_w.astype(float).copy(), True) if triggered else (cur.copy(), False)


def make_cost_fn(assets: Sequence[str], spread: float = SPREAD, fx_fee: float = FX_FEE,
                 fx_assets: Iterable[str] = SECTORS) -> Callable[[np.ndarray, np.ndarray], float]:
    """Cost of moving from old to new weights, as a fraction of portfolio value:
    spread x sum|dw| over all assets + fx_fee x sum|dw| over the FX-fee assets (the sector ETFs)."""
    fx_set = set(fx_assets)
    rates = np.array([spread + (fx_fee if a in fx_set else 0.0) for a in assets], dtype=float)

    def cost_fn(old_w: np.ndarray, new_w: np.ndarray) -> float:
        return float(np.abs(np.asarray(new_w, dtype=float) - np.asarray(old_w, dtype=float)) @ rates)

    return cost_fn


def _weights_to_array(weights: Any, col: Mapping[str, int]) -> np.ndarray:
    arr = np.zeros(len(col))
    items = weights.items() if hasattr(weights, "items") else []
    for asset, w in items:
        if asset not in col:
            raise KeyError(f"weight for unknown asset {asset!r}")
        arr[col[asset]] += float(w)
    if np.any(arr < -1e-12) or abs(arr.sum() - 1.0) > 1e-9:
        raise ValueError(f"weights must be non-negative and sum to 1, got {dict(weights)}")
    return np.clip(arr, 0.0, None)


def simulate(returns: pd.DataFrame, weight_fn: Callable[[pd.Timestamp, Optional[pd.DataFrame], Optional[pd.Series]], Any],
             cost_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
             signal_data: Optional[pd.DataFrame] = None, initial_value: float = 1.0) -> Dict[str, Any]:
    """Runs a long-only strategy through time without look-ahead.

    returns: rows d_0..d_N; row k holds each asset's return over (d_{k-1}, d_k] (row 0 is ignored).
    weight_fn(d_k, history, drifted) -> target weights decided at d_k (dict/Series summing to 1), for d_0..d_{N-1}:
        history = signal_data.loc[:d_k] (data up to and including d_k only; None if signal_data is None),
        drifted = the weights after the latest return (None at d_0, where the initial allocation is free).
    Weights decided at d_k earn the d_k -> d_{k+1} return and then drift with returns; trades are charged
    cost_fn(drifted, new) x portfolio value. No trade happens at the final date.
    Returns equity (d_0..d_N), weights held after each date, turnover (sum|dw|), cost_frac and cost_value per date.
    """
    if len(returns) < 2:
        raise ValueError("need at least two dates")
    assets = list(returns.columns)
    col = {a: i for i, a in enumerate(assets)}
    rets = returns.to_numpy(dtype=float)
    dates = returns.index
    n = len(dates)
    equity = np.empty(n)
    turnover = np.zeros(n)
    cost_frac = np.zeros(n)
    cost_value = np.zeros(n)
    held_w = np.empty((n, len(assets)))

    def history(d: pd.Timestamp) -> Optional[pd.DataFrame]:
        return None if signal_data is None else signal_data.loc[:d]

    w = _weights_to_array(weight_fn(dates[0], history(dates[0]), None), col)
    value = float(initial_value)
    equity[0] = value
    held_w[0] = w
    for k in range(1, n):
        r = rets[k]
        held = w > 1e-12
        if not np.all(np.isfinite(r[held])):
            raise ValueError(f"missing return for a held asset on {dates[k].date()}")
        growth = w * (1.0 + np.where(np.isfinite(r), r, 0.0))
        g = float(growth.sum())
        value *= g
        drifted = growth / g
        w = drifted
        if k < n - 1:
            new_w = _weights_to_array(weight_fn(dates[k], history(dates[k]), pd.Series(drifted, index=assets)), col)
            tv = float(np.abs(new_w - drifted).sum())
            if tv > 1e-10:
                c = float(cost_fn(drifted, new_w)) if cost_fn is not None else 0.0
                turnover[k], cost_frac[k], cost_value[k] = tv, c, value * c
                value *= 1.0 - c
                w = new_w
        equity[k] = value
        held_w[k] = w
    return {
        "equity": pd.Series(equity, index=dates),
        "weights": pd.DataFrame(held_w, index=dates, columns=assets),
        "turnover": pd.Series(turnover, index=dates),
        "cost_frac": pd.Series(cost_frac, index=dates),
        "cost_value": pd.Series(cost_value, index=dates),
    }


class StaticWeights:
    """S0: fixed weights (100% global equity never needs a trade)."""

    def __init__(self, target: Mapping[str, float]):
        self.target = dict(target)

    def __call__(self, date_, hist, drifted):
        return self.target


class CoreSatelliteBands:
    """S1 / S2 / S2b: sleeve-level +/-band rebalancing; the satellite (if any) is re-selected and equal-weighted monthly."""

    def __init__(self, sleeves: Mapping[str, float], core: str, gold: str, gilts: str,
                 satellite_universe: Sequence[str] = (), select: Optional[Callable[[pd.DataFrame], List[str]]] = None,
                 band: float = BAND):
        self.targets = pd.Series({k: float(v) for k, v in sleeves.items() if v > 0})
        self.core, self.gold, self.gilts = core, gold, gilts
        self.universe = list(satellite_universe)
        self.select = select
        self.band = band
        self.band_events: List[pd.Timestamp] = []
        self.selections: List[Tuple[pd.Timestamp, Tuple[str, ...]]] = []

    def __call__(self, date_, hist, drifted):
        if drifted is None:
            sleeves = self.targets.copy()
        else:
            current = pd.Series({"core": drifted.get(self.core, 0.0), "gold": drifted.get(self.gold, 0.0),
                                 "gilts": drifted.get(self.gilts, 0.0),
                                 "satellite": float(drifted.reindex(self.universe).fillna(0.0).sum()) if self.universe else 0.0})
            sleeves, hit = band_rebalance(current, self.targets, self.band)
            if hit:
                self.band_events.append(date_)
        w: Dict[str, float] = {self.core: sleeves.get("core", 0.0), self.gold: sleeves.get("gold", 0.0),
                               self.gilts: sleeves.get("gilts", 0.0)}
        if "satellite" in self.targets.index:
            names = list(self.select(hist)) if self.select is not None else list(self.universe)
            self.selections.append((date_, tuple(names)))
            for name in names:
                w[name] = w.get(name, 0.0) + float(sleeves["satellite"]) / len(names)
        return w


class SatelliteOnly:
    """100% sector sleeve (used to test momentum selection on its own): top-n by 12-1 momentum, or all names."""

    def __init__(self, universe: Sequence[str], top_n: Optional[int] = None):
        self.universe, self.top_n = list(universe), top_n

    def __call__(self, date_, hist, drifted):
        names = top_by_momentum(hist[self.universe], self.top_n) if self.top_n else self.universe
        return {n: 1.0 / len(names) for n in names}


class SmaSwitch:
    """S3: risk_on asset while the month-end signal price is above its SMA, otherwise risk_off."""

    def __init__(self, signal_col: str, risk_on: str, risk_off: str, months: int = SMA_MONTHS):
        self.signal_col, self.risk_on, self.risk_off, self.months = signal_col, risk_on, risk_off, months

    def __call__(self, date_, hist, drifted):
        return {self.risk_on: 1.0} if sma_signal(hist[self.signal_col], self.months) else {self.risk_off: 1.0}


class PrecomputedSwitch:
    """S4: risk_on/risk_off from a causal, precomputed boolean target per decision date."""

    def __init__(self, targets: pd.Series, risk_on: str, risk_off: str):
        self.targets, self.risk_on, self.risk_off = targets, risk_on, risk_off

    def __call__(self, date_, hist, drifted):
        state = self.targets.get(date_)
        if state is None or pd.isna(state):
            raise ValueError(f"no weekly signal available on {pd.Timestamp(date_).date()}")
        return {self.risk_on: 1.0} if bool(state) else {self.risk_off: 1.0}


# ----------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------
def slice_period(equity: pd.Series, start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.Series:
    """Equity curve for a calendar-year period, including the base value just before the period's first month."""
    idx = equity.index
    if start_year is not None:
        before = idx[idx < pd.Timestamp(year=start_year, month=1, day=1)]
        base = before[-1] if len(before) else idx[0]
    else:
        base = idx[0]
    end = idx[idx <= pd.Timestamp(year=end_year, month=12, day=31)][-1] if end_year is not None else idx[-1]
    return equity.loc[base:end]


def calendar_year_returns(equity: pd.Series) -> pd.Series:
    """Compounded return per calendar year (the first/last years may be partial)."""
    out = {}
    for year in sorted(set(equity.index[1:].year)):
        prior = equity[equity.index < pd.Timestamp(year=year, month=1, day=1)]
        start_val = float(prior.iloc[-1]) if len(prior) else float(equity.iloc[0])
        upto = equity[equity.index <= pd.Timestamp(year=year, month=12, day=31)]
        out[year] = float(upto.iloc[-1]) / start_val - 1.0
    return pd.Series(out, dtype=float)


def compute_metrics(equity: pd.Series, benchmark: Optional[pd.Series] = None, turnover: Optional[pd.Series] = None,
                    cost_frac: Optional[pd.Series] = None, cost_value: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Performance metrics of a month-end equity curve (first value = start of the period).

    Trades dated after the first and up to the last date of the curve count towards turnover and costs.
    """
    eq = equity.dropna().astype(float)
    n = len(eq) - 1
    if n < 1:
        raise ValueError("need at least one monthly return")
    years = n / 12.0
    r = eq.pct_change(fill_method=None).dropna()
    cagr = (float(eq.iloc[-1]) / float(eq.iloc[0])) ** (1.0 / years) - 1.0
    vol = float(r.std(ddof=1) * math.sqrt(12)) if n > 1 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)) * math.sqrt(12))
    sortino = float(r.mean() * 12 / downside) if downside > 0 else float("nan")
    drawdown = eq / eq.cummax() - 1.0
    max_dd = float(drawdown.min())
    dd_trough = drawdown.idxmin()
    dd_peak = eq.loc[:dd_trough].idxmax()
    roll12 = (eq / eq.shift(12) - 1.0).dropna()
    roll60 = ((eq / eq.shift(60)) ** (1.0 / 5.0) - 1.0).dropna()
    out: Dict[str, Any] = {
        "start": eq.index[0], "end": eq.index[-1], "months": n, "years": years,
        "cagr": cagr, "vol": vol, "sortino": sortino, "max_dd": max_dd, "dd_peak": dd_peak, "dd_trough": dd_trough,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else float("nan"),
        "worst_12m": float(roll12.min()) if len(roll12) else float("nan"),
        "roll5_min": float(roll60.min()) if len(roll60) else float("nan"),
        "roll5_median": float(roll60.median()) if len(roll60) else float("nan"),
        "roll5_max": float(roll60.max()) if len(roll60) else float("nan"),
        "final_value_per_10k": 10000.0 * float(eq.iloc[-1]) / float(eq.iloc[0]),
    }
    if turnover is not None:
        mask = (turnover.index > eq.index[0]) & (turnover.index <= eq.index[-1])
        out["turnover_one_way_pa"] = float(turnover[mask].sum()) / 2.0 / years
        out["trades"] = int((turnover[mask] > 1e-10).sum())
    if cost_frac is not None:
        mask = (cost_frac.index > eq.index[0]) & (cost_frac.index <= eq.index[-1])
        out["cost_drag_pa"] = float(cost_frac[mask].sum()) / years
    if cost_value is not None:
        mask = (cost_value.index > eq.index[0]) & (cost_value.index <= eq.index[-1])
        out["costs_per_10k"] = 10000.0 * float(cost_value[mask].sum()) / float(eq.iloc[0])
    if benchmark is not None:
        mine, theirs = calendar_year_returns(eq), calendar_year_returns(benchmark.loc[eq.index[0]:eq.index[-1]])
        common = mine.index.intersection(theirs.index)
        wins = int((mine[common] > theirs[common] + 1e-12).sum())
        out["years_beating"] = wins
        out["years_total"] = int(len(common))
        out["pct_years_beating"] = wins / len(common) if len(common) else float("nan")
    return out


def difference_stats(strategy: pd.Series, control: pd.Series) -> Dict[str, float]:
    """Monthly return difference (strategy - control): annualised mean, t-stat and share of months ahead."""
    d = (strategy.pct_change(fill_method=None) - control.pct_change(fill_method=None)).dropna()
    if len(d) < 3:
        return {"ann_diff": float("nan"), "t_stat": float("nan"), "hit_rate": float("nan"), "months": len(d)}
    sd = float(d.std(ddof=1))
    return {"ann_diff": float(d.mean() * 12), "t_stat": float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else float("nan"),
            "hit_rate": float((d > 0).mean()), "months": int(len(d))}


# ----------------------------------------------------------------------------------------------------------------
# Data preparation, strategies and the full run
# ----------------------------------------------------------------------------------------------------------------
def prepare_data(closes: Mapping[str, pd.Series], today: Optional[Any] = None) -> Dict[str, Any]:
    """Cleans the raw daily closes and builds the daily and month-end GBP panels plus data notes."""
    missing = [t for t in REQUIRED_TICKERS if closes.get(t) is None or len(pd.Series(closes[t]).dropna()) == 0]
    if missing:
        raise ValueError(f"missing required price history: {', '.join(missing)}")
    notes: Dict[str, Any] = {"glitches": {}, "monthly_flags": {}, "signal_proxy": None}
    raw: Dict[str, pd.Series] = {}
    for ticker, series in closes.items():
        if series is None:
            continue
        s = pd.to_numeric(pd.Series(series), errors="coerce").dropna().sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if s.empty:
            continue
        if ticker.upper().endswith(".L"):
            fixed, events = repair_price_glitches(s)
            notes["glitches"][ticker] = events
            s = fixed.dropna()
        raw[ticker] = s

    fx_start = raw[FX].index[0]
    calendar = reduce(lambda a, b: a.union(b), [s.index for s in raw.values()])
    calendar = calendar[calendar >= fx_start]

    def on_calendar(s: pd.Series) -> pd.Series:
        return s.reindex(s.index.union(calendar)).ffill().reindex(calendar)

    fx = on_calendar(raw[FX])
    daily: Dict[str, pd.Series] = {t: usd_to_gbp(on_calendar(raw[t]), fx) for t in USD_TICKERS}
    daily[SHY_USD] = on_calendar(raw[SHY])
    daily[FX] = fx
    daily[IGLS] = on_calendar(raw[IGLS])
    daily[GILTS] = splice_series(daily[IGLS], daily[SHY])
    daily[GILTS_HEDGED] = splice_series(daily[IGLS], daily[SHY_USD])
    daily[EQUITY_SIGNAL] = daily[EQUITY].copy()
    for ticker, drift, label in SIGNAL_PROXIES:
        if ticker in raw:
            proxy_gbp = usd_to_gbp(on_calendar(raw[ticker]), fx)
            daily[EQUITY_SIGNAL] = extend_with_proxy(daily[EQUITY], proxy_gbp, drift).reindex(calendar)
            daily["PROXY_" + ticker] = proxy_gbp
            notes["signal_proxy"] = {"ticker": ticker, "drift": drift, "label": label}
            break
    for ticker in CROSS_CHECK_TICKERS:
        if ticker in raw:
            daily[ticker] = on_calendar(raw[ticker])
    daily_df = pd.DataFrame(daily)

    today_ts = pd.Timestamp(today if today is not None else date.today()).normalize()
    monthly = daily_df.resample("ME").last()
    monthly = monthly[monthly.index < today_ts.replace(day=1)]  # drop the current, incomplete month
    if monthly.empty:
        raise ValueError("no complete month of data")
    end = monthly.index[-1]
    daily_df = daily_df[daily_df.index <= end]

    for ticker in list(USD_TICKERS) + [IGLS] + [t for t in CROSS_CHECK_TICKERS if t in daily_df]:
        source = daily_df[SHY_USD] if ticker == SHY else daily_df[ticker]
        flags = flag_monthly_outliers(source.resample("ME").last())
        if flags:
            notes["monthly_flags"][ticker] = flags

    weekly = daily_df[EQUITY_SIGNAL].dropna().resample("W-FRI").last().dropna()
    required = [EQUITY, GOLD, GILTS, GILTS_HEDGED] + list(SECTORS)
    ok = monthly[required].notna().all(axis=1)
    ok &= monthly[list(SECTORS)].shift(12).notna().all(axis=1)
    ok &= monthly[EQUITY_SIGNAL].notna().rolling(SMA_MONTHS).sum().eq(SMA_MONTHS)
    if len(weekly) < 25:
        raise ValueError("not enough weekly history for the S4 signal")
    ok &= monthly.index >= weekly.index[24]
    if not ok.any():
        raise ValueError("the required series never overlap")
    start = ok[ok].index[0]
    return {"daily": daily_df, "monthly": monthly, "weekly_signal_closes": weekly, "start": start, "end": end,
            "notes": notes, "raw_first_dates": {t: s.index[0] for t, s in raw.items()},
            "igls_first_date": raw[IGLS].index[0]}


def _weekly_targets(daily: pd.DataFrame, weekly: pd.Series, start: pd.Timestamp, symbol: str) -> pd.Series:
    """Weekly Stage 2 states re-indexed to each week's actual last trading date (when the decision is made)."""
    signals = weekly_stage2_series(weekly, symbol=symbol, start=start - pd.Timedelta(days=13))
    dates = pd.Series(daily.index, index=daily.index)[daily[EQUITY_SIGNAL].notna()]
    last_dates = dates.resample("W-FRI").last().reindex(signals.index)
    return pd.Series(signals.to_numpy(), index=pd.DatetimeIndex(last_dates.to_numpy())).sort_index()


def run_strategies(data: Mapping[str, Any], gilts: str = GILTS,
                   weekly_targets: Optional[Mapping[str, pd.Series]] = None) -> Dict[str, Dict[str, Any]]:
    """Simulates S0-S4 for one choice of gilt series. Returns per strategy the simulate() output (monthly equity)."""
    monthly, daily, start, end = data["monthly"], data["daily"], data["start"], data["end"]
    assets = [EQUITY, GOLD, gilts] + list(SECTORS)
    m_ret = monthly.loc[start:end, assets].pct_change(fill_method=None)
    signal_data = monthly.loc[:end]
    cost_fn = make_cost_fn(assets)
    sector_cols = list(SECTORS)
    specs: Dict[str, Any] = {
        "S0": StaticWeights({EQUITY: 1.0}),
        "S1": CoreSatelliteBands(S1_SLEEVES, EQUITY, GOLD, gilts),
        "S2": CoreSatelliteBands(S2_SLEEVES, EQUITY, GOLD, gilts, SECTORS,
                                 select=lambda hist: top_by_momentum(hist[sector_cols], MOM_TOP_N)),
        "S2b": CoreSatelliteBands(S2_SLEEVES, EQUITY, GOLD, gilts, SECTORS, select=None),
        "S3": SmaSwitch(EQUITY_SIGNAL, EQUITY, gilts),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for code, fn in specs.items():
        sim = simulate(m_ret, fn, cost_fn, signal_data=signal_data)
        sim["fn"] = fn
        out[code] = sim

    if weekly_targets is None:
        weekly_targets = {"S4": _weekly_targets(daily, data["weekly_signal_closes"], start, "VWRP.L"),
                          "S4-pure": _weekly_targets(daily, data["weekly_signal_closes"], start, "ACWI")}
    daily_start = daily.index[daily.index <= start][-1]
    d_idx = daily.index[(daily.index >= daily_start) & (daily.index <= end)]
    d_ret = daily.loc[d_idx, [EQUITY, gilts]].pct_change(fill_method=None)
    d_cost = make_cost_fn([EQUITY, gilts])
    for code, targets in weekly_targets.items():
        fn = PrecomputedSwitch(targets.reindex(d_idx, method="ffill"), EQUITY, gilts)
        sim = simulate(d_ret, fn, d_cost)
        sim["daily_equity"] = sim["equity"]
        monthly_eq = sim["equity"].resample("ME").last()
        sim["equity"] = monthly_eq.loc[start:end]
        sim["targets"] = targets
        sim["fn"] = fn
        out[code] = sim
    lengths = {code: len(sim["equity"]) for code, sim in out.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"strategy equity curves are misaligned: {lengths}")
    return out


def _strategy_extras(code: str, sim: Mapping[str, Any], years: float, equity_col: str) -> Dict[str, Any]:
    fn = sim.get("fn")
    extras: Dict[str, Any] = {}
    if isinstance(fn, CoreSatelliteBands):
        extras["band_rebalances"] = len(fn.band_events)
        extras["band_dates"] = [d.strftime("%Y-%m") for d in fn.band_events]
        if fn.select is not None and fn.selections:
            changes = sum(len(set(b) - set(a)) for (_, a), (_, b) in zip(fn.selections[:-1], fn.selections[1:]))
            extras["satellite_name_changes_pa"] = changes / years
            extras["latest_satellite"] = list(fn.selections[-1][1])
    if code in ("S3", "S4", "S4-pure"):
        w = sim["weights"]
        on = w[equity_col] > 0.5
        extras["switches"] = int((sim["turnover"] > 1.5).sum())
        extras["time_in_equity"] = float(on.iloc[:-1].mean()) if len(on) > 1 else float("nan")
        extras["currently_in_equity"] = bool(on.iloc[-1])
    return extras


def evaluate(strategies: Mapping[str, Mapping[str, Any]], periods: Sequence[Tuple[str, Optional[int], Optional[int]]] = PERIODS,
             benchmark: str = "S0") -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Metrics per strategy and period."""
    bench_eq = strategies[benchmark]["equity"]
    table: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for code, sim in strategies.items():
        table[code] = {}
        for name, y0, y1 in periods:
            eq = slice_period(sim["equity"], y0, y1)
            if len(eq) < 2:
                continue
            m = compute_metrics(eq, None if code == benchmark else bench_eq, sim["turnover"], sim["cost_frac"], sim["cost_value"])
            table[code][name] = m
    for code in table:
        for name, m in table[code].items():
            b = table[benchmark].get(name)
            m["cagr_gap"] = m["cagr"] - b["cagr"] if b else float("nan")
    return table


def _pair_stats(monthly: pd.DataFrame, a: str, b: str, end: pd.Timestamp) -> Optional[Dict[str, Any]]:
    if a not in monthly or b not in monthly:
        return None
    x = monthly[[a, b]].loc[:end].dropna()
    r = x.pct_change(fill_method=None).dropna()
    if len(r) < 24:
        return None
    years = len(r) / 12.0
    return {"from": x.index[0], "to": x.index[-1], "months": int(len(r)), "corr": float(r[a].corr(r[b])),
            "cagr_a": float((x[a].iloc[-1] / x[a].iloc[0]) ** (1 / years) - 1),
            "cagr_b": float((x[b].iloc[-1] / x[b].iloc[0]) ** (1 / years) - 1),
            "vol_a": float(r[a].std() * math.sqrt(12)), "vol_b": float(r[b].std() * math.sqrt(12))}


def data_checks(data: Mapping[str, Any]) -> Dict[str, Any]:
    monthly, daily, start, end = data["monthly"], data["daily"], data["start"], data["end"]
    checks: Dict[str, Any] = {
        "vwrl_vs_acwi": _pair_stats(monthly, "VWRL.L", EQUITY, end),
        "sgln_vs_gld": _pair_stats(monthly, "SGLN.L", GOLD, end),
        "igls_vs_shy_gbp": _pair_stats(monthly, IGLS, SHY, end),
        "igls_vs_shy_usd": _pair_stats(monthly, IGLS, SHY_USD, end),
    }
    proxy = data["notes"].get("signal_proxy")
    if proxy:
        checks["proxy_vs_acwi"] = _pair_stats(monthly, EQUITY, "PROXY_" + proxy["ticker"], end)
    splice_end = data["igls_first_date"]
    seg = daily.loc[start:splice_end]
    if len(seg) > 1:
        checks["splice"] = {"from": seg.index[0], "to": seg.index[-1],
                            "shy_gbp": float(seg[SHY].iloc[-1] / seg[SHY].iloc[0] - 1),
                            "shy_usd": float(seg[SHY_USD].iloc[-1] / seg[SHY_USD].iloc[0] - 1)}
    return checks


def run_all(closes: Mapping[str, pd.Series], today: Optional[Any] = None) -> Dict[str, Any]:
    """Full backtest from raw daily closes: main run, gilt-splice sensitivity, momentum sleeve test, data checks."""
    data = prepare_data(closes, today=today)
    start, end = data["start"], data["end"]
    weekly_targets = {"S4": _weekly_targets(data["daily"], data["weekly_signal_closes"], start, "VWRP.L"),
                      "S4-pure": _weekly_targets(data["daily"], data["weekly_signal_closes"], start, "ACWI")}
    main = run_strategies(data, GILTS, weekly_targets)
    hedged = run_strategies(data, GILTS_HEDGED, weekly_targets)
    metrics = evaluate(main)
    metrics_hedged = evaluate(hedged)
    years = metrics["S0"]["Full period"]["years"]
    extras = {code: _strategy_extras(code, sim, years, EQUITY) for code, sim in main.items()}

    # Momentum selection on its own: 100% top-3 sector sleeve vs 100% equal-weight 9-sector sleeve,
    # after costs (0.05% + 0.15% FX on every trade) and, for reference, before costs.
    monthly = data["monthly"]
    sector_ret = monthly.loc[start:end, list(SECTORS)].pct_change(fill_method=None)
    sector_cost = make_cost_fn(list(SECTORS))
    sig = monthly.loc[:end, list(SECTORS)]
    sleeves = {
        "momentum": simulate(sector_ret, SatelliteOnly(SECTORS, MOM_TOP_N), sector_cost, signal_data=sig),
        "equal_weight": simulate(sector_ret, SatelliteOnly(SECTORS), sector_cost, signal_data=sig),
        "momentum_gross": simulate(sector_ret, SatelliteOnly(SECTORS, MOM_TOP_N), None, signal_data=sig),
        "equal_weight_gross": simulate(sector_ret, SatelliteOnly(SECTORS), None, signal_data=sig),
    }
    sleeve: Dict[str, Dict[str, Any]] = {k: {} for k in list(sleeves) + ["diff", "diff_gross", "portfolio_diff"]}
    for name, y0, y1 in PERIODS:
        eqs = {k: slice_period(sim["equity"], y0, y1) for k, sim in sleeves.items()}
        if len(eqs["momentum"]) < 2:
            continue
        for k, sim in sleeves.items():
            sleeve[k][name] = compute_metrics(eqs[k], None, sim["turnover"], sim["cost_frac"], sim["cost_value"])
        sleeve["diff"][name] = difference_stats(eqs["momentum"], eqs["equal_weight"])
        sleeve["diff_gross"][name] = difference_stats(eqs["momentum_gross"], eqs["equal_weight_gross"])
        # Whole-portfolio effect of the momentum satellite: S2 vs its no-momentum control S2b (after costs)
        sleeve["portfolio_diff"][name] = difference_stats(slice_period(main["S2"]["equity"], y0, y1),
                                                          slice_period(main["S2b"]["equity"], y0, y1))

    # Context: the same global equity index measured in USD (sterling usually falls in crises, cushioning GBP investors)
    window = monthly.loc[start:end]
    acwi_usd = window[EQUITY] * window[FX]
    context = {"acwi_usd_max_dd": float((acwi_usd / acwi_usd.cummax() - 1.0).min()),
               "acwi_gbp_max_dd": float((window[EQUITY] / window[EQUITY].cummax() - 1.0).min())}
    return {
        "meta": {"start": start, "end": end, "months": int(metrics["S0"]["Full period"]["months"]),
                 "generated": date.today().isoformat(), "spread": SPREAD, "fx_fee": FX_FEE, "band": BAND},
        "metrics": metrics, "metrics_hedged": metrics_hedged, "extras": extras, "sleeve": sleeve, "context": context,
        "checks": data_checks(data), "notes": data["notes"], "igls_first_date": data["igls_first_date"],
        "equity": {code: sim["equity"] for code, sim in main.items()},
        "equity_hedged": {code: sim["equity"] for code, sim in hedged.items()},
    }


# ----------------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------------
def _pct(x: Any, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{digits}f}%"


def _pp(x: Any, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{digits}f} pp"


def _num(x: Any, digits: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], align: Optional[str] = None) -> str:
    align = align or ("l" + "r" * (len(headers) - 1))
    marks = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(marks[a] for a in align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _month(ts: Any) -> str:
    return pd.Timestamp(ts).strftime("%b %Y")


def _period_title(name: str, m: Mapping[str, Any]) -> str:
    first_month = (pd.Timestamp(m["start"]) + pd.offsets.MonthEnd(1))
    return f"{name.replace('latest', _month(m['end']))} ({_month(first_month)} – {_month(m['end'])}, {m['months']} months)"


def _years_beating(m: Mapping[str, Any]) -> str:
    if "years_total" not in m:
        return "—"
    return f"{m['years_beating']}/{m['years_total']}"


def render_markdown(results: Mapping[str, Any]) -> str:
    """User-facing report (plain English, tables first)."""
    met, meta, extras, sleeve = results["metrics"], results["meta"], results["extras"], results["sleeve"]
    checks, notes = results["checks"], results["notes"]
    codes = [c for c in STRATEGY_ORDER if c in met]
    full = {c: met[c]["Full period"] for c in codes}
    s0 = full["S0"]
    lines: List[str] = []
    lines.append("# Phase 2 backtest: do simple rules beat just holding a global index fund?")
    lines.append("")
    lines.append(f"*{_month(s0['start'] + pd.offsets.MonthEnd(1))} – {_month(meta['end'])} ({meta['months']} months), "
                 f"in GBP, after trading costs, no tax (ISA). Generated {meta['generated']} by `scripts/backtest.py`. "
                 "Past performance on one historical path; not a forecast.*")
    lines.append("")
    lines.append("## Headline (full period, after costs)")
    lines.append("")
    rows = []
    for c in codes:
        m = full[c]
        rows.append([f"**{c}** {STRATEGY_LABELS[c]}", _pct(m["cagr"]), _pct(m["max_dd"]), _num(m["sortino"]),
                     _pct(m["worst_12m"]), "—" if c == "S0" else _pp(m["cagr_gap"])])
    lines.append(_table(["Strategy", "CAGR", "Max drawdown", "Sortino", "Worst 12m", "Gap vs S0"], rows))
    lines.append("")
    lines.append(f"£10,000 in S0 at the start grew to about £{s0['final_value_per_10k']:,.0f}. "
                 "CAGR = compound annual growth rate; drawdown = fall from a previous peak (month-end values); "
                 "Sortino = return per unit of downside volatility (higher is better).")
    lines.append("")
    lines.append("### Full-period detail")
    lines.append("")
    rows = []
    for c in codes:
        m = full[c]
        e = extras.get(c, {})
        trades = (f"{e['band_rebalances']} band resets" if "band_rebalances" in e else
                  f"{e['switches']} switches, {e['time_in_equity'] * 100:.0f}% in equity" if "switches" in e else "none")
        rows.append([c, _pct(m["vol"]), _num(m["calmar"]), _pct(m.get("turnover_one_way_pa", 0.0), 0),
                     _pct(m.get("cost_drag_pa", 0.0), 2), f"£{m.get('costs_per_10k', 0.0):,.0f}",
                     _years_beating(m),
                     f"{_pct(m['roll5_min'])} / {_pct(m['roll5_median'])} / {_pct(m['roll5_max'])}", trades])
    lines.append(_table(["", "Volatility", "Calmar", "Turnover/yr", "Costs/yr", "Costs per £10k", "Years beating S0",
                         "Rolling 5y CAGR min / median / max", "Trading"], rows))
    lines.append("")
    lines.append("Turnover = share of the portfolio replaced per year (a full switch counts as 100%). Costs per £10k = "
                 "total costs paid over the period per £10,000 held at its start. Years beating S0 counts calendar "
                 "years (first and last are partial).")
    lines.append("")
    lines.append("## Sub-periods")
    for name, _, _ in PERIODS[1:]:
        if name not in met["S0"]:
            continue
        lines.append("")
        lines.append(f"### {_period_title(name, met['S0'][name])}")
        lines.append("")
        rows = []
        for c in codes:
            m = met[c].get(name)
            if not m:
                continue
            rows.append([c, _pct(m["cagr"]), _pct(m["max_dd"]), _num(m["sortino"]), _pct(m["worst_12m"]),
                         "—" if c == "S0" else _pp(m["cagr_gap"]), _years_beating(m), _pct(m["vol"]), _num(m["calmar"]),
                         f"{_pct(m['roll5_min'])} / {_pct(m['roll5_median'])} / {_pct(m['roll5_max'])}",
                         _pct(m.get("turnover_one_way_pa", 0.0), 0), _pct(m.get("cost_drag_pa", 0.0), 2)])
        lines.append(_table(["", "CAGR", "Max DD", "Sortino", "Worst 12m", "Gap vs S0", "Years > S0", "Vol", "Calmar",
                             "Rolling 5y min / med / max", "Turnover/yr", "Costs/yr"], rows))
    lines.append("")
    lines.append(_render_momentum_section(results))
    lines.append("")
    lines.append(_render_splice_section(results))
    lines.append("")
    lines.append(_render_meaning(results))
    lines.append("")
    lines.append(_render_method_and_caveats(results))
    lines.append("")
    lines.append(_render_data_checks(results))
    return "\n".join(lines).rstrip() + "\n"


def _render_momentum_section(results: Mapping[str, Any]) -> str:
    sleeve = results["sleeve"]
    lines = ["## Momentum on its own: top 3 sectors vs all 9", "",
             "The satellite is only 20% of S2, so its effect on the whole portfolio is diluted. Here the two sector "
             "sleeves are compared at 100% weight, after the same costs (0.05% spread + 0.15% FX on every trade).", ""]
    rows = []
    for name, _, _ in PERIODS:
        if name not in sleeve["diff"]:
            continue
        mo, ew, d, pdiff = sleeve["momentum"][name], sleeve["equal_weight"][name], sleeve["diff"][name], sleeve["portfolio_diff"][name]
        label = name.replace("latest", _month(mo["end"]))
        rows.append([label, _pct(mo["cagr"]), _pct(ew["cagr"]), _pp(mo["cagr"] - ew["cagr"]), _num(d["t_stat"], 1),
                     _pct(d["hit_rate"], 0), _pct(mo["max_dd"]), _pct(ew["max_dd"]), _pp(pdiff["ann_diff"], 2)])
    lines.append(_table(["Period", "Top-3 momentum CAGR", "All-9 CAGR", "Difference", "t-stat", "Months ahead",
                         "Top-3 max DD", "All-9 max DD", "S2 − S2b (avg pp/yr)"], rows))
    lines.append("")
    lines.append("A t-stat below about 2 means the difference is within what luck alone would often produce.")
    return "\n".join(lines)


def _render_splice_section(results: Mapping[str, Any]) -> str:
    met, hed, checks = results["metrics"], results["metrics_hedged"], results["checks"]
    sp = checks.get("splice")
    lines = ["## Does the pre-2009 gilt proxy matter?", ""]
    if sp:
        lines.append(f"IGLS.L (UK 0-5y gilts) starts on {pd.Timestamp(sp['to']).strftime('%d %b %Y')}. Before that, the "
                     f"instructed proxy is SHY converted to GBP, which returned {_pct(sp['shy_gbp'])} over the splice "
                     f"window versus {_pct(sp['shy_usd'])} in USD: almost all of it was sterling's 2008 collapse, a "
                     "gain a UK gilt holder did not get. The table re-runs everything with SHY's USD return "
                     "(roughly a currency-hedged holding) instead.")
        lines.append("")
    rows = []
    for c in [c for c in STRATEGY_ORDER if c in met]:
        a, b = met[c]["Full period"], hed[c]["Full period"]
        a8, b8 = met[c].get("2008-2014"), hed[c].get("2008-2014")
        rows.append([c, _pct(a["cagr"]), _pct(b["cagr"]), _pct(a["max_dd"]), _pct(b["max_dd"]),
                     _pct(a8["cagr"]) if a8 else "n/a", _pct(b8["cagr"]) if b8 else "n/a"])
    lines.append(_table(["", "CAGR (SHY in GBP)", "CAGR (SHY in USD)", "Max DD (GBP)", "Max DD (USD)",
                         "2008-14 CAGR (GBP)", "2008-14 CAGR (USD)"], rows))
    return "\n".join(lines)


def _consistency(met: Mapping[str, Any], a: str, b: str) -> Tuple[int, int]:
    names = [n for n, _, _ in PERIODS[1:] if n in met[a] and n in met[b]]
    wins = sum(1 for n in names if met[a][n]["cagr"] > met[b][n]["cagr"])
    return wins, len(names)


def phase3_verdict(results: Mapping[str, Any]) -> Tuple[bool, str]:
    """Pre-set rule: switch to rules-based momentum only if S2 beats its control S2b after costs over the full period
    AND in every sub-period, AND the sleeve-level momentum premium is statistically meaningful (t-stat >= 2)."""
    met, sleeve = results["metrics"], results["sleeve"]
    full_gap = met["S2"]["Full period"]["cagr"] - met["S2b"]["Full period"]["cagr"]
    wins, total = _consistency(met, "S2", "S2b")
    t = sleeve["diff"]["Full period"]["t_stat"]
    ok = full_gap > 0 and wins == total and total > 0 and t >= 2.0
    reason = (f"S2 vs S2b: {full_gap * 100:+.2f} pp/yr over the full period, ahead in {wins} of {total} sub-periods; "
              f"sleeve-level t-stat {t:.1f}")
    return ok, reason


def _render_meaning(results: Mapping[str, Any]) -> str:
    met, extras, sleeve, hed = results["metrics"], results["extras"], results["sleeve"], results["metrics_hedged"]
    f = {c: met[c]["Full period"] for c in met}
    lines = ["## What this means", ""]

    # (1) Momentum
    gap_b = f["S2"]["cagr"] - f["S2b"]["cagr"]
    w_b, n_b = _consistency(met, "S2", "S2b")
    w_1, n_1 = _consistency(met, "S2", "S1")
    w_0, n_0 = _consistency(met, "S2", "S0")
    d = sleeve["diff"]["Full period"]
    mo, ew = sleeve["momentum"]["Full period"], sleeve["equal_weight"]["Full period"]
    sub_gaps = ", ".join(f"{n.replace('latest', _month(met['S0'][n]['end']))}: {_pp(met['S2'][n]['cagr'] - met['S2b'][n]['cagr'], 2)}"
                         for n, _, _ in PERIODS[1:] if n in met["S2"])
    consistent = gap_b > 0 and w_b == n_b
    lines.append("**1. Does momentum sector selection (S2) add value after costs?** "
                 + ("Yes, consistently. " if consistent and d["t_stat"] >= 2 else
                    "Not convincingly. " if gap_b > 0 else "No. ")
                 + f"Over the full period S2 returned {_pct(f['S2']['cagr'])} a year against {_pct(f['S2b']['cagr'])} for "
                 f"its no-momentum control S2b ({_pp(gap_b, 2)}); by sub-period: {sub_gaps}. "
                 f"At 100% weight the top-3 momentum sleeve made {_pct(mo['cagr'])} a year vs {_pct(ew['cagr'])} for all 9 "
                 f"sectors (t-stat {d['t_stat']:.1f}, ahead in {d['hit_rate'] * 100:.0f}% of months), with "
                 f"{mo.get('turnover_one_way_pa', 0) * 100:.0f}% turnover a year. "
                 f"Against the simpler portfolios, S2 beat S1 in {w_1} of {n_1} and S0 in {w_0} of {n_0} sub-periods "
                 f"(full period: {_pp(f['S2']['cagr_gap'])} vs S0, {_pp(f['S2']['cagr'] - f['S1']['cagr'])} vs S1).")
    lines.append("")

    # (2) Defensive sleeve
    s1, s0 = f["S1"], f["S0"]
    dd_better = [n for n, _, _ in PERIODS[1:] if n in met["S1"] and met["S1"][n]["max_dd"] > met["S0"][n]["max_dd"]]
    so_better = [n for n, _, _ in PERIODS[1:] if n in met["S1"] and met["S1"][n]["sortino"] > met["S0"][n]["sortino"]]
    n_sub = len([n for n, _, _ in PERIODS[1:] if n in met["S1"]])
    lines.append("**2. Does a 15% defensive sleeve (S1) improve risk-adjusted returns and drawdowns?** "
                 f"Over the full period S1's Sortino was {_num(s1['sortino'])} vs {_num(s0['sortino'])}, its worst fall "
                 f"{_pct(s1['max_dd'])} vs {_pct(s0['max_dd'])} and its worst 12 months {_pct(s1['worst_12m'])} vs "
                 f"{_pct(s0['worst_12m'])}, for a CAGR cost of {_pp(-s1['cagr_gap'])} a year "
                 f"({_pct(s1['cagr'])} vs {_pct(s0['cagr'])}). The drawdown was smaller in {len(dd_better)} of {n_sub} "
                 f"sub-periods and the Sortino higher in {len(so_better)} of {n_sub}. "
                 f"With the hedged gilt proxy S1's CAGR is {_pct(hed['S1']['Full period']['cagr'])}.")
    lines.append("")

    # (3) Trend filters
    parts = []
    for c in ("S3", "S4", "S4-pure"):
        if c not in f:
            continue
        m, e, h = f[c], extras.get(c, {}), hed[c]["Full period"]
        dd_sub = sum(1 for n, _, _ in PERIODS[1:] if n in met[c] and met[c][n]["max_dd"] > met["S0"][n]["max_dd"])
        parts.append(f"{c}: max drawdown {_pct(m['max_dd'])} vs {_pct(s0['max_dd'])}, worst 12 months "
                     f"{_pct(m['worst_12m'])}, CAGR {_pct(m['cagr'])} ({_pp(m['cagr_gap'])} vs S0; {_pp(h['cagr'] - hed['S0']['Full period']['cagr'])} "
                     f"with the hedged gilt proxy), {e.get('switches', 0)} switches, "
                     f"{m.get('turnover_one_way_pa', 0) * 100:.0f}% turnover and {m.get('cost_drag_pa', 0) * 100:.2f}% costs a year, "
                     f"smaller drawdown than S0 in {dd_sub} of {n_sub} sub-periods")
    lines.append("**3. Do the trend filters (S3/S4) reduce drawdowns, and at what cost?** " + "; ".join(parts) + ".")
    lines.append("")
    ok, reason = phase3_verdict(results)
    lines.append(f"**Bottom line for Phase 3 (rules-based momentum for satellite selection): {'YES' if ok else 'NO'}.** "
                 f"Pre-set test: S2 must beat S2b over the full period and in every sub-period, with a sleeve-level "
                 f"t-stat of at least 2. Result: {reason}.")
    return "\n".join(lines)


def _render_method_and_caveats(results: Mapping[str, Any]) -> str:
    notes = results["notes"]
    proxy = notes.get("signal_proxy")
    proxy_txt = (f"Before ACWI launched (March 2008) the S3/S4 signals are warmed up on the {proxy['label']} "
                 f"({proxy['ticker']}); this affects signals only, never returns." if proxy else
                 "No pre-launch proxy was available, so signals start once ACWI has enough history.")
    return "\n".join([
        "## Method in brief", "",
        "- Daily adjusted closes (dividends reinvested) from Yahoo Finance, converted to GBP (USD price ÷ GBPUSD) and "
        "sampled at month-end; the current incomplete month is dropped.",
        "- Signals at a month-end use only data up to that month-end; the resulting weights earn the next month's "
        "return and drift with the market until the next trade. S4 uses Friday closes and daily returns, switching "
        "from the next week.",
        "- Costs: 0.05% of every trade, plus 0.15% FX on sector-ETF trades (core and defensive sleeves are assumed to be "
        "held via GBP lines VWRP.L, SGLN.L and IGLS.L). No extra fund fees: proxy prices already include theirs.",
        "- Fixed textbook parameters (85/7.5/7.5, 65/20/15, ±5pp bands, 12-1 momentum top 3, 10-month SMA, the "
        "tool's 30-week EMA rule). Nothing was fitted, so the whole window is out-of-sample for the parameters.",
        "- S4 replicates `is_weekly_stage_2` from `strategy_engine.evaluate_weekly_trend`: price ≥ 98.5% of the "
        "30-week EMA and the EMA ≥ 99.5% of its value 4 weeks earlier, on 5 years of weekly GBP closes. Because the "
        "tool exempts VWRP via a 50-week 'crisis shield' (stays in when the 5y CAGR ≥ 10%, the 50-week EMA is rising "
        "and price ≥ 96.5% of it), S4 includes that exception and S4-pure shows the rule without it.",
        "",
        "## Caveats", "",
        "- **Price-only rules.** Free data has no point-in-time fundamentals, so the tool's quality screens are untested.",
        "- **Sector ETFs, not stocks.** Deliberate: back-testing today's hand-picked watchlist would bake in "
        "survivorship bias. But it means single-stock picking is not tested here.",
        "- **Gilt splice.** UK short gilts only from April 2009 (IGLS.L); before that SHY converted to GBP, which mostly "
        "measures sterling's 2008 fall. See the robustness table: treat 2008-09 results for anything holding gilts "
        "with care.",
        "- **ACWI stands in for VWRP** (similar index, slightly higher fee). " + proxy_txt,
        "- **One historical path.** 18 years with one big crash, sterling shocks and a long US-tech-led bull market; "
        "a different path could rank the strategies differently. Month-end sampling hides intra-month drawdowns; "
        "trades are assumed at the signal close.",
        "- **Hindsight.** The strategy menu was chosen in 2026 by people who know how 2008-2026 went.",
    ])


def _render_data_checks(results: Mapping[str, Any]) -> str:
    notes, checks = results["notes"], results["checks"]
    lines = ["## Data checks", ""]
    for ticker, events in notes.get("glitches", {}).items():
        if not events:
            lines.append(f"- {ticker}: no unit glitches found.")
            continue
        dropped = [e for e in events if e["action"] == "dropped"]
        rescaled = [e for e in events if e["action"] != "dropped"]
        span = f"{pd.Timestamp(events[0]['date']).date()} to {pd.Timestamp(events[-1]['date']).date()}"
        lines.append(f"- {ticker}: {len(dropped)} glitched daily prices dropped and {len(rescaled)} rescaled "
                     f"({span}); none remain after repair.")
    flags = notes.get("monthly_flags", {})
    lines.append("- Monthly returns beyond ±50% after repair: " +
                 ("none." if not flags else "; ".join(f"{t}: " + ", ".join(f"{d.date()} {v:+.0%}" for d, v in fl)
                                                     for t, fl in flags.items())))
    labels = {"vwrl_vs_acwi": ("VWRL.L", "ACWI in GBP"), "sgln_vs_gld": ("SGLN.L", "GLD in GBP"),
              "igls_vs_shy_gbp": ("IGLS.L", "SHY in GBP"), "igls_vs_shy_usd": ("IGLS.L", "SHY in USD"),
              "proxy_vs_acwi": ("ACWI", "signal proxy (both in GBP)")}
    for key, (a, b) in labels.items():
        c = checks.get(key)
        if not c:
            continue
        lines.append(f"- {a} vs {b}, {_month(c['from'])} – {_month(c['to'])}: monthly correlation {c['corr']:.2f}, "
                     f"CAGR {_pct(c['cagr_a'])} vs {_pct(c['cagr_b'])}, volatility {_pct(c['vol_a'])} vs {_pct(c['vol_b'])}.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# JSON + CLI
# ----------------------------------------------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Series):
        return {pd.Timestamp(k).strftime("%Y-%m-%d"): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Timestamp, date)):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) or math.isinf(float(obj)) else round(float(obj), 8)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def results_to_json(results: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in results.items()}
    payload["phase3_verdict"] = dict(zip(("switch_to_momentum", "reason"), phase3_verdict(results)))
    return json.dumps(_jsonable(payload), indent=1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 backtest: rules-based strategies vs holding VWRP (GBP, after costs).")
    parser.add_argument("--out", help="write the markdown report to this path")
    parser.add_argument("--json", help="write the full results (metrics and monthly equity curves) as JSON")
    args = parser.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr)

    log("Downloading daily prices from Yahoo Finance...")
    closes, failures = download_daily_closes(all_tickers(), log=log)
    missing = [t for t in REQUIRED_TICKERS if t not in closes]
    if missing:
        log(f"ERROR: could not download required series after retries: {', '.join(missing)}")
        for t in missing:
            log(f"  {t}: {failures.get(t)}")
        return 2
    for t, err in failures.items():
        log(f"  optional series unavailable: {t} ({err})")
    results = run_all(closes)
    report = render_markdown(results)
    met = results["metrics"]
    print(f"Window {results['meta']['start'].date()} -> {results['meta']['end'].date()} ({results['meta']['months']} months)")
    for code in [c for c in STRATEGY_ORDER if c in met]:
        m = met[code]["Full period"]
        print(f"  {code:8s} CAGR {m['cagr'] * 100:6.2f}%  maxDD {m['max_dd'] * 100:6.1f}%  Sortino {m['sortino']:.2f}  "
              f"worst12m {m['worst_12m'] * 100:6.1f}%  gap {m['cagr_gap'] * 100:+.2f}pp")
    ok, reason = phase3_verdict(results)
    print(f"Phase 3 momentum switch: {'YES' if ok else 'NO'} ({reason})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Report written to {args.out}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(results_to_json(results))
        print(f"JSON written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

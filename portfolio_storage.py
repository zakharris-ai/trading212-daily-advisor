"""Structured data storage and history tracking for Trading 212 portfolio system.

Manages persistent daily snapshots, trade execution logs, position entry dates,
and continuous equity curve history.
"""

import json
import logging
import math
import re
from datetime import datetime, date, timedelta, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import LONDON_TZ, MODEL_VERSION
from t212_client import resolve_canonical_ticker

logger = logging.getLogger("storage")

DATA_DIR = Path(__file__).parent / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
HISTORY_FILE = DATA_DIR / "portfolio_history.json"
TRADE_LOG_FILE = DATA_DIR / "trade_log.json"
ENTRIES_FILE = DATA_DIR / "position_entries.json"
TRIMS_FILE = DATA_DIR / "position_trims.json"
CLOSED_FILE = DATA_DIR / "position_closed.json"
LEDGER_FILE = DATA_DIR / "recommendation_ledger.json"

# Benchmark for deposit-adjusted performance tracking ("everything in VWRP").
BENCHMARK_TICKER = "VWRP.L"
# Cash movements that are external flows (money in/out), not investment returns.
EXTERNAL_FLOW_TYPES = ("DEPOSIT", "WITHDRAW")
# Share-quantity tolerance when comparing holdings (fractional shares).
_QTY_EPS = 1e-4

_ISO_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?)(\.\d+)?)?\s*(Z|[+-]\d{2}:?\d{2})?$",
    re.IGNORECASE,
)


def init_storage() -> None:
    """Ensures data storage directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_json_file(file_path: Path, default: Any) -> Any:
    """Safely loads a JSON file or returns default."""
    if not file_path.is_file():
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read %s: %s", file_path, e)
        return default


def _save_json_file(file_path: Path, data: Any) -> None:
    """Safely writes JSON data to file with pretty formatting."""
    init_storage()
    tmp_path = file_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        tmp_path.replace(file_path)
    except Exception as e:
        logger.error("Failed to save %s: %s", file_path, e)
        if tmp_path.exists():
            tmp_path.unlink()


def _save_json_if_changed(file_path: Path, data: Any) -> bool:
    """Writes `data` only when it differs from what is on disk. Returns True if a write happened."""
    if file_path.is_file() and _load_json_file(file_path, None) == data:
        return False
    _save_json_file(file_path, data)
    return True


def _today_str() -> str:
    """Today's calendar date in Europe/London (YYYY-MM-DD)."""
    return datetime.now(LONDON_TZ).date().isoformat()


def _utc_now() -> datetime:
    """Current time as an aware UTC datetime (isolated so tests can pin it)."""
    return datetime.now(timezone.utc)


def _canon(ticker: Any) -> str:
    """Canonical Yahoo-style key for a ticker, so ticker-format changes never create phantom trades."""
    raw = str(ticker or "").strip()
    return resolve_canonical_ticker(raw) if raw else ""


def _to_float(val: Any, default: float = 0.0) -> float:
    """Float conversion mapping None, NaN, infinities and garbage to `default`."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(f) or math.isinf(f)) else f


def _canon_date_map(raw_map: Any) -> Dict[str, str]:
    """Canonicalises {ticker: 'YYYY-MM-DD'} keys; when two keys collide the latest date wins."""
    result: Dict[str, str] = {}
    if not isinstance(raw_map, dict):
        return result
    for ticker, d in raw_map.items():
        key = _canon(ticker)
        if not key or not d:
            continue
        d = str(d)
        if key not in result or d > result[key]:
            result[key] = d
    return result


def _parse_ts(val: Any) -> Optional[datetime]:
    """Parses an ISO-8601 timestamp/date into an aware UTC datetime (naive values are treated as UTC)."""
    if not val:
        return None
    m = _ISO_TS_RE.match(str(val).strip())
    if not m:
        return None
    day, clock, frac, tz = m.groups()
    text = f"{day}T{clock or '00:00:00'}"
    if clock and len(clock) == 5:
        text += ":00"
    if frac:
        text += "." + (frac[1:] + "000000")[:6]
    if tz:
        text += "+00:00" if tz.upper() == "Z" else (tz if ":" in tz else f"{tz[:3]}:{tz[3:]}")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_position_entries() -> Dict[str, Dict[str, str]]:
    """Returns recorded entry dates per account: {account_name: {ticker: 'YYYY-MM-DD'}}.

    Read-only: ticker keys are canonicalised in memory (e.g. a stale 'ASMLa_EQ' key resolves to
    'ASML.AS'); nothing is written to disk.
    """
    entries = _load_json_file(ENTRIES_FILE, {})
    if not isinstance(entries, dict):
        return {}
    return {acc: _canon_date_map(m) for acc, m in entries.items() if isinstance(m, dict)}


def update_position_entry(account_name: str, ticker: str, entry_date_str: str) -> None:
    """Updates the recorded entry date for an account holding."""
    entries = get_position_entries()
    if account_name not in entries:
        entries[account_name] = {}
    entries[account_name][_canon(ticker) or ticker] = entry_date_str
    _save_json_file(ENTRIES_FILE, entries)


def get_position_trims() -> Dict[str, Dict[str, str]]:
    """Returns recorded trim dates per account: {account_name: {ticker: 'YYYY-MM-DD'}}.

    Read-only: the trims file is merged in memory with TRIM events from the trade log; nothing is
    written to disk.
    """
    raw_trims = _load_json_file(TRIMS_FILE, {})
    trims: Dict[str, Dict[str, str]] = {}
    if isinstance(raw_trims, dict):
        trims = {acc: _canon_date_map(m) for acc, m in raw_trims.items() if isinstance(m, dict)}
    # Backfill from trade_log if trims file is missing entries
    trade_log = _load_json_file(TRADE_LOG_FILE, [])
    for t in trade_log if isinstance(trade_log, list) else []:
        if not isinstance(t, dict):
            continue
        acc = t.get("account")
        ticker = _canon(t.get("ticker"))
        action = str(t.get("action", ""))
        t_date = t.get("date")
        if acc and ticker and "TRIM" in action and t_date:
            if acc not in trims:
                trims[acc] = {}
            if ticker not in trims[acc] or str(t_date) > trims[acc][ticker]:
                trims[acc][ticker] = str(t_date)
    return trims


def update_position_trim(account_name: str, ticker: str, trim_date_str: str) -> None:
    """Updates the recorded trim date for an account holding."""
    trims = get_position_trims()
    if account_name not in trims:
        trims[account_name] = {}
    trims[account_name][_canon(ticker) or ticker] = trim_date_str
    _save_json_file(TRIMS_FILE, trims)


def get_closed_positions() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Returns the closed-position archive (read-only).

    Shape: {account_name: {ticker: {"closed_date": 'YYYY-MM-DD', "entry_date": 'YYYY-MM-DD'|None,
    "shares_sold": float}}}, holding the most recent full exit per ticker.
    """
    raw = _load_json_file(CLOSED_FILE, {})
    closed: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not isinstance(raw, dict):
        return closed
    for acc, acc_map in raw.items():
        if not isinstance(acc_map, dict):
            continue
        result: Dict[str, Dict[str, Any]] = {}
        for ticker, rec in acc_map.items():
            key = _canon(ticker)
            if not key or not isinstance(rec, dict):
                continue
            if key not in result or str(rec.get("closed_date", "")) > str(result[key].get("closed_date", "")):
                result[key] = dict(rec)
        closed[acc] = result
    return closed


def load_latest_snapshot(before_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Loads the most recent daily snapshot.
    
    Args:
        before_date: If provided (YYYY-MM-DD), finds the latest snapshot strictly prior to this date.

    Read-only: never creates directories or files.
    """
    if not SNAPSHOTS_DIR.is_dir():
        return None
    snapshot_files = sorted(SNAPSHOTS_DIR.glob("*.json"))
    if not snapshot_files:
        return None

    valid_files = []
    for sf in snapshot_files:
        stem = sf.stem  # e.g. "2026-09-09"
        if before_date and stem >= before_date:
            continue
        valid_files.append(sf)

    if not valid_files:
        return None

    latest_file = valid_files[-1]
    return _load_json_file(latest_file, None)


def load_portfolio_history() -> List[Dict[str, Any]]:
    """Loads the chronological portfolio history time series."""
    return _load_json_file(HISTORY_FILE, [])


def _position_qty(pos: Optional[Dict[str, Any]]) -> float:
    """Share quantity of a position dict (0.0 when missing or malformed)."""
    return _to_float((pos or {}).get("quantity"), 0.0)


def _initial_fill_date(pos: Optional[Dict[str, Any]]) -> Optional[str]:
    """Date part (YYYY-MM-DD) of a Trading 212 position's initialFillDate, if present and well-formed."""
    raw = str((pos or {}).get("initialFillDate") or "").strip()[:10]
    if not raw:
        return None
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError:
        return None


def _fill_based_entry_date(pos: Optional[Dict[str, Any]], current_date_str: str, last_close: Optional[str]) -> str:
    """Entry date for a position with no recorded entry: its initialFillDate (date part) when that is not in
    the future and not older than the ticker's last recorded close (a fill from before a full exit belongs to
    the old position); otherwise the run date."""
    fill_date = _initial_fill_date(pos)
    if fill_date and fill_date <= current_date_str and (not last_close or fill_date >= str(last_close)):
        return fill_date
    return current_date_str


def _trade_key(account: Any, ticker: Any, date_str: Any, action: Any) -> str:
    """Stable global dedupe key for a trade event: account|ticker|date|action."""
    return f"{account}|{_canon(ticker)}|{date_str}|{action}"


def detect_and_record_trades(
    accounts_data: Dict[str, Dict[str, Any]],
    current_date_str: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Detects trade executions (buys/sells) by comparing current holdings against recorded entries/prior snapshot.
    
    Automatically records newly acquired or increased positions into position_entries.json
    so anti-churn protections remain active.

    Integrity rules:
    - Iterates the UNION of prior-snapshot and current tickers, so full exits are logged as
      'SELL (CLOSED)' and the ticker moves from position_entries.json into the closed archive
      (position_closed.json) together with its close date.
    - Dedupes globally on account|ticker|date|action across the whole trade log (idempotent re-runs).
    - Tickers are canonicalised with resolve_canonical_ticker, so format changes never create phantom trades.
    - A NEW ENTRY is dated from the position's initialFillDate (date part) when available and not older
      than the ticker's last recorded close; otherwise from the run date.
    - Accounts absent from `accounts_data` are left untouched (never treated as closed).
    """
    if current_date_str is None:
        current_date_str = _today_str()

    prior_snapshot = load_latest_snapshot(before_date=current_date_str)
    entries = get_position_entries()
    trims = get_position_trims()
    closed = get_closed_positions()
    trade_log = _load_json_file(TRADE_LOG_FILE, [])
    if not isinstance(trade_log, list):
        trade_log = []
    logged_keys = {
        _trade_key(t.get("account"), t.get("ticker"), t.get("date"), t.get("action"))
        for t in trade_log
        if isinstance(t, dict)
    }
    new_trades: List[Dict[str, Any]] = []

    prior_holdings: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if prior_snapshot and "accounts" in prior_snapshot:
        for acc_name, acc_data in prior_snapshot["accounts"].items():
            holdings: Dict[str, Dict[str, Any]] = {}
            for pos in (acc_data or {}).get("positions", []) or []:
                key = _canon(pos.get("ticker"))
                if key and key not in holdings:
                    holdings[key] = pos
            prior_holdings[acc_name] = holdings

    def _record(event: Dict[str, Any]) -> bool:
        """Appends a trade event unless its account|ticker|date|action key is already logged."""
        key = _trade_key(event["account"], event["ticker"], event["date"], event["action"])
        if key in logged_keys:
            return False
        logged_keys.add(key)
        new_trades.append(event)
        trade_log.append(event)
        return True

    for acc_name, acc_info in accounts_data.items():
        current_positions = acc_info.get("raw_positions")
        if not current_positions:
            current_positions = acc_info.get("analyses") or current_positions
        if current_positions is None:
            # Fail closed: missing position data (API failure) must never be read as "everything sold".
            logger.warning("No position data for %s; skipping trade detection for this account.", acc_name)
            continue

        acc_entries = entries.setdefault(acc_name, {})
        acc_trims = trims.setdefault(acc_name, {})
        acc_closed = closed.setdefault(acc_name, {})

        current: Dict[str, Dict[str, Any]] = {}
        for pos in current_positions:
            key = _canon(pos.get("yf_ticker") or pos.get("ticker"))
            if not key:
                continue
            if key in current:
                logger.warning("Duplicate position for %s in %s; keeping the first.", key, acc_name)
                continue
            current[key] = pos
        prior_acc_holdings = prior_holdings.get(acc_name, {})

        # Union of current and prior tickers (current order first, then names that disappeared).
        tickers = list(current) + [t for t in prior_acc_holdings if t not in current]
        for ticker in tickers:
            pos = current.get(ticker)
            prior_pos = prior_acc_holdings.get(ticker)
            curr_qty = _position_qty(pos)
            prior_qty = _position_qty(prior_pos)
            ref_pos = pos or prior_pos or {}
            pos_price = _to_float(ref_pos.get("currentPrice") or ref_pos.get("price"), 0.0)
            pos_name = ref_pos.get("name") or ticker
            pos_currency = ref_pos.get("currency", "USD")
            timestamp = datetime.now().isoformat()

            if curr_qty > _QTY_EPS and prior_qty <= _QTY_EPS:
                # Newly acquired (or first-ever seen) position: date it from the broker's initial fill.
                last_close = (acc_closed.get(ticker) or {}).get("closed_date")
                entry_date = _fill_based_entry_date(pos, current_date_str, last_close)
                existing = acc_entries.get(ticker)
                if existing and not (last_close and str(existing) < str(last_close)):
                    acc_entries[ticker] = max(str(existing), entry_date)
                else:
                    acc_entries[ticker] = entry_date

                if _record({
                    "date": entry_date,
                    "timestamp": timestamp,
                    "account": acc_name,
                    "ticker": ticker,
                    "name": pos_name,
                    "action": "BUY (NEW ENTRY)",
                    "shares_added": curr_qty,
                    "new_total_shares": curr_qty,
                    "price": pos_price,
                    "currency": pos_currency,
                }):
                    logger.info("Recorded new entry for %s (%s): %.2f shares", ticker, acc_name, curr_qty)

            elif curr_qty > prior_qty + _QTY_EPS:
                # Increased existing position
                shares_added = curr_qty - prior_qty
                acc_entries[ticker] = current_date_str
                if _record({
                    "date": current_date_str,
                    "timestamp": timestamp,
                    "account": acc_name,
                    "ticker": ticker,
                    "name": pos_name,
                    "action": "BUY (ACCUMULATION)",
                    "shares_added": shares_added,
                    "new_total_shares": curr_qty,
                    "price": pos_price,
                    "currency": pos_currency,
                }):
                    logger.info("Recorded position addition for %s (%s): +%.2f shares", ticker, acc_name, shares_added)

            elif prior_qty > _QTY_EPS and curr_qty <= _QTY_EPS:
                # Fully exited since the prior snapshot: archive the entry and log the close.
                prev_entry = acc_entries.pop(ticker, None)
                existing_close = acc_closed.get(ticker)
                if not (existing_close and existing_close.get("closed_date") == current_date_str):
                    acc_closed[ticker] = {
                        "closed_date": current_date_str,
                        "entry_date": prev_entry,
                        "shares_sold": prior_qty,
                    }
                event = {
                    "date": current_date_str,
                    "timestamp": timestamp,
                    "account": acc_name,
                    "ticker": ticker,
                    "name": pos_name,
                    "action": "SELL (CLOSED)",
                    "shares_sold": prior_qty,
                    "remaining_shares": 0.0,
                    "price": pos_price,
                    "currency": pos_currency,
                }
                if pos is None:
                    event["price_source"] = "prior_snapshot"
                if _record(event):
                    logger.info("Recorded position close for %s (%s): -%.2f shares", ticker, acc_name, prior_qty)

            elif curr_qty < prior_qty - _QTY_EPS:
                # Reduced position (trimmed, still held)
                shares_trimmed = prior_qty - curr_qty
                acc_trims[ticker] = current_date_str
                if _record({
                    "date": current_date_str,
                    "timestamp": timestamp,
                    "account": acc_name,
                    "ticker": ticker,
                    "name": pos_name,
                    "action": "SELL (TRIMMED)",
                    "shares_sold": shares_trimmed,
                    "remaining_shares": curr_qty,
                    "price": pos_price,
                    "currency": pos_currency,
                }):
                    logger.info("Recorded position trim for %s (%s): -%.2f shares", ticker, acc_name, shares_trimmed)

            elif curr_qty > _QTY_EPS:
                # Existing position with unchanged quantity: preserve entry date or set default
                # (the broker's initial fill date when known, so a missing record never re-arms the new-entry hold)
                if ticker not in acc_entries:
                    last_close = (acc_closed.get(ticker) or {}).get("closed_date")
                    acc_entries[ticker] = _fill_based_entry_date(pos, current_date_str, last_close)

    if new_trades:
        _save_json_file(TRADE_LOG_FILE, trade_log)
    _save_json_if_changed(ENTRIES_FILE, entries)
    _save_json_if_changed(TRIMS_FILE, trims)
    closed_out = {acc: m for acc, m in closed.items() if m}
    if closed_out or CLOSED_FILE.is_file():
        _save_json_if_changed(CLOSED_FILE, closed_out)

    return new_trades


def compute_external_flows(
    accounts_data: Dict[str, Dict[str, Any]],
    current_date_str: Optional[str] = None,
    gbp_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Net external cash flows per account since the previous snapshot (read-only).

    DEPOSIT counts as +amount and WITHDRAW as -amount (GBP), taken from cash_data['recent_transactions']
    and deduped by 'reference'. Only transactions after the previous snapshot's timestamp, and not already
    recorded in it, are counted, so every flow is attributed to exactly one day. On the first tracked day
    (no previous snapshot) nothing is attributed.

    Returns:
        {"by_account": {acc: net_gbp}, "items": {acc: [flow, ...]}, "total_gbp": float,
         "prior_date": 'YYYY-MM-DD'|None, "complete": bool}
    """
    if current_date_str is None:
        current_date_str = _today_str()
    prior = load_latest_snapshot(before_date=current_date_str)
    prior_ts: Optional[datetime] = None
    prior_refs: Dict[str, set] = {}
    if prior:
        prior_ts = _parse_ts(prior.get("timestamp_utc")) or _parse_ts(prior.get("timestamp"))
        if prior_ts is None:
            prior_day = _parse_ts(prior.get("date"))
            # No usable timestamp: only attribute transactions dated after the prior snapshot's day.
            prior_ts = (prior_day + timedelta(days=1) - timedelta(microseconds=1)) if prior_day else None
        for acc, acc_data in (prior.get("accounts") or {}).items():
            prior_refs[acc] = {
                str(f.get("reference"))
                for f in ((acc_data or {}).get("external_flows") or [])
                if isinstance(f, dict) and f.get("reference")
            }

    by_account: Dict[str, float] = {}
    items: Dict[str, List[Dict[str, Any]]] = {}
    complete = True
    for acc, acc_info in accounts_data.items():
        cash = acc_info.get("cash") or {}
        if cash.get("recent_transactions_ok") is False:
            complete = False
        net = 0.0
        acc_items: List[Dict[str, Any]] = []
        if prior is not None:
            seen = set(prior_refs.get(acc, set()))
            currency = str(cash.get("currency") or "GBP").upper()
            for tx in cash.get("recent_transactions") or []:
                if not isinstance(tx, dict):
                    continue
                tx_type = str(tx.get("type") or "").upper()
                if tx_type not in EXTERNAL_FLOW_TYPES:
                    continue
                ts = _parse_ts(tx.get("dateTime"))
                if ts is None or (prior_ts is not None and ts <= prior_ts):
                    continue
                ref = str(tx.get("reference") or "").strip() or f"{tx_type}|{tx.get('dateTime')}|{tx.get('amount')}"
                if ref in seen:
                    continue
                seen.add(ref)
                amount = abs(_to_float(tx.get("amount"), 0.0))
                if currency == "USD" and gbp_rate:
                    amount = amount / float(gbp_rate)
                signed = amount if tx_type == "DEPOSIT" else -amount
                net += signed
                acc_items.append({
                    "reference": ref,
                    "type": tx_type,
                    "amount_gbp": round(signed, 2),
                    "dateTime": tx.get("dateTime"),
                })
        by_account[acc] = round(net, 2)
        items[acc] = acc_items

    if not complete:
        logger.warning("Recent cash transactions unavailable for at least one account; external flows may be understated.")
    return {
        "by_account": by_account,
        "items": items,
        "total_gbp": round(sum(by_account.values()), 2),
        "prior_date": prior.get("date") if prior else None,
        "complete": complete,
    }


def compute_twr_factor(prev_value: Any, curr_value: Any, net_flow: Any = 0.0) -> Optional[float]:
    """Daily time-weighted return factor V_t / (V_{t-1} + net_flow_t); None when undefined.

    The period's external flow is treated as arriving right after the previous valuation, so a
    deposit neither inflates nor dilutes the measured return.
    """
    denom = _to_float(prev_value) + _to_float(net_flow)
    curr = _to_float(curr_value, float("nan"))
    if denom <= 0 or math.isnan(curr):
        return None
    return curr / denom


def _build_performance(
    prior_snapshot: Optional[Dict[str, Any]],
    account_values: Dict[str, float],
    flows_by_account: Dict[str, float],
    snapshot_date: str,
) -> Dict[str, Any]:
    """Chains time-weighted return indices (base 100 on the first tracked day) from the prior snapshot."""
    prior_perf = (prior_snapshot or {}).get("performance") or {}
    prior_accounts = (prior_snapshot or {}).get("accounts") or {}
    has_prior_perf = isinstance(prior_perf, dict) and prior_perf.get("twr_index_total") is not None
    if not has_prior_perf:
        prior_perf = {}

    perf: Dict[str, Any] = {
        "twr_base_date": prior_perf.get("twr_base_date") or snapshot_date,
        "external_flows_gbp": round(sum(_to_float(flows_by_account.get(a)) for a in account_values), 2),
        "external_flows_by_account_gbp": {a: round(_to_float(flows_by_account.get(a)), 2) for a in account_values},
    }

    num = den = 0.0
    for acc, curr_val in account_values.items():
        key = f"twr_index_{acc.lower()}"
        prev_idx = prior_perf.get(key)
        prev_val = _to_float((prior_accounts.get(acc) or {}).get("total_value_gbp"))
        flow = _to_float(flows_by_account.get(acc))
        factor = compute_twr_factor(prev_val, curr_val, flow) if prev_val > 0 else None
        if prev_idx is None:
            idx = 100.0  # first tracked day for this account
        elif factor is None:
            idx = _to_float(prev_idx, 100.0)  # undefined factor: carry the index forward
        else:
            idx = _to_float(prev_idx, 100.0) * factor
        perf[key] = round(idx, 6)
        if factor is not None:
            num += _to_float(curr_val)
            den += prev_val + flow

    prev_total = prior_perf.get("twr_index_total")
    if prev_total is None:
        total_idx = 100.0
    elif den > 0:
        total_idx = _to_float(prev_total, 100.0) * (num / den)
    else:
        total_idx = _to_float(prev_total, 100.0)
    perf["twr_index_total"] = round(total_idx, 6)
    return perf


def _fetch_benchmark_closes(base_date_str: str, as_of_date_str: str) -> Tuple[Optional[float], Optional[float]]:
    """Returns (close on/before the base date, latest close on/before as-of date) for VWRP.L.

    Uses yfinance inside a try; returns (None, None) when data is unavailable.
    """
    try:
        import yfinance as yf

        base_d = date.fromisoformat(base_date_str)
        as_of_d = date.fromisoformat(as_of_date_str)
        hist = yf.Ticker(BENCHMARK_TICKER).history(
            start=(base_d - timedelta(days=10)).isoformat(),
            end=(as_of_d + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
        )
        if hist is None or hist.empty or "Close" not in hist:
            return None, None
        closes = [
            (ts.date(), float(val))
            for ts, val in hist["Close"].dropna().items()
            if ts.date() <= as_of_d
        ]
        if not closes:
            return None, None
        base_candidates = [val for d, val in closes if d <= base_d]
        base_close = base_candidates[-1] if base_candidates else None
        return base_close, closes[-1][1]
    except Exception as e:
        logger.warning("Benchmark %s close unavailable: %s", BENCHMARK_TICKER, e)
        return None, None


_MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _positive_index(val: Any) -> Optional[float]:
    """A finite, strictly positive index level; None for missing, zero, negative, NaN, inf, bools or garbage."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0:
        return None
    return f


def _history_date(val: Any) -> Optional[date]:
    """Calendar date of a history record ('YYYY-MM-DD', time part ignored); None when missing or malformed."""
    try:
        return date.fromisoformat(str(val or "").strip()[:10])
    except ValueError:
        return None


def _pretty_date(d: date) -> str:
    """'19 Oct 2026' (locale-independent)."""
    return f"{d.day} {_MONTH_ABBR[d.month - 1]} {d.year}"


def _signed_pct_text(val: float) -> str:
    """'+3.2%' / '-1.5%'; a value that rounds to zero reads '+0.0%', never '-0.0%'."""
    rounded = round(val, 1)
    return f"{(rounded if rounded != 0 else 0.0):+.1f}%"


def summarize_tracking_performance(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Since-tracking-began comparison of the portfolio's time-weighted return against the VWRP.L benchmark.

    Pure function: no file or network access (pass it load_portfolio_history()). Only history records with a
    valid date AND a finite, positive `twr_index_total` AND `benchmark_vwrp_index` are used; zero, negative,
    NaN, boolean and non-numeric values are skipped, and when several usable records share a date the last
    one wins. Both returns are latest / first - 1 (in %) over the same window, so they compare like for like
    even when the benchmark was unavailable on the first tracked day.

    Returns {"available", "start_date", "end_date" (ISO dates), "portfolio_return_pct", "benchmark_return_pct",
    "difference_pp", "days", "text"}. The numbers stay None (available False) until two usable daily
    snapshots exist; "text" is always a plain-English sentence for the email, dashboard and console.
    """
    by_date: Dict[date, Tuple[float, float]] = {}
    for rec in history if isinstance(history, (list, tuple)) else []:
        if not isinstance(rec, dict):
            continue
        rec_date = _history_date(rec.get("date"))
        twr = _positive_index(rec.get("twr_index_total"))
        bench = _positive_index(rec.get("benchmark_vwrp_index"))
        if rec_date is None or twr is None or bench is None:
            continue
        by_date[rec_date] = (twr, bench)
    points = sorted(by_date.items())

    summary: Dict[str, Any] = {
        "available": False,
        "start_date": None,
        "end_date": None,
        "portfolio_return_pct": None,
        "benchmark_return_pct": None,
        "difference_pp": None,
        "days": None,
        "text": "Performance tracking vs VWRP starts with the next daily snapshot.",
    }
    if not points:
        return summary

    start_d, (twr_first, bench_first) = points[0]
    end_d, (twr_last, bench_last) = points[-1]
    summary["start_date"] = start_d.isoformat()
    summary["end_date"] = end_d.isoformat()
    started_text = (
        f"Performance tracking vs VWRP started on {_pretty_date(start_d)}; "
        "the comparison appears after the second daily snapshot."
    )
    if len(points) < 2:
        summary["text"] = started_text
        return summary

    portfolio_pct = (twr_last / twr_first - 1.0) * 100.0
    benchmark_pct = (bench_last / bench_first - 1.0) * 100.0
    if not (math.isfinite(portfolio_pct) and math.isfinite(benchmark_pct)):
        summary["text"] = started_text
        return summary

    summary.update({
        "available": True,
        "portfolio_return_pct": round(portfolio_pct, 2),
        "benchmark_return_pct": round(benchmark_pct, 2),
        "difference_pp": round(portfolio_pct - benchmark_pct, 2),
        "days": (end_d - start_d).days,
        "text": (
            f"Since tracking began ({_pretty_date(start_d)}): your portfolio {_signed_pct_text(portfolio_pct)} "
            f"vs VWRP {_signed_pct_text(benchmark_pct)} (time-weighted, net of deposits and withdrawals)."
        ),
    })
    return summary


def compute_day_over_day_delta(
    current_combined_total: float,
    current_account_totals: Dict[str, float],
    current_date_str: Optional[str] = None,
    external_flows_by_account: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Computes day-over-day change (£ and %) against the prior daily snapshot.

    External cash flows (deposits minus withdrawals since the prior snapshot) are subtracted, so
    money added or removed is never reported as a gain or loss.
    """
    if current_date_str is None:
        current_date_str = _today_str()
    flows = external_flows_by_account or {}

    prior_snapshot = load_latest_snapshot(before_date=current_date_str)
    if not prior_snapshot:
        return {
            "has_prior": False,
            "prior_date": None,
            "combined_delta_gbp": 0.0,
            "combined_delta_pct": 0.0,
            "external_flows_gbp": 0.0,
            "accounts": {acc: {"delta_gbp": 0.0, "delta_pct": 0.0} for acc in current_account_totals},
        }

    total_flow = sum(_to_float(flows.get(acc)) for acc in current_account_totals)
    prior_combined = _to_float(prior_snapshot.get("combined", {}).get("total_value_gbp", 0.0))
    raw_delta = current_combined_total - prior_combined if prior_combined > 0 else 0.0
    delta_gbp = raw_delta - total_flow if prior_combined > 0 else 0.0
    combined_base = prior_combined + total_flow
    delta_pct = (delta_gbp / combined_base * 100.0) if prior_combined > 0 and combined_base > 0 else 0.0

    acc_deltas = {}
    prior_accounts = prior_snapshot.get("accounts", {})
    for acc, curr_val in current_account_totals.items():
        p_acc_val = _to_float(prior_accounts.get(acc, {}).get("total_value_gbp", 0.0))
        acc_flow = _to_float(flows.get(acc))
        p_delta = (curr_val - p_acc_val - acc_flow) if p_acc_val > 0 else 0.0
        acc_base = p_acc_val + acc_flow
        p_pct = (p_delta / acc_base * 100.0) if p_acc_val > 0 and acc_base > 0 else 0.0
        acc_deltas[acc] = {
            "delta_gbp": p_delta,
            "delta_pct": p_pct,
            "prior_val_gbp": p_acc_val,
            "external_flow_gbp": acc_flow,
        }

    return {
        "has_prior": True,
        "prior_date": prior_snapshot.get("date"),
        "prior_combined_gbp": prior_combined,
        "combined_delta_gbp": delta_gbp,
        "combined_delta_pct": delta_pct,
        "raw_combined_delta_gbp": raw_delta,
        "external_flows_gbp": total_flow,
        "accounts": acc_deltas,
    }


def save_daily_snapshot(
    accounts_data: Dict[str, Dict[str, Any]],
    consolidated_view: Dict[str, Any],
    macro_regime: Dict[str, Any],
    results_by_account: Dict[str, Any],
    gbp_rate: float,
    snapshot_date: Optional[str] = None,
    external_flows: Optional[Dict[str, Any]] = None,
) -> Path:
    """Saves a comprehensive daily snapshot of balances, positions, weights, and plans.

    Also records deposit-adjusted performance: external cash flows since the previous snapshot,
    chained time-weighted return indices (base 100) and the VWRP.L benchmark index.
    """
    init_storage()
    if snapshot_date is None:
        snapshot_date = _today_str()
    if external_flows is None:
        external_flows = compute_external_flows(accounts_data, current_date_str=snapshot_date, gbp_rate=gbp_rate)
    flows_by_account = external_flows.get("by_account", {}) or {}
    flow_items = external_flows.get("items", {}) or {}
    prior_snapshot = load_latest_snapshot(before_date=snapshot_date)

    snapshot_file = SNAPSHOTS_DIR / f"{snapshot_date}.json"

    # Build structured account entries
    accounts_output = {}
    for acc_name, acc_info in accounts_data.items():
        cash = acc_info.get("cash", {})
        analyses = acc_info.get("analyses", [])
        rebal = results_by_account.get(acc_name, {}).get("rebalancing_plan", {})

        pos_list = []
        for p in analyses:
            pos_list.append({
                "ticker": p.get("ticker"),
                "search_symbol": p.get("search_symbol"),
                "name": p.get("name"),
                "quantity": p.get("quantity", 0.0),
                "avg_price": p.get("avg_price", 0.0),
                "price": p.get("price", 0.0),
                "currency": p.get("currency", "USD"),
                "price_gbp": p.get("price_gbp", 0.0),
                "value_gbp": p.get("value_gbp", 0.0),
                "unrealized_pnl": p.get("unrealized_pnl", 0.0),
                "weight_pct": p.get("portfolio_weight_pct", 0.0),
                "signal": p.get("signal"),
                "action_short": p.get("action_short"),
                "conviction_score": p.get("conviction_score"),
                "trailing_stop": p.get("trailing_stop"),
                "target_price": p.get("target_price"),
            })

        acc_share = consolidated_view.get("account_shares", {}).get(acc_name, {})

        accounts_output[acc_name] = {
            "total_value_gbp": acc_share.get("total_value_gbp", cash.get("total", 0.0)),
            "invested_gbp": acc_share.get("invested_gbp", 0.0),
            "liquid_cash_gbp": cash.get("free_gbp", cash.get("free", 0.0)),
            "blocked_cash_gbp": cash.get("blocked_gbp", cash.get("blocked", 0.0)),
            "ppl_gbp": cash.get("ppl", 0.0),
            "share_of_combined_pct": acc_share.get("share_of_total_pct", 0.0),
            "external_flows_gbp": round(_to_float(flows_by_account.get(acc_name)), 2),
            "external_flows": flow_items.get(acc_name, []),
            "positions": pos_list,
            "rebalance_actions": {
                "sells": rebal.get("sell_orders", []),
                "buys": rebal.get("buy_orders", []),
                "pending_sells": rebal.get("pending_sell_orders", []),
                "pending_buys": rebal.get("pending_buys", {}),
                "status_badge": rebal.get("status_badge", ""),
                "data_degraded_reason": rebal.get("data_degraded_reason"),
            },
        }

    # Deposit-adjusted performance (TWR) and the "everything in VWRP" benchmark
    performance = _build_performance(
        prior_snapshot,
        {k: _to_float(v.get("total_value_gbp")) for k, v in accounts_output.items()},
        flows_by_account,
        snapshot_date,
    )
    performance["external_flows_complete"] = bool(external_flows.get("complete", True))
    base_close, latest_close = _fetch_benchmark_closes(performance["twr_base_date"], snapshot_date)
    performance["benchmark_ticker"] = BENCHMARK_TICKER
    performance["benchmark_vwrp_close"] = latest_close
    performance["benchmark_vwrp_base_close"] = base_close
    performance["benchmark_vwrp_index"] = (
        round(100.0 * latest_close / base_close, 6) if (base_close and latest_close) else None
    )

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "timestamp_utc": _utc_now().isoformat(),
        "date": snapshot_date,
        "model_version": MODEL_VERSION,
        "gbp_usd_rate": gbp_rate,
        "macro": {
            "risk_climate": macro_regime.get("risk_climate"),
            "macro_score": macro_regime.get("macro_score"),
            "vix": macro_regime.get("vix"),
            "term_structure": macro_regime.get("vix_term_structure"),
            "spy_dist_days": macro_regime.get("spy_dist_days"),
            "qqq_dist_days": macro_regime.get("qqq_dist_days"),
            "macro_data_ok": macro_regime.get("macro_data_ok", True),
            "macro_missing": macro_regime.get("macro_missing", []),
        },
        "combined": {
            "total_value_gbp": consolidated_view.get("total_net_worth_gbp", 0.0),
            "total_invested_gbp": consolidated_view.get("total_invested_gbp", 0.0),
            "total_liquid_cash_gbp": consolidated_view.get("total_liquid_cash_gbp", 0.0),
            "total_blocked_cash_gbp": consolidated_view.get("total_blocked_cash_gbp", 0.0),
            "total_ppl_gbp": consolidated_view.get("total_ppl_gbp", 0.0),
            "external_flows_gbp": performance["external_flows_gbp"],
            "thematic_breakdown": consolidated_view.get("thematic_breakdown", []),
            "total_unique_holdings": len(consolidated_view.get("holdings", {})),
        },
        "accounts": accounts_output,
        "performance": performance,
    }

    _save_json_file(snapshot_file, snapshot)
    logger.info("Saved daily portfolio snapshot to %s", snapshot_file)

    # Append to long-term history timeseries
    append_history_record(
        combined_total=snapshot["combined"]["total_value_gbp"],
        accounts_totals={k: v["total_value_gbp"] for k, v in accounts_output.items()},
        total_ppl=snapshot["combined"]["total_ppl_gbp"],
        snapshot_date=snapshot_date,
        performance=performance,
    )

    return snapshot_file


def append_history_record(
    combined_total: float,
    accounts_totals: Dict[str, float],
    total_ppl: float,
    snapshot_date: Optional[str] = None,
    performance: Optional[Dict[str, Any]] = None,
) -> None:
    """Maintains a chronological time series in data/portfolio_history.json.

    When `performance` is supplied the record also carries external_flows_gbp (net deposits minus
    withdrawals since the previous snapshot), external_flows_by_account_gbp, the time-weighted return
    indices (twr_index_total, twr_index_<account>) and benchmark_vwrp_index (None if unavailable).
    """
    init_storage()
    if snapshot_date is None:
        snapshot_date = _today_str()

    history = _load_json_file(HISTORY_FILE, [])

    record = {
        "date": snapshot_date,
        "combined_total": round(combined_total, 2),
        "accounts": {k: round(v, 2) for k, v in accounts_totals.items()},
        "total_ppl": round(total_ppl, 2),
        "updated_at": datetime.now().isoformat(),
    }
    if performance:
        record["external_flows_gbp"] = performance.get("external_flows_gbp", 0.0)
        record["external_flows_by_account_gbp"] = performance.get("external_flows_by_account_gbp", {})
        for key, value in performance.items():
            if key.startswith("twr_index_"):
                record[key] = value
        record["benchmark_vwrp_index"] = performance.get("benchmark_vwrp_index")

    # Upsert record for today
    found = False
    for i, r in enumerate(history):
        if r.get("date") == snapshot_date:
            history[i] = record
            found = True
            break
    if not found:
        history.append(record)

    # Keep sorted by date
    history.sort(key=lambda x: x.get("date", ""))
    _save_json_file(HISTORY_FILE, history)
    logger.info("Updated portfolio history record for %s", snapshot_date)


def append_recommendation_records(records: List[Dict[str, Any]]) -> int:
    """Appends recommendation records to data/recommendation_ledger.json; returns how many were added.

    Records are deduped on date|model_version|account|action|ticker so a forced same-day re-send does
    not double-count. An existing ledger that cannot be parsed is never overwritten.
    """
    if not records:
        return 0
    ledger: List[Dict[str, Any]] = []
    if LEDGER_FILE.is_file():
        try:
            with open(LEDGER_FILE, "r", encoding="utf-8") as f:
                ledger = json.load(f)
        except Exception as e:
            logger.error("Recommendation ledger %s is unreadable (%s); not overwriting it.", LEDGER_FILE, e)
            return 0
        if not isinstance(ledger, list):
            logger.error("Recommendation ledger %s is not a JSON list; not overwriting it.", LEDGER_FILE)
            return 0

    def _ledger_key(rec: Dict[str, Any]) -> str:
        return "|".join(str(rec.get(k, "")) for k in ("date", "model_version", "account", "action", "ticker"))

    seen = {_ledger_key(r) for r in ledger if isinstance(r, dict)}
    added = 0
    for rec in records:
        key = _ledger_key(rec)
        if key in seen:
            continue
        seen.add(key)
        ledger.append(rec)
        added += 1
    if added:
        _save_json_file(LEDGER_FILE, ledger)
    return added

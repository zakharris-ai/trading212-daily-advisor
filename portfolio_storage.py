"""Structured data storage and history tracking for Trading 212 portfolio system.

Manages persistent daily snapshots, trade execution logs, position entry dates,
and continuous equity curve history.
"""

import json
import logging
from datetime import datetime, date
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("storage")

DATA_DIR = Path(__file__).parent / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
HISTORY_FILE = DATA_DIR / "portfolio_history.json"
TRADE_LOG_FILE = DATA_DIR / "trade_log.json"
ENTRIES_FILE = DATA_DIR / "position_entries.json"
TRIMS_FILE = DATA_DIR / "position_trims.json"


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


def get_position_entries() -> Dict[str, Dict[str, str]]:
    """Returns recorded entry dates per account: {account_name: {ticker: 'YYYY-MM-DD'}}."""
    return _load_json_file(ENTRIES_FILE, {})


def update_position_entry(account_name: str, ticker: str, entry_date_str: str) -> None:
    """Updates the recorded entry date for an account holding."""
    entries = get_position_entries()
    if account_name not in entries:
        entries[account_name] = {}
    entries[account_name][ticker] = entry_date_str
    _save_json_file(ENTRIES_FILE, entries)


def get_position_trims() -> Dict[str, Dict[str, str]]:
    """Returns recorded trim dates per account: {account_name: {ticker: 'YYYY-MM-DD'}}."""
    trims = _load_json_file(TRIMS_FILE, {})
    # Backfill from trade_log if trims file is missing entries
    trade_log = _load_json_file(TRADE_LOG_FILE, [])
    updated = False
    for t in trade_log:
        acc = t.get("account")
        ticker = t.get("ticker")
        action = str(t.get("action", ""))
        t_date = t.get("date")
        if acc and ticker and "TRIM" in action and t_date:
            if acc not in trims:
                trims[acc] = {}
            if ticker not in trims[acc] or t_date > trims[acc][ticker]:
                trims[acc][ticker] = t_date
                updated = True
    if updated:
        _save_json_file(TRIMS_FILE, trims)
    return trims


def update_position_trim(account_name: str, ticker: str, trim_date_str: str) -> None:
    """Updates the recorded trim date for an account holding."""
    trims = get_position_trims()
    if account_name not in trims:
        trims[account_name] = {}
    trims[account_name][ticker] = trim_date_str
    _save_json_file(TRIMS_FILE, trims)


def load_latest_snapshot(before_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Loads the most recent daily snapshot.
    
    Args:
        before_date: If provided (YYYY-MM-DD), finds the latest snapshot strictly prior to this date.
    """
    init_storage()
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


def detect_and_record_trades(
    accounts_data: Dict[str, Dict[str, Any]],
    current_date_str: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Detects trade executions (buys/sells) by comparing current holdings against recorded entries/prior snapshot.
    
    Automatically records newly acquired or increased positions into position_entries.json
    so anti-churn protections remain active.
    """
    if current_date_str is None:
        current_date_str = date.today().isoformat()

    prior_snapshot = load_latest_snapshot(before_date=current_date_str)
    entries = get_position_entries()
    trims = get_position_trims()
    trade_log = _load_json_file(TRADE_LOG_FILE, [])
    new_trades = []

    prior_holdings = {}
    if prior_snapshot and "accounts" in prior_snapshot:
        for acc_name, acc_data in prior_snapshot["accounts"].items():
            prior_holdings[acc_name] = {
                pos["ticker"]: pos for pos in acc_data.get("positions", [])
            }

    for acc_name, acc_info in accounts_data.items():
        if acc_name not in entries:
            entries[acc_name] = {}
        if acc_name not in trims:
            trims[acc_name] = {}

        current_positions = acc_info.get("raw_positions", []) or acc_info.get("analyses", [])
        prior_acc_holdings = prior_holdings.get(acc_name, {})

        for pos in current_positions:
            ticker = pos.get("yf_ticker") or pos.get("ticker", "")
            curr_qty = float(pos.get("quantity", 0.0) or 0.0)
            prior_pos = prior_acc_holdings.get(ticker)
            prior_qty = float(prior_pos.get("quantity", 0.0)) if prior_pos else 0.0
            pos_price = float(pos.get("currentPrice") or pos.get("price", 0.0) or 0.0)
            pos_name = pos.get("name") or ticker

            # Helper to check if this exact trade state was already logged today
            def _already_logged(act_str: str, target_qty: float) -> bool:
                for existing in reversed(trade_log[-40:]):
                    if (
                        existing.get("date") == current_date_str
                        and existing.get("account") == acc_name
                        and existing.get("ticker") == ticker
                        and existing.get("action") == act_str
                    ):
                        logged_q = float(existing.get("new_total_shares") or existing.get("remaining_shares") or 0.0)
                        if abs(logged_q - target_qty) < 1e-4:
                            return True
                return False

            # If position didn't exist before, or quantity increased: it's a new buy/add
            if prior_pos is None and curr_qty > 0:
                # First time seeing this position or newly acquired
                if ticker not in entries[acc_name]:
                    entries[acc_name][ticker] = current_date_str
                
                if not _already_logged("BUY (NEW ENTRY)", curr_qty):
                    trade_event = {
                        "date": current_date_str,
                        "timestamp": datetime.now().isoformat(),
                        "account": acc_name,
                        "ticker": ticker,
                        "name": pos_name,
                        "action": "BUY (NEW ENTRY)",
                        "shares_added": curr_qty,
                        "new_total_shares": curr_qty,
                        "price": pos_price,
                        "currency": pos.get("currency", "USD"),
                    }
                    new_trades.append(trade_event)
                    trade_log.append(trade_event)
                    logger.info("Recorded new entry for %s (%s): %.2f shares", ticker, acc_name, curr_qty)

            elif curr_qty > prior_qty + 1e-4:
                # Increased existing position
                shares_added = curr_qty - prior_qty
                entries[acc_name][ticker] = current_date_str
                if not _already_logged("BUY (ACCUMULATION)", curr_qty):
                    trade_event = {
                        "date": current_date_str,
                        "timestamp": datetime.now().isoformat(),
                        "account": acc_name,
                        "ticker": ticker,
                        "name": pos_name,
                        "action": "BUY (ACCUMULATION)",
                        "shares_added": shares_added,
                        "new_total_shares": curr_qty,
                        "price": pos_price,
                        "currency": pos.get("currency", "USD"),
                    }
                    new_trades.append(trade_event)
                    trade_log.append(trade_event)
                    logger.info("Recorded position addition for %s (%s): +%.2f shares", ticker, acc_name, shares_added)

            elif curr_qty < prior_qty - 1e-4 and prior_qty > 0:
                # Reduced position (Trimmed or Closed)
                shares_trimmed = prior_qty - curr_qty
                if curr_qty > 0:
                    trims[acc_name][ticker] = current_date_str
                act_label = "SELL (TRIMMED)" if curr_qty > 0 else "SELL (CLOSED)"
                if not _already_logged(act_label, curr_qty):
                    trade_event = {
                        "date": current_date_str,
                        "timestamp": datetime.now().isoformat(),
                        "account": acc_name,
                        "ticker": ticker,
                        "name": pos_name,
                        "action": act_label,
                        "shares_sold": shares_trimmed,
                        "remaining_shares": curr_qty,
                        "price": pos_price,
                        "currency": pos.get("currency", "USD"),
                    }
                    new_trades.append(trade_event)
                    trade_log.append(trade_event)
                    logger.info("Recorded position trim/close for %s (%s): -%.2f shares", ticker, acc_name, shares_trimmed)

            else:
                # Existing position with unchanged quantity: preserve entry date or set default
                if ticker not in entries[acc_name]:
                    entries[acc_name][ticker] = current_date_str

    if new_trades:
        _save_json_file(TRADE_LOG_FILE, trade_log)
    _save_json_file(ENTRIES_FILE, entries)
    _save_json_file(TRIMS_FILE, trims)

    return new_trades


def compute_day_over_day_delta(
    current_combined_total: float,
    current_account_totals: Dict[str, float],
    current_date_str: Optional[str] = None
) -> Dict[str, Any]:
    """Computes day-over-day change (£ and %) against the prior daily snapshot."""
    if current_date_str is None:
        current_date_str = date.today().isoformat()

    prior_snapshot = load_latest_snapshot(before_date=current_date_str)
    if not prior_snapshot:
        return {
            "has_prior": False,
            "prior_date": None,
            "combined_delta_gbp": 0.0,
            "combined_delta_pct": 0.0,
            "accounts": {acc: {"delta_gbp": 0.0, "delta_pct": 0.0} for acc in current_account_totals},
        }

    prior_combined = prior_snapshot.get("combined", {}).get("total_value_gbp", 0.0)
    delta_gbp = current_combined_total - prior_combined if prior_combined > 0 else 0.0
    delta_pct = (delta_gbp / prior_combined * 100.0) if prior_combined > 0 else 0.0

    acc_deltas = {}
    prior_accounts = prior_snapshot.get("accounts", {})
    for acc, curr_val in current_account_totals.items():
        p_acc_val = prior_accounts.get(acc, {}).get("total_value_gbp", 0.0)
        p_delta = curr_val - p_acc_val if p_acc_val > 0 else 0.0
        p_pct = (p_delta / p_acc_val * 100.0) if p_acc_val > 0 else 0.0
        acc_deltas[acc] = {
            "delta_gbp": p_delta,
            "delta_pct": p_pct,
            "prior_val_gbp": p_acc_val,
        }

    return {
        "has_prior": True,
        "prior_date": prior_snapshot.get("date"),
        "prior_combined_gbp": prior_combined,
        "combined_delta_gbp": delta_gbp,
        "combined_delta_pct": delta_pct,
        "accounts": acc_deltas,
    }


def save_daily_snapshot(
    accounts_data: Dict[str, Dict[str, Any]],
    consolidated_view: Dict[str, Any],
    macro_regime: Dict[str, Any],
    results_by_account: Dict[str, Any],
    gbp_rate: float,
    snapshot_date: Optional[str] = None
) -> Path:
    """Saves a comprehensive daily snapshot of balances, positions, weights, and plans."""
    init_storage()
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

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
            "positions": pos_list,
            "rebalance_actions": {
                "sells": rebal.get("sell_orders", []),
                "buys": rebal.get("buy_orders", []),
                "pending_sells": rebal.get("pending_sell_orders", []),
                "pending_buys": rebal.get("pending_buys", {}),
                "status_badge": rebal.get("status_badge", ""),
            },
        }

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "date": snapshot_date,
        "gbp_usd_rate": gbp_rate,
        "macro": {
            "risk_climate": macro_regime.get("risk_climate"),
            "macro_score": macro_regime.get("macro_score"),
            "vix": macro_regime.get("vix"),
            "term_structure": macro_regime.get("vix_term_structure"),
            "spy_dist_days": macro_regime.get("spy_dist_days"),
            "qqq_dist_days": macro_regime.get("qqq_dist_days"),
        },
        "combined": {
            "total_value_gbp": consolidated_view.get("total_net_worth_gbp", 0.0),
            "total_invested_gbp": consolidated_view.get("total_invested_gbp", 0.0),
            "total_liquid_cash_gbp": consolidated_view.get("total_liquid_cash_gbp", 0.0),
            "total_blocked_cash_gbp": consolidated_view.get("total_blocked_cash_gbp", 0.0),
            "total_ppl_gbp": consolidated_view.get("total_ppl_gbp", 0.0),
            "thematic_breakdown": consolidated_view.get("thematic_breakdown", []),
            "total_unique_holdings": len(consolidated_view.get("holdings", {})),
        },
        "accounts": accounts_output,
    }

    _save_json_file(snapshot_file, snapshot)
    logger.info("Saved daily portfolio snapshot to %s", snapshot_file)

    # Append to long-term history timeseries
    append_history_record(
        combined_total=snapshot["combined"]["total_value_gbp"],
        accounts_totals={k: v["total_value_gbp"] for k, v in accounts_output.items()},
        total_ppl=snapshot["combined"]["total_ppl_gbp"],
        snapshot_date=snapshot_date
    )

    return snapshot_file


def append_history_record(
    combined_total: float,
    accounts_totals: Dict[str, float],
    total_ppl: float,
    snapshot_date: Optional[str] = None
) -> None:
    """Maintains a chronological time series in data/portfolio_history.json."""
    init_storage()
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    history = _load_json_file(HISTORY_FILE, [])

    record = {
        "date": snapshot_date,
        "combined_total": round(combined_total, 2),
        "accounts": {k: round(v, 2) for k, v in accounts_totals.items()},
        "total_ppl": round(total_ppl, 2),
        "updated_at": datetime.now().isoformat(),
    }

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

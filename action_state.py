"""Small persisted state shared by the daily digest and the intraday watcher (all files live in data/).

  * latest_actions.json: the sells/buys the latest real digest told you to place, with the share count each holding
    had at the time, so the watcher can tell whether they have been placed.
  * glide_plan.json: this month's glide-path trims as sized on the monthly buy day, so later digests can carry an
    unplaced trim forward instead of dropping it (the buy-day email often lands after the London close).
  * alert_state.json: which intraday alerts were already sent today (at most one email per trigger per day).

Advisory only: nothing here places or cancels an order.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def _path(name: str) -> Path:
    return DATA_DIR / name


def _load(name: str, default: Any) -> Any:
    p = _path(name)
    if not p.is_file():
        return default
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # corrupt or partial file: behave as if absent
        logger.warning("Could not read %s: %s", p, e)
        return default


def _save(name: str, data: Any) -> bool:
    p = _path(name)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        tmp.replace(p)
        return True
    except Exception as e:
        logger.error("Failed to save %s: %s", p, e)
        return False


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def _root(ticker: Any) -> str:
    t = str(ticker or "").upper().strip()
    return t.split(".")[0] if "." in t else t


def held_quantity(analyses: Optional[List[Dict[str, Any]]], ticker: str) -> float:
    """Quantity currently held of `ticker` (earmarked lines excluded), matching on the ticker root."""
    root = _root(ticker)
    return sum(
        max(0.0, _num(p.get("quantity"), 0.0)) for p in (analyses or [])
        if isinstance(p, dict) and not p.get("is_earmarked") and _root(p.get("ticker")) == root
    )


def _order_row(o: Dict[str, Any], analyses: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    return {
        "ticker": o.get("ticker", ""),
        "search_symbol": o.get("search_symbol") or _root(o.get("ticker")),
        "name": o.get("name") or o.get("ticker", ""),
        "shares": round(_num(o.get("shares"), 0.0), 6),
        "estimated_gbp": round(_num(o.get("estimated_gbp"), 0.0), 2),
        "order_reason": o.get("order_reason") or "",
        "action": o.get("action") or "",
        "qty_before": round(held_quantity(analyses, o.get("ticker", "")), 6),
    }


# --- latest_actions.json --------------------------------------------------------------------------------------
def build_latest_actions(results_by_account: Dict[str, Dict[str, Any]], run_date: str) -> Dict[str, Any]:
    """Today's actionable (not already queued) sells and buys per account, with pre-trade share counts."""
    accounts: Dict[str, Any] = {}
    for acc_name, acc in (results_by_account or {}).items():
        if not isinstance(acc, dict):
            continue
        plan = acc.get("rebalancing_plan") if isinstance(acc.get("rebalancing_plan"), dict) else {}
        analyses = acc.get("analyses") or []
        sells = [_order_row(o, analyses) for o in plan.get("sell_orders") or [] if isinstance(o, dict) and not o.get("is_pending")]
        buys = [_order_row(o, analyses) for o in plan.get("buy_orders") or [] if isinstance(o, dict) and not o.get("is_pending")]
        accounts[acc_name] = {"sells": sells, "buys": buys}
    return {"date": run_date, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "accounts": accounts}


def save_latest_actions(results_by_account: Dict[str, Dict[str, Any]], run_date: str) -> bool:
    return _save("latest_actions.json", build_latest_actions(results_by_account, run_date))


def load_latest_actions() -> Dict[str, Any]:
    d = _load("latest_actions.json", {})
    return d if isinstance(d, dict) else {}


# --- glide_plan.json ------------------------------------------------------------------------------------------
def build_glide_plan(results_by_account: Dict[str, Dict[str, Any]], run_date: str) -> Optional[Dict[str, Any]]:
    """This month's buy-day glide-path trims (proposals, whether or not the policy made them orders); None if none."""
    accounts: Dict[str, List[Dict[str, Any]]] = {}
    for acc_name, acc in (results_by_account or {}).items():
        plan = (acc or {}).get("rebalancing_plan") if isinstance(acc, dict) else None
        glide = (plan or {}).get("rebalance_glide") if isinstance(plan, dict) else None
        if not isinstance(glide, dict) or glide.get("trigger") != "glide_path" or not glide.get("is_buy_day"):
            continue
        rows = [_order_row(o, acc.get("analyses") or []) for o in glide.get("proposals") or [] if isinstance(o, dict)]
        if rows:
            accounts[acc_name] = rows
    if not accounts:
        return None
    return {"month": str(run_date)[:7], "buy_day": str(run_date), "accounts": accounts}


def save_glide_plan(results_by_account: Dict[str, Dict[str, Any]], run_date: str) -> bool:
    plan = build_glide_plan(results_by_account, run_date)
    return _save("glide_plan.json", plan) if plan else False


def load_glide_plan() -> Dict[str, Any]:
    d = _load("glide_plan.json", {})
    return d if isinstance(d, dict) else {}


# --- alert_state.json -----------------------------------------------------------------------------------------
def load_alert_state(today: date) -> Dict[str, Any]:
    """{'date': 'YYYY-MM-DD', 'sent': [...keys]} for today (a new day starts empty)."""
    d = _load("alert_state.json", {})
    if not isinstance(d, dict) or d.get("date") != today.isoformat() or not isinstance(d.get("sent"), list):
        return {"date": today.isoformat(), "sent": []}
    return d


def save_alert_state(state: Dict[str, Any]) -> bool:
    return _save("alert_state.json", state)

#!/usr/bin/env python3
"""Intraday watch: emails you as soon as the plan says to act, instead of waiting for the daily digest.

Runs every 30 minutes on weekdays, 08:00-21:00 UK time, on GitHub Actions. An external scheduler (cron-job.org)
dispatches it on time; see the README. It never places or cancels orders and never sends the daily digest itself. It
emails a short alert only when there is something new:

  ACTION    * a sell from the latest digest still hasn't been placed while its market is open (the digest often lands
              after the London close, so the reminder goes out at the next open);
            * the tech circuit-breaker tripped at a month-end close (cut satellites to the risk-off level).
  HEADS-UP  * the Nasdaq-100 trades below the level that would trip the circuit-breaker at month-end (once a day), and
              on the month's last trading day, "today's close decides";
            * a holding is down 10%+ on the day (funds 6%+), or the household's holdings are down 3%+ since the last
              digest. Each comes with what the plan says, which is usually that no rule triggers on a one-day fall.

It also starts today's digest after 09:00 when GitHub's own timer hasn't sent it yet (workflow_dispatch). Each alert
goes out at most once per London day (data/alert_state.json).

Usage: python watch.py [--dry-run] [--force]
  --dry-run  print the alerts; no email, no state written, no dispatch
  --force    ignore the market-hours window and today's dedupe
"""

from __future__ import annotations

import argparse
import html as _html
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("watch")

STOCK_DROP_PCT = -10.0
FUND_DROP_PCT = -6.0
HOUSEHOLD_DROP_PCT = -3.0
REMIND_AFTER_DIGEST_MIN = 90
WATCH_START = (8, 0)
WATCH_END = (21, 0)


# --- small helpers ------------------------------------------------------------------------------------------------
def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def _root(ticker: Any) -> str:
    t = str(ticker or "").upper().strip()
    return t.split(".")[0] if "." in t else t


def london_now() -> datetime:
    import config
    return datetime.now(config.LONDON_TZ)


def in_watch_window(now: datetime) -> bool:
    return now.weekday() < 5 and WATCH_START <= (now.hour, now.minute) < WATCH_END


def _hhmm_to_min(hhmm: Optional[str]) -> Optional[int]:
    try:
        h, m = (int(x) for x in str(hhmm).split(":")[:2])
        return h * 60 + m
    except (TypeError, ValueError):
        return None


def market_status(ticker: str, now: datetime) -> Tuple[bool, str, str]:
    """(open now?, exchange label, 'place between HH:MM and HH:MM UK') for a Yahoo-style ticker, today."""
    import event_calendar as ec
    label, open_l, close_l, tz = ec.exchange_hours(ticker)
    d = now.date()
    o_uk = ec._to_uk_hhmm(d, ec._shift_hhmm(open_l, 15), tz) or "08:15"
    c_uk = ec._to_uk_hhmm(d, ec._shift_hhmm(close_l, -15), tz) or "16:15"
    window = f"between {o_uk} and {c_uk} UK"
    if now.weekday() >= 5:
        return False, label, window
    closed_days = set()
    try:
        if label == "London":
            from strategy_engine import uk_bank_holidays
            closed_days = uk_bank_holidays(d.year)
        elif label == "US":
            closed_days = {ec._parse_date(e["date"]) for e in ec.ONE_OFF_EVENTS
                           if e.get("kind") == "market_closed" and "US" in str(e.get("title")) and "closed" in str(e.get("title"))}
    except Exception:  # pragma: no cover - holiday lookup is best-effort
        closed_days = set()
    if d in closed_days:
        return False, label, window
    mins = now.hour * 60 + now.minute
    o, c = _hhmm_to_min(o_uk), _hhmm_to_min(c_uk)
    return (o is not None and c is not None and o <= mins < c), label, window


# --- inputs (I/O at the edges, so the rules below stay testable) --------------------------------------------------
def fetch_live_accounts() -> Optional[Dict[str, Dict[str, Any]]]:
    """{account: {'positions': [{'ticker','qty','price','ppl'}], 'total', 'free'}} from the live (read-only) API."""
    import config
    from t212_client import Trading212Client, resolve_canonical_ticker
    out: Dict[str, Dict[str, Any]] = {}
    for acc, cfg in (config.ACCOUNTS or {}).items():
        client = Trading212Client(key_id=cfg.get("key_id", ""), secret=cfg.get("secret", ""),
                                  base_url=cfg.get("base_url", "https://live.trading212.com/api/v0"))
        if not client.is_configured():
            continue
        positions = client.get_open_positions()
        time.sleep(1.0)
        cash = client.get_account_cash() or {}
        time.sleep(1.0)
        if positions is None:
            logger.warning("Trading 212 positions unavailable for %s; skipping this run.", acc)
            return None
        rows = []
        for p in positions:
            raw = p.get("ticker", "")
            rows.append({"ticker": p.get("yf_ticker") or resolve_canonical_ticker(raw), "raw": raw,
                         "qty": _num(p.get("quantity"), 0.0), "price": _num(p.get("currentPrice"), 0.0),
                         "ppl": _num(p.get("ppl"), 0.0)})
        out[acc] = {"positions": rows, "total": _num(cash.get("total"), 0.0), "free": _num(cash.get("free"), 0.0)}
    return out or None


def fetch_previous_closes(tickers: List[str], today: date) -> Dict[str, float]:
    """Last daily close strictly before `today` per ticker (one batched Yahoo request; {} on failure)."""
    if not tickers:
        return {}
    try:
        import yfinance as yf
        data = yf.download(sorted(set(tickers)), period="10d", interval="1d", auto_adjust=False, progress=False,
                           group_by="ticker", threads=True)
    except Exception as e:
        logger.warning("Previous closes unavailable: %s", e)
        return {}
    out: Dict[str, float] = {}
    for t in set(tickers):
        try:
            frame = data[t] if len(set(tickers)) > 1 else data
            closes = frame["Close"].dropna()
            closes = closes[[ts.date() < today for ts in closes.index]]
            if len(closes):
                out[t] = float(closes.iloc[-1])
        except Exception:
            continue
    return out


def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def last_digest_sent_at(data_dir: str) -> Optional[datetime]:
    """London datetime of the last digest email (data/last_email_sent.json), if known."""
    import config
    d = _load_json(os.path.join(data_dir, "last_email_sent.json"), {})
    try:
        naive = datetime.strptime(f"{d.get('date')} {d.get('time') or '00:00:00'}", "%Y-%m-%d %H:%M:%S")
        return naive.replace(tzinfo=config.LONDON_TZ)
    except (TypeError, ValueError):
        return None


def last_snapshot(data_dir: str) -> Dict[str, Any]:
    hist = _load_json(os.path.join(data_dir, "portfolio_history.json"), [])
    return hist[-1] if isinstance(hist, list) and hist and isinstance(hist[-1], dict) else {}


# --- rules ----------------------------------------------------------------------------------------------------------
def _fmt_shares(x: float) -> str:
    return f"{x:,.4f}".rstrip("0").rstrip(".")


def unplaced_sell_alerts(actions: Dict[str, Any], live: Dict[str, Dict[str, Any]], now: datetime,
                         digest_sent: Optional[datetime]) -> List[Dict[str, Any]]:
    """Reminder per account for sells from the latest digest (last 3 days) that haven't been placed yet."""
    a_date = actions.get("date") if isinstance(actions, dict) else None
    try:
        d = datetime.strptime(str(a_date), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return []
    today = now.date()
    if d > today or (today - d).days > 3:
        return []
    if digest_sent and digest_sent.date() == today and now - digest_sent < timedelta(minutes=REMIND_AFTER_DIGEST_MIN):
        return []  # the digest only just arrived; don't nag
    alerts = []
    for acc, block in (actions.get("accounts") or {}).items():
        held: Dict[str, float] = {}
        for p in (live.get(acc) or {}).get("positions") or []:
            held[_root(p.get("ticker"))] = held.get(_root(p.get("ticker")), 0.0) + _num(p.get("qty"), 0.0)
        lines, any_open = [], False
        for o in (block or {}).get("sells") or []:
            planned = _num(o.get("shares"), 0.0)
            if planned <= 0:
                continue
            qty_now = held.get(_root(o.get("ticker")), 0.0)
            placed = max(0.0, _num(o.get("qty_before"), qty_now) - qty_now)
            remaining = min(qty_now, planned - placed)
            if remaining <= max(0.0001, planned * 0.02):
                continue
            is_open, label, window = market_status(str(o.get("ticker", "")), now)
            any_open = any_open or is_open
            frac = remaining / planned if planned > 0 else 1.0
            when = "place it now" if is_open else f"place it {window}"
            lines.append(
                f"{acc}: sell {_fmt_shares(remaining)} {o.get('name') or o.get('ticker')} ({o.get('search_symbol') or _root(o.get('ticker'))})"
                f" ≈ £{_num(o.get('estimated_gbp'), 0.0) * frac:,.0f}: {when}"
                f" ({label}{' market open' if is_open else ''}; from the {d.strftime('%a %d %b')} plan)."
            )
        if lines and any_open:
            alerts.append({
                "key": f"remind|{a_date}|{acc}", "level": "action",
                "title": f"{acc}: trim{'s' if len(lines) > 1 else ''} from the latest plan not placed yet",
                "lines": lines,
                "note": "These are the plan's own sells (glide path, caps or circuit-breaker). In an ISA there is no tax "
                        "on selling, gains or losses.",
            })
    return alerts


def breaker_alerts(regime: Optional[Dict[str, Any]], now: datetime, cut_text: str) -> List[Dict[str, Any]]:
    """Circuit-breaker tripped (action) / trading below the trip level / month-end decision day (heads-up)."""
    r = regime if isinstance(regime, dict) else {}
    state, today = r.get("regime"), now.date()
    alerts: List[Dict[str, Any]] = []
    if state == "RISK_OFF":
        month = r.get("as_of_month_end") or ""
        alerts.append({
            "key": f"breaker_tripped|{month}", "level": "action",
            "title": "Tech circuit-breaker tripped at the month-end close",
            "lines": [
                f"The Nasdaq-100 closed {month} {abs(_num(r.get('pct_vs_sma'), 0.0)):.1f}% below its "
                f"{int(_num(r.get('sma_months'), 10))}-month average.",
                f"The plan now says: {cut_text or 'trim satellites above plan'}, tech and semiconductors first. Today's "
                "digest lists the exact sells; if it isn't in your inbox yet, it is on its way.",
            ],
            "note": "This is the plan's crash rule. Historically it halved the worst falls of a tech-heavy portfolio; "
                    "the cash is reinvested by the normal buy rules once the trend recovers.",
        })
        return alerts
    if state != "RISK_ON":
        return alerts
    trip, latest, dist = r.get("trip_level"), r.get("latest_close"), r.get("trip_distance_pct")
    if not trip or not latest or dist is None:
        return alerts
    try:
        from event_calendar import _last_london_business_day, day_label
        check = _last_london_business_day(today.year, today.month)
    except Exception:  # pragma: no cover
        check, day_label = None, (lambda d, a: str(d))
    proxy = r.get("proxy") or "Nasdaq-100"
    fmt = (lambda v: f"{v:,.0f}") if max(trip, latest) >= 1000 else (lambda v: f"{v:,.2f}")
    if check and today == check and (now.hour, now.minute) >= (14, 0) and abs(dist) <= 3.0:
        alerts.append({
            "key": f"breaker_decides|{today.isoformat()}", "level": "warning",
            "title": "Today's close decides the tech circuit-breaker",
            "lines": [
                f"{proxy} is at {fmt(latest)}; it trips if it closes below {fmt(trip)} today "
                f"({'already below' if dist > 0 else f'{abs(dist):.1f}% above'}).",
                f"If it trips, tomorrow's email will tell you to {cut_text or 'trim satellites above plan'}. "
                "Nothing to do before the close.",
            ],
            "note": "Only the month-end close counts; that is the rule that was tested.",
        })
    elif dist > 0:
        alerts.append({
            "key": f"breaker_below|{today.isoformat()}", "level": "warning",
            "title": "Nasdaq-100 is below the circuit-breaker level (no action yet)",
            "lines": [
                f"{proxy} is at {fmt(latest)}, {dist:.1f}% below the level ({fmt(trip)}) that would trip the "
                f"circuit-breaker at the month-end close"
                + (f" ({day_label(check, today)})." if check else "."),
                f"Nothing to do today. If it is still below at that close, the next email will tell you to "
                f"{cut_text or 'trim satellites above plan'}.",
            ],
            "note": "Intra-month dips don't count: acting on daily swings sells low more often than it helps.",
        })
    return alerts


def move_alerts(live: Dict[str, Dict[str, Any]], prev_closes: Dict[str, float], today_sells: set,
                names: Dict[str, str], now: datetime, is_fund: Callable[[str], bool]) -> List[Dict[str, Any]]:
    """Heads-up for holdings down 10%+ (funds 6%+) on the day, vs the previous close."""
    by_root: Dict[str, Dict[str, Any]] = {}
    for acc, blk in (live or {}).items():
        for p in blk.get("positions") or []:
            t = str(p.get("ticker") or "")
            prev = prev_closes.get(t)
            px = _num(p.get("price"), 0.0)
            if not prev or prev <= 0 or px <= 0:
                continue
            ratio = px / prev
            if not 0.5 <= ratio <= 1.5:  # quote-unit mismatch (pence vs pounds): try to reconcile, else skip
                for adj in (px / 100.0, px * 100.0):
                    if 0.5 <= adj / prev <= 1.5:
                        ratio = adj / prev
                        break
                else:
                    continue
            chg = (ratio - 1.0) * 100.0
            item = by_root.setdefault(_root(t), {"ticker": t, "chg": chg, "accounts": []})
            item["accounts"].append(acc)
    alerts = []
    for root, it in by_root.items():
        limit = FUND_DROP_PCT if is_fund(it["ticker"]) else STOCK_DROP_PCT
        if it["chg"] > limit:
            continue
        name = names.get(root) or root
        label = root if name in (root, it["ticker"]) else f"{name} ({root})"
        on_list = root in today_sells
        alerts.append({
            "key": f"move|{root}|{now.date().isoformat()}", "level": "warning",
            "title": f"{label} {it['chg']:+.1f}% today",
            "lines": [
                f"Held in: {', '.join(sorted(set(it['accounts'])))}.",
                "It's already on your sell list: place that trim." if on_list else
                "What the plan says: no sell signal. No rule triggers on a one-day fall; the daily digest re-checks "
                "trends and caps, and the month-end circuit-breaker is the crash rule.",
            ],
            "note": "Big one-day drops without news have tended to partly bounce back; news-driven ones to drift on. "
                    "There's no reliable edge in selling after the fact.",
        })
    return alerts


def household_alert(live: Dict[str, Dict[str, Any]], snapshot: Dict[str, Any], now: datetime) -> List[Dict[str, Any]]:
    """Heads-up when open positions have lost 3%+ of household value since the last digest (P&L based, so cash
    deposits/withdrawals don't count)."""
    base_total = _num(snapshot.get("combined_total"), 0.0)
    base_ppl = snapshot.get("total_ppl")
    if base_total <= 0 or base_ppl is None:
        return []
    live_ppl = sum(_num(p.get("ppl"), 0.0) for blk in (live or {}).values() for p in blk.get("positions") or [])
    move = live_ppl - _num(base_ppl, 0.0)
    pct = move / base_total * 100.0
    if pct > HOUSEHOLD_DROP_PCT:
        return []
    when = str(snapshot.get("updated_at") or snapshot.get("date") or "the last digest")[:16].replace("T", " ")
    return [{
        "key": f"household|{now.date().isoformat()}", "level": "warning",
        "title": f"Holdings down {pct:.1f}% since the last digest",
        "lines": [
            f"Open positions are down about £{abs(move):,.0f} since {when} (≈ {pct:.1f}% of the household).",
            "What the plan says: nothing to do unless an action above says so. The 40/55/5 mix moves like this in a "
            "tech sell-off; the month-end circuit-breaker is the crash rule.",
        ],
        "note": "Selling into a sell-off on daily moves has historically cost more than it saved.",
    }]


# --- delivery -----------------------------------------------------------------------------------------------------
def render_alert_html(alerts: List[Dict[str, Any]], now: datetime) -> str:
    esc = _html.escape
    blocks = ""
    for a in alerts:
        color, bg, tag = (("#b91c1c", "#fef2f2", "DO NOW") if a.get("level") == "action"
                          else ("#b45309", "#fffbeb", "HEADS-UP"))
        lines = "".join(f"<li style='margin-bottom: 3px;'>{esc(str(l))}</li>" for l in a.get("lines") or [])
        note = f"<div style='font-size: 12px; color: #64748b; margin-top: 4px;'>{esc(str(a['note']))}</div>" if a.get("note") else ""
        blocks += (
            f"<div style='background: {bg}; border-left: 4px solid {color}; border-radius: 8px; padding: 10px 12px; "
            f"margin-bottom: 10px;'><div style='font-size: 11px; font-weight: 800; color: {color};'>{tag}</div>"
            f"<div style='font-size: 15px; font-weight: 800; color: #0f172a; margin: 2px 0 4px;'>{esc(str(a.get('title') or ''))}</div>"
            f"<ul style='margin: 0 0 0 18px; padding: 0; font-size: 13px; color: #334155;'>{lines}</ul>{note}</div>"
        )
    return (
        "<html><body style='font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: #f8fafc; "
        "padding: 16px;'><div style='max-width: 640px; margin: 0 auto;'>"
        f"<div style='font-size: 12px; color: #64748b; margin-bottom: 8px;'>Intraday watch · {now.strftime('%a %d %b %H:%M')} UK · "
        "advisory only: nothing is traded for you</div>"
        f"{blocks}"
        "<div style='font-size: 11px; color: #94a3b8;'>Sent only when something needs your attention, at most once per "
        "trigger per day. The daily digest still has the full picture.</div></div></body></html>"
    )


def alert_subject(alerts: List[Dict[str, Any]]) -> str:
    acts = [a for a in alerts if a.get("level") == "action"]
    if acts:
        return f"🚨 Action needed: {acts[0]['title']}" + (f" (+{len(alerts) - 1} more)" if len(alerts) > 1 else "")
    return f"⚠️ Heads-up: {alerts[0]['title']}" + (f" (+{len(alerts) - 1} more)" if len(alerts) > 1 else "")


def dispatch_digest_if_due(now: datetime, data_dir: str, state: Dict[str, Any], dry_run: bool) -> Optional[str]:
    """Starts today's digest workflow (workflow_dispatch) after 09:00 when it hasn't been sent; returns a state key."""
    if (now.hour, now.minute) < (9, 0) or now.hour >= 20:
        return None
    sent = last_digest_sent_at(data_dir)
    if sent and sent.date() == now.date():
        return None
    key = f"digest_dispatch|{now.date().isoformat()}|{now.hour // 2}"
    if key in state.get("sent", []):
        return None
    repo, token = os.getenv("GITHUB_REPOSITORY"), os.getenv("GITHUB_TOKEN")
    if not repo or not token:
        return None
    if dry_run:
        logger.info("[dry-run] Would dispatch today's digest (not sent yet).")
        return None
    try:
        import requests
        resp = requests.post(
            f"https://api.github.com/repos/{repo}/actions/workflows/daily_trading_digest.yml/dispatches",
            headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
                     "X-GitHub-Api-Version": "2022-11-28"},
            json={"ref": "main"}, timeout=20,
        )
        logger.info("Digest dispatch: HTTP %s", resp.status_code)
        return key if resp.status_code == 204 else None
    except Exception as e:
        logger.warning("Digest dispatch failed: %s", e)
        return None


# --- orchestration --------------------------------------------------------------------------------------------------
def run(dry_run: bool = False, force: bool = False) -> int:
    import action_state
    now = london_now()
    if not force and not in_watch_window(now):
        logger.info("Outside the watch window (weekdays 08:00-21:00 UK); nothing to do.")
        return 0
    data_dir = str(action_state.DATA_DIR)
    state = action_state.load_alert_state(now.date())
    sent_keys = set() if force else set(state.get("sent", []))

    dispatched = dispatch_digest_if_due(now, data_dir, state, dry_run)

    live = fetch_live_accounts()
    if not live:
        logger.warning("No live account data; skipping this run.")
        if dispatched:
            state["sent"] = sorted(set(state.get("sent", [])) | {dispatched})
            action_state.save_alert_state(state)
        return 0

    actions = action_state.load_latest_actions()
    names: Dict[str, str] = {}
    today_sells: set = set()
    for blk in (actions.get("accounts") or {}).values():
        for o in (blk or {}).get("sells") or []:
            names[_root(o.get("ticker"))] = o.get("name") or o.get("ticker")
            if actions.get("date") == now.date().isoformat():
                today_sells.add(_root(o.get("ticker")))

    alerts: List[Dict[str, Any]] = []
    try:
        alerts += unplaced_sell_alerts(actions, live, now, last_digest_sent_at(data_dir))
    except Exception as e:
        logger.warning("Reminder check failed: %s", e)
    try:
        from event_calendar import _breaker_cut_text
        from strategy_engine import compute_tech_regime
        alerts += breaker_alerts(compute_tech_regime(), now, _breaker_cut_text())
    except Exception as e:
        logger.warning("Circuit-breaker check failed: %s", e)
    try:
        from strategy_engine import (BUCKET_SATELLITE, NASDAQ100_ETF_KEYS, SECTOR_ETF_KEYS, _SEMIS_FUND_KEYS,
                                     classify_bucket)

        def is_fund(t: str) -> bool:
            sym = t.upper()
            return classify_bucket(t) != BUCKET_SATELLITE or any(k in sym for k in NASDAQ100_ETF_KEYS + _SEMIS_FUND_KEYS + SECTOR_ETF_KEYS)

        tickers = [p["ticker"] for blk in live.values() for p in blk["positions"] if p.get("ticker")]
        alerts += move_alerts(live, fetch_previous_closes(tickers, now.date()), today_sells, names, now, is_fund)
    except Exception as e:
        logger.warning("Big-move check failed: %s", e)
    try:
        alerts += household_alert(live, last_snapshot(data_dir), now)
    except Exception as e:
        logger.warning("Household check failed: %s", e)

    new = [a for a in alerts if a["key"] not in sent_keys]
    new.sort(key=lambda a: 0 if a.get("level") == "action" else 1)
    # Log only the trigger keys (no amounts): workflow logs can be public on a public fork
    for a in alerts:
        logger.info("%s %s: %s", "NEW" if a in new else "already sent", a["level"].upper(), a["key"])
    if not new:
        logger.info("Nothing new to tell you (%d alert(s) already sent today).", len(alerts))
        if dispatched and not dry_run:
            state["sent"] = sorted(set(state.get("sent", [])) | {dispatched})
            action_state.save_alert_state(state)
        return 0
    subject = alert_subject(new)
    html_body = render_alert_html(new, now)
    if dry_run:
        print(f"\n[dry-run] Would email: {subject}\n")
        for a in new:
            print(f"  [{a['level'].upper()}] {a['title']}")
            for line in a.get("lines") or []:
                print(f"     - {line}")
        return 0
    from mailer import send_email_digest
    ok = send_email_digest(html_body, subject_prefix=subject)
    keys = {a["key"] for a in new} if ok else set()
    if dispatched:
        keys.add(dispatched)
    if keys:
        state["sent"] = sorted(set(state.get("sent", [])) | keys)
        action_state.save_alert_state(state)
    logger.info("Alert email %s (%d alert(s)).", "sent" if ok else "NOT sent", len(new))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Intraday watch: email only when the plan says to act.")
    parser.add_argument("--dry-run", action="store_true", help="print alerts; no email, state or dispatch")
    parser.add_argument("--force", action="store_true", help="ignore the market-hours window and today's dedupe")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        return run(dry_run=args.dry_run, force=args.force)
    except Exception as e:  # never fail noisily on a data hiccup; the next run tries again
        logger.exception("Intraday watch failed: %s", e)
        return 0


if __name__ == "__main__":
    sys.exit(main())

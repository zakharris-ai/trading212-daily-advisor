"""Main entry point orchestrating portfolio retrieval, deep research analysis, watchlist scanning, and rebalancing."""

import argparse
from datetime import datetime
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from config import ACCOUNTS, EARMARKED_TICKERS, LONDON_TZ, MODEL_VERSION, REBALANCE_CONFIG, RECIPIENT_EMAILS
from mailer import format_price, generate_html_digest, send_email_digest
from portfolio_storage import (
    append_recommendation_records,
    compute_day_over_day_delta,
    compute_external_flows,
    detect_and_record_trades,
    get_position_entries,
    get_position_trims,
    load_portfolio_history,
    save_daily_snapshot,
)
from dashboard_generator import generate_dashboard_html
from research_agent import DeepResearchAgent
from strategy_engine import (
    analyze_position,
    build_rebalancing_plan,
    build_consolidated_portfolio_view,
    generate_tldr_summary,
    get_gbp_usd_rate,
    scan_watchlist_opportunities,
)
from t212_client import Trading212Client, resolve_canonical_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def london_today() -> str:
  """Today's calendar date in Europe/London (YYYY-MM-DD).

  Single source for every date this run derives (dedupe marker, snapshot, trade log, ledger), so a
  UTC runner and a London laptop always agree on what "today" is.
  """
  return datetime.now(LONDON_TZ).strftime("%Y-%m-%d")


def _abort_run(message: str, *args: Any) -> None:
  """Logs a fatal data-integrity error and exits non-zero BEFORE any state is written.

  Nothing (snapshot, trade log, entries/trims, email, sent-marker, ledger) has been persisted at the
  call sites, so the next scheduled tick simply retries from a clean state.
  """
  logger.error(message, *args)
  logger.error(
      "Aborting run (exit code 1): no snapshot, trade log, email or sent-marker was written; "
      "the next scheduled tick will retry."
  )
  sys.exit(1)


def _num(val: Any, default: float = 0.0) -> float:
  """Float conversion mapping None/NaN/garbage to `default`."""
  try:
    f = float(val)
  except (TypeError, ValueError):
    return default
  return default if (math.isnan(f) or math.isinf(f)) else f


def _fmt(val: Any, spec: str = "{:.1f}", na: str = "n/a") -> str:
  """Formats a number for the console, showing `na` (never an optimistic default) when it is missing."""
  try:
    f = float(val)
  except (TypeError, ValueError):
    return na
  if math.isnan(f) or math.isinf(f):
    return na
  return spec.format(f)


def _reserve_figures(plan: Dict[str, Any]) -> Tuple[float, float, float]:
  """(reserve floor, funded by free cash, shortfall) in GBP, from the plan's C5 fields with safe fallbacks."""
  floor = _num(plan.get("reserve_floor_gbp"), _num(plan.get("buffer_reserve_gbp"), 0.0))
  cash = max(0.0, _num(plan.get("available_cash_gbp"), 0.0))
  funded = _num(plan.get("reserve_funded_gbp"), min(cash, floor))
  shortfall = _num(plan.get("reserve_shortfall_gbp"), max(0.0, floor - funded))
  return floor, funded, shortfall


def _reserve_status_text(plan: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> str:
  """Plain-English cash-reserve status: FUNDED only when free cash actually covers the floor."""
  floor, funded, shortfall = _reserve_figures(plan)
  topup_note = ""
  if cfg and _num(cfg.get("reserve_topup_gbp"), 0.0) > 0:
    until = cfg.get("reserve_topup_until")
    topup_note = (
        f" (includes £{_num(cfg.get('reserve_topup_gbp')):,.0f} temporary top-up"
        + (f" until {until}" if until else "")
        + ")"
    )
  if shortfall > 0.5:
    return (
        f"Cash Reserve Floor £{floor:,.0f}{topup_note}: UNFUNDED — only £{funded:,.0f} is held in free cash"
        f" (£{shortfall:,.0f} short). Buys stay blocked; this money is not sitting in cash to withdraw."
    )
  return (
      f"Cash Reserve Floor £{floor:,.0f}{topup_note}: FUNDED — £{funded:,.0f} held in free cash;"
      f" buys never dip below it."
  )


def _earmarked_figures(plan: Dict[str, Any], analyses: Optional[List[Dict[str, Any]]] = None) -> Tuple[float, List[str]]:
  """(earmarked GBP, earmarked tickers) for one account, from the plan's C5 fields.

  If the plan carries neither earmarked_gbp nor earmarked_tickers (older strategy_engine), the configured
  EARMARKED_TICKERS held in this account are valued from its analyses instead. Display only: this never
  feeds a trading decision.
  """
  amount = plan.get("earmarked_gbp")
  tickers = plan.get("earmarked_tickers")
  if amount is None and tickers is None:
    earmark_set = {resolve_canonical_ticker(t) for t in EARMARKED_TICKERS}
    held = [
        a for a in (analyses or [])
        if resolve_canonical_ticker(str(a.get("ticker") or "")) in earmark_set
    ]
    tickers = sorted({resolve_canonical_ticker(str(a.get("ticker") or "")) for a in held})
    amount = sum(max(0.0, _num(a.get("value_gbp"), 0.0)) for a in held)
  if isinstance(tickers, str):
    tickers = [tickers]
  return _num(amount, 0.0), sorted(str(t) for t in (tickers or []) if t)


def _build_consolidated_view_with_plans(
    accounts_data: Dict[str, Any],
    gbp_rate: float,
    account_plans: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
  """C7: consolidated view with per-account plans (post-trade metrics no longer collapse shared tickers).

  Falls back to the legacy signature when strategy_engine does not accept `account_plans` yet.
  """
  # Compatibility shim: older strategy_engine versions do not accept `account_plans`; remove once every
  # deployment ships build_consolidated_portfolio_view(..., account_plans=...).
  try:
    return build_consolidated_portfolio_view(accounts_data, gbp_rate=gbp_rate, account_plans=account_plans)
  except TypeError as e:
    if "account_plans" not in str(e):
      raise  # a genuine TypeError inside the function must not be masked
    logger.warning(
        "build_consolidated_portfolio_view() does not accept account_plans yet (%s); "
        "retrying without per-account plans.",
        e,
    )
    return build_consolidated_portfolio_view(accounts_data, gbp_rate=gbp_rate)


def _finite_or_none(val: Any) -> Optional[float]:
  """Returns a finite float, or None (keeps the JSON ledger strictly valid)."""
  try:
    f = float(val)
  except (TypeError, ValueError):
    return None
  return None if (math.isnan(f) or math.isinf(f)) else f


def _build_ledger_records(results_by_account: Dict[str, Any], date_str: str) -> List[Dict[str, Any]]:
  """One recommendation-ledger record per recommended (non-pending) sell/trim and buy order."""
  records: List[Dict[str, Any]] = []
  for acc_name, data in results_by_account.items():
    plan = data.get("rebalancing_plan") or {}
    sells = [s for s in plan.get("sell_orders", []) or [] if not s.get("is_pending")]
    for order in sells + list(plan.get("buy_orders", []) or []):
      amount = _finite_or_none(order.get("estimated_gbp"))
      records.append({
          "date": date_str,
          "model_version": MODEL_VERSION,
          "account": acc_name,
          "action": order.get("action"),
          "ticker": order.get("ticker"),
          "amount_gbp": round(amount, 2) if amount is not None else None,
          "price": _finite_or_none(order.get("price")),
          "currency": order.get("currency"),
          "reason": order.get("plain_english_why") or order.get("reason") or "",
      })
  return records

# Realistic sample data used for testing and offline verification
DEFAULT_MOCK_PORTFOLIO = {
    "cash": {"free": 2450.00, "ppl": 2150.00},
    "positions": [
        {
            "ticker": "NVDA_US_EQ",
            "yf_ticker": "NVDA",
            "quantity": 30.0,
            "averagePrice": 112.50,
            "ppl": 480.00,
        },
        {
            "ticker": "MSFT_US_EQ",
            "yf_ticker": "MSFT",
            "quantity": 15.0,
            "averagePrice": 410.00,
            "ppl": 525.00,
        },
        {
            "ticker": "TSLA_US_EQ",
            "yf_ticker": "TSLA",
            "quantity": 25.0,
            "averagePrice": 220.00,
            "ppl": -250.00,
        },
        {
            "ticker": "VUAAl_EQ",
            "yf_ticker": "VUAA.L",
            "quantity": 80.0,
            "averagePrice": 75.00,
            "ppl": 496.00,
        },
    ],
}

MOCK_PORTFOLIOS = {
    "Main": DEFAULT_MOCK_PORTFOLIO,
    "Secondary": {
        "cash": {"free": 1100.00, "ppl": 1340.00},
        "positions": [
            {
                "ticker": "AAPL_US_EQ",
                "yf_ticker": "AAPL",
                "quantity": 20.0,
                "averagePrice": 195.00,
                "ppl": 580.00,
            },
            {
                "ticker": "AMZN_US_EQ",
                "yf_ticker": "AMZN",
                "quantity": 25.0,
                "averagePrice": 170.00,
                "ppl": 450.00,
            },
            {
                "ticker": "VWRPl_EQ",
                "yf_ticker": "VWRP.L",
                "quantity": 60.0,
                "averagePrice": 82.00,
                "ppl": 310.00,
            },
        ],
    },
}


def run_pipeline(
    dry_run: bool = False,
    use_mock: bool = False,
    force: bool = False,
    open_dashboard: bool = False,
    serve_port: Optional[int] = None,
):
  if not use_mock:
    any_configured = any(
        Trading212Client(
            key_id=cfg.get("key_id", ""),
            secret=cfg.get("secret", ""),
            base_url=cfg.get("base_url", "https://live.trading212.com/api/v0"),
        ).is_configured()
        for cfg in ACCOUNTS.values()
    )
    if not any_configured:
      # Fail closed (exit 1) so a scheduled run with missing secrets shows as failed instead of silently passing.
      logger.error("No Trading 212 API credentials configured in .env (or GitHub Actions secrets).")
      print("\n⚠️  No Trading 212 API key configured yet!")
      print("👉 Run './setup.sh' to launch the interactive setup wizard, or edit '.env' with your T212_KEY_ID and T212_SECRET.")
      print("💡 Tip: Want to test the engine with sample data first? Run: ./run.sh --dry-run --mock\n")
      sys.exit(1)

  logger.info(
      "Starting Deep Research Trading Intelligence & Capital Rebalancing Run..."
  )
  run_date = london_today()

  # 1. Macro Climate Research
  research_agent = DeepResearchAgent()
  macro_regime = research_agent.get_macro_regime()
  logger.info(
      "Macro Climate: %s (VIX: %s, 10Y: %s%%)",
      macro_regime.get("risk_climate", "n/a"),
      _fmt(macro_regime.get("vix")),
      _fmt(macro_regime.get("us10y_yield"), "{:.2f}"),
  )
  if not macro_regime.get("macro_data_ok", True):
    logger.warning(
        "Macro data incomplete (missing: %s); regime readings are partial.",
        ", ".join(str(m) for m in (macro_regime.get("macro_missing") or [])) or "unspecified",
    )

  try:
    gbp_rate = get_gbp_usd_rate()
  except RuntimeError as e:
    _abort_run("Live GBP/USD rate unavailable: %s", e)
  checked_rate = _finite_or_none(gbp_rate)
  if checked_rate is None or checked_rate <= 0:
    _abort_run("Invalid GBP/USD rate returned: %r", gbp_rate)
  gbp_rate = checked_rate
  logger.info("Live Currency Anchor: 1 GBP = $%.4f USD", gbp_rate)

  raw_accounts_data = {}

  for account_name, cfg in ACCOUNTS.items():
    analyses = []
    cash_data = {}

    if use_mock:
      logger.info("Loading mock portfolio for: %s", account_name)
      mock_data = MOCK_PORTFOLIOS.get(
          account_name, DEFAULT_MOCK_PORTFOLIO
      )
      cash_data = dict(mock_data["cash"])
      raw_positions = [dict(p) for p in mock_data["positions"]]
      pending_orders_data = mock_data.get("pending_orders", {"has_orders_scope": False, "orders": [], "pending_buys": {}, "pending_sells": {}})
    else:
      client = Trading212Client(
          key_id=cfg.get("key_id", ""),
          secret=cfg.get("secret", ""),
          base_url=cfg.get("base_url", "https://live.trading212.com/api/v0"),
      )
      if not client.is_configured():
        logger.info(
            "Account '%s' is not configured yet. Skipping.", account_name
        )
        continue

      logger.info("Connecting to live Trading 212 API for: %s...", account_name)
      import time
      time.sleep(1.0)
      raw_positions = client.get_open_positions()
      if raw_positions is None:
        _abort_run(
            "Trading 212 positions unavailable for account '%s' (API error, timeout or malformed response).",
            account_name,
        )
      time.sleep(1.0)
      raw_cash = client.get_account_cash()

      # Data-integrity guard: the broker reports invested capital but returned no positions, so the
      # positions payload is incomplete. Never treat that as an empty account.
      broker_invested = _num((raw_cash or {}).get("invested"), 0.0)
      if not raw_positions and broker_invested > 0:
        _abort_run(
            "Trading 212 cash for account '%s' shows %.2f invested but the positions endpoint returned none.",
            account_name,
            broker_invested,
        )
      time.sleep(1.0)
      pending_orders_data = client.get_pending_orders()

      # Cash resolution (live broker API is authoritative; fallback only on API failure)
      if raw_cash is not None and "free" in raw_cash:
        cash_data = raw_cash
        free_cash = float(cash_data.get("free", 0.0) or 0.0)
        blocked_cash = float(cash_data.get("blocked", 0.0) or 0.0)
        cash_curr = cash_data.get("currency", cfg.get("cash_currency", "GBP"))
      else:
        logger.warning(
            "Could not fetch live cash balance from Trading 212 API for %s; using fallback.",
            account_name,
        )
        cash_data = {}
        free_cash = float(cfg.get("manual_cash", 0.0) or 0.0)
        blocked_cash = 0.0
        cash_curr = cfg.get("cash_currency", "GBP")

      if cash_curr.upper() == "USD" and free_cash > 0:
        free_cash_gbp = free_cash / gbp_rate
        cash_label = f"${free_cash:,.2f} (£{free_cash_gbp:,.2f})"
        blocked_cash_gbp = blocked_cash / gbp_rate
      else:
        free_cash_gbp = free_cash
        cash_label = f"£{free_cash:,.2f}"
        blocked_cash_gbp = blocked_cash

      cash_data["free"] = free_cash
      cash_data["free_gbp"] = free_cash_gbp
      cash_data["free_label"] = cash_label
      cash_data["blocked"] = blocked_cash
      cash_data["blocked_gbp"] = blocked_cash_gbp

      logger.info(
          "Cash resolved for %s: Free=%s (GBP: £%.2f, Blocked: £%.2f, Total: £%.2f)",
          account_name,
          cash_label,
          free_cash_gbp,
          blocked_cash_gbp,
          cash_data.get("total", 0.0) or 0.0,
      )

      recent_txs = cash_data.get("recent_transactions", [])
      for tx in recent_txs:
        if tx.get("type") in ("WITHDRAW", "DEPOSIT"):
          logger.info(
              "Recent cash movement for %s: %s %.2f %s at %s",
              account_name,
              tx.get("type"),
              tx.get("amount", 0.0),
              tx.get("currency", "GBP"),
              tx.get("dateTime", ""),
          )

      if not cash_data.get("total") and raw_positions:
        total_ppl = sum(pos.get("ppl", 0.0) or 0.0 for pos in raw_positions)
        cash_data["ppl"] = total_ppl

    logger.info(
        "Retrieved %d positions for %s.", len(raw_positions), account_name
    )

    for pos in raw_positions:
      raw_t = pos.get("ticker", "")
      yf_ticker = pos.get("yf_ticker") or resolve_canonical_ticker(raw_t)
      pos["yf_ticker"] = yf_ticker

    raw_accounts_data[account_name] = {
        "cfg": cfg,
        "cash": cash_data,
        "raw_positions": raw_positions,
        "pending_orders": pending_orders_data,
    }

  if not raw_accounts_data:
    # Fail closed: nothing was fetched, so no report, snapshot, email or sent-marker is produced.
    logger.error("No account data available to generate report (no configured Trading 212 account returned data).")
    print("\n⚠️  No Trading 212 account data available!")
    print("👉 Run './setup.sh' to launch the interactive setup wizard, or edit '.env' with your T212_KEY_ID and T212_SECRET.")
    print("💡 Tip: Want to test the engine with sample data first? Run: ./run.sh --dry-run --mock\n")
    sys.exit(1)

  # Detect and record trade executions since previous snapshot BEFORE auditing positions
  # (writes the trade log, entries, trims and closed archive, so never in --dry-run or --mock)
  if not use_mock and not dry_run:
    detected_trades = detect_and_record_trades(raw_accounts_data, current_date_str=run_date)
    if detected_trades:
      logger.info("Auto-detected %d trade execution events.", len(detected_trades))

  # Conduct deep fundamental and technical audit on existing holdings with fresh trade/trim history
  for account_name, acc_info in raw_accounts_data.items():
    analyses = []
    tracked_entries = get_position_entries().get(account_name, {})
    tracked_trims = get_position_trims().get(account_name, {})
    for pos in acc_info["raw_positions"]:
      yf_ticker = pos["yf_ticker"]
      raw_t = pos.get("ticker", "")
      canon_t = resolve_canonical_ticker(yf_ticker) if yf_ticker else ""
      if canon_t in tracked_entries:
        pos["last_entry_date"] = tracked_entries[canon_t]
      elif yf_ticker in tracked_entries:
        pos["last_entry_date"] = tracked_entries[yf_ticker]
      elif raw_t in tracked_entries:
        pos["last_entry_date"] = tracked_entries[raw_t]

      if canon_t in tracked_trims:
        pos["last_trim_date"] = tracked_trims[canon_t]
      elif yf_ticker in tracked_trims:
        pos["last_trim_date"] = tracked_trims[yf_ticker]
      elif raw_t in tracked_trims:
        pos["last_trim_date"] = tracked_trims[raw_t]

      logger.info("Auditing %s for %s...", yf_ticker, account_name)
      analysis = analyze_position(yf_ticker, pos, gbp_rate=gbp_rate)
      analyses.append(analysis)
    acc_info["analyses"] = analyses

  # 1. Build Consolidated Unified Portfolio View
  consolidated_view = build_consolidated_portfolio_view(
      raw_accounts_data, gbp_rate=gbp_rate
  )

  # 2. Deep Research Watchlist Scan for High-Alpha Opportunities
  logger.info("Scanning watchlist for high-conviction alpha...")
  # scan_collector receives EVERY non-excluded watchlist ticker's analysis (including gate failures),
  # so the dashboard can show real per-ticker data rather than only the tickers that passed.
  watchlist_scan = []
  new_opps = scan_watchlist_opportunities(set(), gbp_rate=gbp_rate, scan_collector=watchlist_scan)

  # 3. Generate Account-Isolated Rebalancing Plans with Unified Global Context
  results_by_account = {}
  for account_name, acc_info in raw_accounts_data.items():
    cfg = acc_info["cfg"]
    cash_data = acc_info["cash"]
    analyses = acc_info["analyses"]
    pending_orders_data = acc_info["pending_orders"]

    available_cash = cash_data.get("free_gbp", cash_data.get("free", 0.0))
    blocked_cash_val = cash_data.get("blocked_gbp", cash_data.get("blocked", 0.0))
    min_reserve = cfg.get("min_cash_reserve_gbp", 1500.0)

    official_total = float(cash_data.get("total", 0.0) or 0.0)
    rebalance_plan = build_rebalancing_plan(
        analyses=analyses,
        available_cash_gbp=available_cash,
        new_opportunities=new_opps,
        gbp_rate=gbp_rate,
        min_cash_reserve_gbp=min_reserve,
        blocked_cash_gbp=blocked_cash_val,
        pending_orders_data=pending_orders_data,
        global_context=consolidated_view,
        official_total_gbp=official_total if official_total > 0 else None,
        account_name=account_name,
    )

    results_by_account[account_name] = {
        "cash": cash_data,
        "pending_orders": pending_orders_data,
        "analyses": analyses,
        "new_opportunities": new_opps,
        "rebalancing_plan": rebalance_plan,
        "consolidated_view": consolidated_view,
    }

  # 4. Rebuild the consolidated view with each account's own plan, so post-trade metrics are computed
  #    per account (tickers held in both accounts are no longer collapsed into one entry).
  account_plans = {
      acc_name: res["rebalancing_plan"] for acc_name, res in results_by_account.items()
  }
  consolidated_view = _build_consolidated_view_with_plans(raw_accounts_data, gbp_rate, account_plans)
  for res in results_by_account.values():
    res["consolidated_view"] = consolidated_view

  # 5. Compute day-over-day net worth and P&L changes against prior snapshot
  #    (net of deposits/withdrawals since that snapshot, so new money is not reported as a gain)
  current_acc_totals = {
      acc_name: acc_info["cash"].get("total") or consolidated_view["account_shares"][acc_name]["total_value_gbp"]
      for acc_name, acc_info in raw_accounts_data.items()
  }
  external_flows = compute_external_flows(
      raw_accounts_data, current_date_str=run_date, gbp_rate=gbp_rate
  )
  dod_metrics = compute_day_over_day_delta(
      current_combined_total=consolidated_view["total_net_worth_gbp"],
      current_account_totals=current_acc_totals,
      current_date_str=run_date,
      external_flows_by_account=external_flows.get("by_account", {}),
  )
  consolidated_view["day_over_day"] = dod_metrics

  # Build the HTML output with Consolidated Portfolio Overview
  html_report = generate_html_digest(
      results_by_account, macro_regime=macro_regime, consolidated_view=consolidated_view
  )

  # Save a local preview copy for review
  output_path = os.path.join(os.path.dirname(__file__), "latest_digest.html")
  with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_report)
  logger.info("Saved local preview to %s", output_path)

  # 6. Persist structured daily snapshot and update portfolio history (only in real live dispatch runs)
  if not use_mock and not dry_run:
    snapshot_file = save_daily_snapshot(
        accounts_data=raw_accounts_data,
        consolidated_view=consolidated_view,
        macro_regime=macro_regime,
        results_by_account=results_by_account,
        gbp_rate=gbp_rate,
        snapshot_date=run_date,
        external_flows=external_flows,
    )
    logger.info("Persisted daily snapshot to %s", snapshot_file)

  # 7. Generate and persist interactive Web Dashboard
  history_data = load_portfolio_history()
  dashboard_html = generate_dashboard_html(
      results_by_account=results_by_account,
      consolidated_view=consolidated_view,
      macro_regime=macro_regime,
      gbp_rate=gbp_rate,
      history=history_data,
      watchlist_scan=watchlist_scan,
  )
  dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
  with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(dashboard_html)
  logger.info("Saved interactive web dashboard to %s", dashboard_path)

  # Print terminal summary and build plain-text email alternative
  summary_lines = []
  def out(line=""):
    print(line)
    summary_lines.append(line)

  out("\n" + "=" * 85)
  out(
      "⚡ WHAT YOU NEED TO DO TODAY —"
      f" {datetime.now(LONDON_TZ).strftime('%Y-%m-%d %H:%M')}"
  )
  if not macro_regime.get("macro_data_ok", True):
    missing_macro = ", ".join(str(m) for m in (macro_regime.get("macro_missing") or [])) or "unspecified inputs"
    out("!" * 85)
    out(f"⚠️ MACRO DATA INCOMPLETE — missing: {missing_macro}.")
    out("   The market backdrop below is built from partial data; treat the regime reading as unverified.")
    out("!" * 85)
  out(f"🌍 Market Backdrop: {macro_regime.get('risk_climate') or 'n/a'} (Macro Score: {_fmt(macro_regime.get('macro_score'), '{:.0f}')}/100)")
  out(f"🛡️ Crash Defcon: {macro_regime.get('crash_defcon') or 'n/a'} (Tail-Crash Prob: {_fmt(macro_regime.get('crash_probability_pct'), '{:.0f}')}%) | {macro_regime.get('crash_shield_directive', '')}")
  out(f"   VIX: {_fmt(macro_regime.get('vix'))} (Term Structure: {_fmt(macro_regime.get('vix_term_structure'), '{:.2f}')}x) | Inst. Distribution: SPY {_fmt(macro_regime.get('spy_dist_days'), '{:.0f}')}d / QQQ {_fmt(macro_regime.get('qqq_dist_days'), '{:.0f}')}d ({macro_regime.get('dist_status') or 'n/a'})")
  out(f"   Real Rates (TIP/IEF): {_fmt(macro_regime.get('real_rate_ratio'), '{:.3f}')} | Oil: ${_fmt(macro_regime.get('crude_oil_price'))} | {macro_regime.get('yield_curve_label', '10Y–3M spread')}: {_fmt(macro_regime.get('yield_curve_spread'), '{:+.2f}')}% | DXY: {_fmt(macro_regime.get('dxy_index'))} | Credit: {_fmt(macro_regime.get('credit_spread_ratio'), '{:.3f}')}")
  out("=" * 85)

  # ⚡ 30-SECOND TL;DR AT THE VERY TOP OF TERMINAL
  tldr_summary = generate_tldr_summary(results_by_account)
  out("\n" + "━" * 85)
  out("⚡ 30-SECOND TL;DR — WHAT YOU NEED TO DO TODAY (BY ACCOUNT)")
  out("━" * 85)
  for acc_name, t_info in tldr_summary.items():
    acc_plan = results_by_account.get(acc_name, {}).get("rebalancing_plan", {}) or {}
    acc_cfg = raw_accounts_data.get(acc_name, {}).get("cfg", {})
    out(f"\n👤 [{acc_name.upper()}'S ACCOUNT] ➔ {t_info['phase_label']}")
    if acc_plan.get("data_degraded_reason"):
      out(f"   ⚠️ DATA DEGRADED — trades suppressed: {acc_plan['data_degraded_reason']}")
    out(f"   Status: {t_info['phase_desc']}")
    if t_info.get("sells_already_queued"):
      out(f"   ⏳ Sells Already Queued in T212: {', '.join(t_info['sells_already_queued'])}")
    if t_info.get("buys_already_queued"):
      out(f"   🟢 Buys Already Queued in T212: {', '.join(t_info['buys_already_queued'])}")
    if t_info.get("sells_to_place"):
      out("   🔴 Sells to execute:")
      for s in t_info["sells_to_place"]:
        out(f"      • {s['instruction']}")
    if t_info.get("buys_to_place"):
      timing_txt = "Once sell fills settle, enter these buys:" if t_info["phase"] == "AWAIT_FILLS_THEN_BUY" else "Enter these buys today:"
      out(f"   🎯 {timing_txt}")
      for idx, b in enumerate(t_info["buys_to_place"], 1):
        out(f"      {idx}. {b['instruction']}")
    if t_info["phase"] == "ALL_SET":
      if t_info.get("cap_breaches"):
        acc_outside = [
            r for r in (consolidated_view.get("outside_holdings") or [])
            if r.get("account") == acc_name and _num(r.get("vested_gbp"), 0.0) > 0
        ]
        outside_note = (
            " It includes your vested " + ", ".join(sorted({str(r.get("ticker")) for r in acc_outside}))
            + " held outside Trading 212, so trimming that outside holding is the main lever."
            if acc_outside else ""
        )
        out("   ✨ No new orders today. The limit(s) above can't be fixed by trading this account today; avoid adding to that sector." + outside_note)
      else:
        out("   ✨ Within all configured limits. No new orders required today.")
    out(f"   🛡️ {_reserve_status_text(acc_plan, acc_cfg)}")
  out("━" * 85)

  # Consolidated Wealth Overview Banner
  out("\n🌐 CONSOLIDATED WEALTH OVERVIEW (ALL ACCOUNTS COMBINED):")
  out(
      f"  💎 Total Wealth: £{consolidated_view['total_net_worth_gbp']:,.2f} |"
      f" Invested: £{consolidated_view['total_invested_gbp']:,.2f} |"
      f" Liquid Cash: £{consolidated_view['total_liquid_cash_gbp']:,.2f} |"
      f" Pending Blocked: £{consolidated_view['total_blocked_cash_gbp']:,.2f}"
  )
  dod = consolidated_view.get("day_over_day", {})
  if dod and dod.get("has_prior"):
    d_gbp = dod.get("combined_delta_gbp", 0.0)
    d_pct = dod.get("combined_delta_pct", 0.0)
    d_sign = "+" if d_gbp >= 0 else ""
    flows_gbp = _num(dod.get("external_flows_gbp"), 0.0)
    flow_note = (
        f" — excludes {'+' if flows_gbp >= 0 else '-'}£{abs(flows_gbp):,.2f} net deposits/withdrawals"
        if abs(flows_gbp) >= 0.01 else ""
    )
    out(f"  📈 Day-over-Day: {d_sign}£{d_gbp:,.2f} ({d_pct:+.2f}%) vs yesterday ({dod.get('prior_date')}){flow_note}")
  shares_str = " | ".join([
      f"{acc}: £{s['total_value_gbp']:,.0f} ({s['share_of_total_pct']}%)"
      for acc, s in consolidated_view.get("account_shares", {}).items()
  ])
  out(f"  👥 Account Split: {shares_str}")
  clusters_str = " | ".join([
      f"{tb['cluster']}: {tb['weight_pct']}%"
      for tb in consolidated_view.get("thematic_breakdown", [])[:4]
  ])
  out(f"  📊 Global Thematic Allocation: {clusters_str}")
  cons_5y = consolidated_view.get("portfolio_5y_risk_profile", {})
  if cons_5y:
    out(
        f"  🛡️ 5-Year Stress-Test (GBP basis): 5Y Sortino: {_fmt(cons_5y.get('current_sortino_5y'), '{:.2f}')} |"
        f" 5Y Hist. CAGR: {_fmt(cons_5y.get('current_cagr_5y_pct'), '{:+.1f}%')} |"
        f" Portfolio Beta: {_fmt(cons_5y.get('current_beta_5y'), '{:.2f}')} |"
        f" 2022-style Stress: {_fmt(cons_5y.get('current_stress_2022_pct'), '{:+.1f}%')} (£{_num(cons_5y.get('current_stress_2022_gbp'), 0.0):,.0f}) |"
        f" Defensive (cash/money-market/gold): {_fmt(cons_5y.get('post_defensive_pct'), '{:.1f}%')}"
    )
    out(
        f"  🔮 5-Year (2031) scenarios: bear £{_num(cons_5y.get('projected_5y_bear_gbp'), 0.0):,.0f} |"
        f" base £{_num(cons_5y.get('projected_5y_base_gbp'), 0.0):,.0f} |"
        f" bull £{_num(cons_5y.get('projected_5y_bull_gbp'), 0.0):,.0f} (long-run asset-class assumptions, not a forecast) |"
        f" {cons_5y.get('crash_readiness_grade', '')}"
    )
  outside_rows = [r for r in (consolidated_view.get("outside_holdings") or []) if _num(r.get("vested_gbp"), 0.0) > 0 or _num(r.get("unvested_gbp"), 0.0) > 0]
  for r in outside_rows:
    out(
        f"  🏢 Held outside Trading 212 ({r.get('account')}): {r.get('ticker')} vested £{_num(r.get('vested_gbp'), 0.0):,.0f}"
        f" (counted in concentration caps) + unvested £{_num(r.get('unvested_gbp'), 0.0):,.0f} (not counted)"
    )
  out("  🔒 Capital Isolation: Money CANNOT move between accounts. Execute orders within each respective Trading 212 app.")
  out("-" * 85)

  for acc_name, data in results_by_account.items():
    plan = data["rebalancing_plan"]
    blocked_val = plan.get("blocked_cash_gbp", 0.0)
    acc_cfg = raw_accounts_data.get(acc_name, {}).get("cfg", {})
    reserve_floor, reserve_funded, reserve_shortfall = _reserve_figures(plan)
    reserve_label = (
        "funded" if reserve_shortfall <= 0.5
        else f"£{reserve_funded:,.0f} funded, £{reserve_shortfall:,.0f} UNFUNDED"
    )
    out(f"\n👤 Portfolio: {acc_name}")
    if plan.get("data_degraded_reason"):
      out(f"  ⚠️ DATA DEGRADED — trades suppressed: {plan['data_degraded_reason']}")
    out(
        f"  💰 Uninvested Cash: £{plan['available_cash_gbp']:,.2f} |"
        f" ⏳ Pending Committed: £{blocked_val:,.2f} |"
        f" 🛡️ Reserve Floor: £{reserve_floor:,.0f} ({reserve_label}) |"
        f" Reinvesting: £{sum(b['estimated_gbp'] for b in plan['buy_orders']):,.0f} |"
        f" Left in Pocket: £{plan['final_cash_balance_gbp']:,.2f}"
    )
    earmarked_gbp, earmarked_tickers = _earmarked_figures(plan, data.get("analyses"))
    if earmarked_tickers or earmarked_gbp >= 0.5:
      out(
          f"  🔒 Earmarked (excluded from the investable book, never traded): "
          f"{', '.join(earmarked_tickers) or 'n/a'} — £{earmarked_gbp:,.0f}"
      )
    else:
      out("  🔒 Earmarked: none held in this account.")
    acc_5y = plan.get("portfolio_5y_risk_profile", {})
    if acc_5y:
      out(
          f"  🛡️ 5-Year Risk ({acc_5y.get('crash_readiness_grade', '')}):"
          f" 5Y Sortino {_fmt(acc_5y.get('current_sortino_5y'), '{:.2f}')} ➔ {_fmt(acc_5y.get('post_sortino_5y'), '{:.2f}')} |"
          f" 2022-style Stress {_fmt(acc_5y.get('current_stress_2022_pct'), '{:+.1f}%')} ➔ {_fmt(acc_5y.get('post_stress_2022_pct'), '{:+.1f}%')} (£{_num(acc_5y.get('post_stress_2022_gbp'), 0.0):,.0f}) |"
          f" Shield: {_fmt(acc_5y.get('post_shield_pct'), '{:.1f}%')} (defensive cash/MMF/gold: {_fmt(acc_5y.get('post_defensive_pct'), '{:.1f}%')}) |"
          f" 2031 bear/base/bull: £{_num(acc_5y.get('projected_5y_bear_gbp'), 0.0):,.0f} / £{_num(acc_5y.get('projected_5y_base_gbp'), 0.0):,.0f} / £{_num(acc_5y.get('projected_5y_bull_gbp'), 0.0):,.0f}"
      )

    pending_sells_list = plan.get("pending_sell_orders", [])
    pending_buys_dict = plan.get("pending_buys", {})
    if pending_sells_list or pending_buys_dict:
      out(f"\n  ⏳ ORDERS ALREADY PENDING IN TRADING 212 (NO ACTION NEEDED):")
      for pso in pending_sells_list:
        limit_p = pso.get("limit_price", 0.0)
        curr = pso.get("currency", "USD")
        curr_sym = "$" if curr == "USD" else "£"
        limit_str = f" @ limit {curr_sym}{limit_p:,.2f}" if limit_p > 0 else ""
        pso_val = float(pso.get("estimated_gbp", 0.0) or 0.0)
        if math.isnan(pso_val): pso_val = 0.0
        out(f"     • SELL {pso['name']} ({pso['search_symbol']}): ~{pso['shares']:,.2f} shares{limit_str} queued for execution (+£{pso_val:,.0f} cash when filled).")
      for p_ticker, pb in pending_buys_dict.items():
        name = pb.get("name", p_ticker)
        val = pb.get("estimated_value", pb.get("value", 0.0))
        qty = pb.get("quantity", 0.0)
        strat = pb.get("strategy", "QUANTITY")
        if strat == "VALUE" or (val > 0 and qty == 0):
          out(f"     • BUY {name} ({p_ticker}): ~£{val:,.0f} value order queued for execution.")
        else:
          out(f"     • BUY {name} ({p_ticker}): ~{qty:,.2f} shares queued for execution (~£{val:,.0f}).")
    elif blocked_val > 0:
      out(f"\n  ⏳ PENDING CAPITAL DETECTED: £{blocked_val:,.2f} is currently blocked in Trading 212 pending transactions/orders.")

    step = 1
    actionable_sells = [s for s in plan.get("sell_orders", []) if not s.get("is_pending")]
    if not actionable_sells and not plan["buy_orders"] and not pending_sells_list:
      reserve_state = "FUNDED" if reserve_shortfall <= 0.5 else "UNFUNDED"
      out(f"\n  🛡️ STEP {step}: NO NEW ORDERS — CASH RESERVE {reserve_state} (£{reserve_floor:,.0f} floor)")
      out(f"       Status: You have £{plan['available_cash_gbp']:,.2f} in cash (plus £{blocked_val:,.2f} in pending orders). {_reserve_status_text(plan, acc_cfg)}")
      out("       Action: No new orders required today.")
      step += 1

    if actionable_sells:
      all_trims_acc = all("TRIM" in s.get("action", "") for s in actionable_sells)
      header_label = "🟠 STEPS TO TRIM & HARVEST PROFITS (NEW ACTIONS REQUIRED):" if all_trims_acc else "🔴 STEPS TO SELL / TRIM (NEW ACTIONS REQUIRED):"
      out(f"\n  {header_label}")
      for so in actionable_sells:
        action_title = so.get("action", "SELL")
        out(
            f"    {step}. {action_title} {so['name']} ({so['search_symbol']}) ➔"
            f" {so['app_instruction']}"
        )
        out(f"       Why: {so['plain_english_why']}")
        step += 1

    if plan["buy_orders"]:
      out("\n  🟢 STEPS TO BUY (PUT MONEY TO WORK):")
      for bo in plan["buy_orders"]:
        type_tag = "[NEW]" if bo["is_new"] else "[ADD]"
        is_lev = any(k in bo.get("ticker", "") for k in ["QQQ3", "3LUS", "3SEM"])
        lev_cap_pct = _num(REBALANCE_CONFIG.get("max_leveraged_etf_pct"), 0.0) * 100.0
        lev_tag = f" [{lev_cap_pct:.0f}% Max Cap Applied]" if is_lev else ""
        alt_str = f"Altman Z: {bo['altman_z']:.1f}" if bo.get("altman_z") is not None else f"Altman: {bo.get('altman_label', 'N/A')}"
        w_trend = bo.get("weekly_trend_label") or "Weekly trend n/a"
        cagr5 = _num(bo.get("cagr_5y_pct"), None)
        sort5 = _num(bo.get("sortino_5y"), None)
        crash22 = _num(bo.get("crash_2022_ret_pct"), None)
        rr = _num(bo.get("risk_reward_ratio"), None)
        role_lbl = bo.get("crash_role_label") or "n/a"
        cagr5_s = f"{cagr5:+.1f}%" if cagr5 is not None else "n/a"
        sort5_s = f"{sort5:.2f}" if sort5 is not None else "n/a"
        crash22_s = f"{crash22:+.1f}%" if crash22 is not None else "n/a"
        rr_s = f"{rr:.1f}:1" if rr is not None else "n/a"
        out(
            f"    {step}. BUY {type_tag} {bo['name']} ({bo['search_symbol']}){lev_tag} ➔"
            f" {bo['app_instruction']}"
        )
        out(f"       5Y Profile: {role_lbl} | 5Y CAGR (GBP): {cagr5_s} | 5Y Sortino: {sort5_s} | 2022 (GBP): {crash22_s} | {w_trend} | {alt_str} | R:R: {rr_s}")
        out(f"       Why: {bo['plain_english_why']}")
        step += 1

    if plan.get("hold_items"):
      hold_names = ", ".join(
          [f"{h.get('name', h['ticker'])} ({h['ticker']}: £{h.get('value_gbp', 0.0):,.0f})" for h in plan["hold_items"]]
      )
      out(f"\n  🟡 STEP {step}: EVERYTHING ELSE — DO NOTHING (RELAX):")
      out(f"     Companies to leave alone: {hold_names}")
      out(
          f"     Why: No buy or sell signal for these today. Hold them; tomorrow's run re-checks every position."
      )

  out("\n" + "=" * 85 + "\n")

  # Send email if not dry-run
  if not dry_run:
    today_str = run_date
    last_sent_file = os.path.join(os.path.dirname(__file__), "data", "last_email_sent.json")
    already_sent_today = False
    if os.path.exists(last_sent_file) and not force:
      try:
        with open(last_sent_file, "r") as f:
          last_data = json.load(f)
          if last_data.get("date") == today_str:
            already_sent_today = True
            logger.info("Daily digest already sent today (%s at %s). Skipping duplicate send. (Use --force to override).", today_str, last_data.get("time"))
            print(f"\nℹ️ Today's digest was already emailed at {last_data.get('time')}. Skipping duplicate dispatch to guarantee exactly 1 report per day (use --force to override).")
      except Exception as e:
        logger.debug("Could not read last_sent_file: %s", e)

    if not already_sent_today:
      print(f"📧 Dispatching 'What You Need To Do Today' email to: {', '.join(RECIPIENT_EMAILS)}...")
      plain_text_summary = "\n".join(summary_lines)
      sent = send_email_digest(html_report, subject_prefix="⚡ What You Need To Do Today", plain_text_summary=plain_text_summary)
      if sent:
        print(f"✅ 'What You Need To Do Today' email successfully sent to: {', '.join(RECIPIENT_EMAILS)}")
        logger.info("✅ 'What You Need To Do Today' email successfully sent.")
        try:
          with open(last_sent_file, "w") as f:
            json.dump({"date": today_str, "time": datetime.now(LONDON_TZ).strftime("%H:%M:%S")}, f)
        except Exception as e:
          logger.error("Could not write sent-marker %s: %s", last_sent_file, e)

        # Recommendation ledger (evidence base): one record per recommended buy/sell order.
        if not use_mock:
          try:
            ledger_records = _build_ledger_records(results_by_account, today_str)
            added = append_recommendation_records(ledger_records)
            logger.info("Recommendation ledger: %d of %d record(s) appended (model %s).", added, len(ledger_records), MODEL_VERSION)
          except Exception as e:
            logger.error("Could not append to the recommendation ledger: %s", e)
      else:
        print("❌ Email NOT sent (check SMTP credentials in config / env). Exiting with code 1; sent-marker not written so the next tick retries.")
        logger.error("Email dispatch failed; exiting with code 1 without writing the sent-marker.")
        sys.exit(1)
  else:
    print("ℹ️ Dry-run complete. Email dispatch skipped.")
    logger.info("Dry-run complete. Email dispatch skipped.")

  print(f"\n📊 Interactive Web Dashboard available at: file://{os.path.abspath(dashboard_path)}")
  print(f"💡 Run './dashboard.sh' or 'python3 main.py --dashboard' to inspect visually.")

  if open_dashboard:
    import webbrowser
    logger.info("Opening dashboard in web browser: %s", dashboard_path)
    webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")

  if serve_port is not None:
    import http.server
    import socketserver
    print(f"\n🚀 Serving Interactive Dashboard at: http://localhost:{serve_port}/dashboard.html (Press Ctrl+C to stop)")
    os.chdir(os.path.dirname(__file__))
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
      def log_message(self, format, *args):
        pass
    with socketserver.TCPServer(("", serve_port), QuietHandler) as httpd:
      try:
        httpd.serve_forever()
      except KeyboardInterrupt:
        print("\n👋 Dashboard server stopped.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Daily Trading 212 Intelligence Notifier with Deep Research"
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Generate report without sending email",
  )
  parser.add_argument(
      "--mock", action="store_true", help="Use mock portfolio data for testing"
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Force email dispatch even if today's digest was already sent",
  )
  parser.add_argument(
      "--dashboard",
      action="store_true",
      help="Open the interactive web dashboard in default browser after running",
  )
  parser.add_argument(
      "--serve",
      type=int,
      nargs="?",
      const=8080,
      default=None,
      help="Serve the dashboard locally on HTTP (default: port 8080)",
  )
  args = parser.parse_args()

  run_pipeline(
      dry_run=args.dry_run,
      use_mock=args.mock,
      force=args.force,
      open_dashboard=args.dashboard,
      serve_port=args.serve,
  )


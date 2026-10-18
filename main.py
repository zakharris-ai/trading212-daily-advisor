"""Main entry point orchestrating portfolio retrieval, deep research analysis, watchlist scanning, and rebalancing."""

import argparse
from datetime import datetime
import json
import logging
import math
import os
import sys
from typing import Optional
from config import ACCOUNTS, RECIPIENT_EMAILS
from mailer import format_price, generate_html_digest, send_email_digest
from portfolio_storage import (
    compute_day_over_day_delta,
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

# Realistic sample data used for testing and offline verification
DEFAULT_MOCK_PORTFOLIO = {
    "cash": {"free": 2450.00, "total": 18500.00, "ppl": 2150.00},
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
        "cash": {"free": 1100.00, "total": 12400.00, "ppl": 1340.00},
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
      logger.error("No Trading 212 API credentials configured in .env.")
      print("\n⚠️  No Trading 212 API key configured yet!")
      print("👉 Run './setup.sh' to launch the interactive setup wizard, or edit '.env' with your T212_KEY_ID and T212_SECRET.")
      print("💡 Tip: Want to test the engine with sample data first? Run: ./run.sh --dry-run --mock\n")
      return

  logger.info(
      "Starting Deep Research Trading Intelligence & Capital Rebalancing Run..."
  )

  # 1. Macro Climate Research
  research_agent = DeepResearchAgent()
  macro_regime = research_agent.get_macro_regime()
  logger.info(
      "Macro Climate: %s (VIX: %.1f, 10Y: %.2f%%)",
      macro_regime["risk_climate"],
      macro_regime["vix"],
      macro_regime["us10y_yield"],
  )

  gbp_rate = get_gbp_usd_rate()
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
      time.sleep(1.0)
      raw_cash = client.get_account_cash()
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
    logger.error("No Trading 212 API credentials configured in .env.")
    print("\n⚠️  No Trading 212 API key configured yet!")
    print("👉 Run './setup.sh' to launch the interactive setup wizard, or edit '.env' with your T212_KEY_ID and T212_SECRET.")
    print("💡 Tip: Want to test the engine with sample data first? Run: ./run.sh --dry-run --mock\n")
    return

  # Detect and record trade executions since previous snapshot BEFORE auditing positions
  if not use_mock:
    detected_trades = detect_and_record_trades(raw_accounts_data)
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
      if yf_ticker in tracked_entries:
        pos["last_entry_date"] = tracked_entries[yf_ticker]
      elif raw_t in tracked_entries:
        pos["last_entry_date"] = tracked_entries[raw_t]

      if yf_ticker in tracked_trims:
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
  new_opps = scan_watchlist_opportunities(set(), gbp_rate=gbp_rate)

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
    )

    results_by_account[account_name] = {
        "cash": cash_data,
        "pending_orders": pending_orders_data,
        "analyses": analyses,
        "new_opportunities": new_opps,
        "rebalancing_plan": rebalance_plan,
        "consolidated_view": consolidated_view,
    }

  # 5. Compute day-over-day net worth and P&L changes against prior snapshot
  current_acc_totals = {
      acc_name: acc_info["cash"].get("total") or consolidated_view["account_shares"][acc_name]["total_value_gbp"]
      for acc_name, acc_info in raw_accounts_data.items()
  }
  dod_metrics = compute_day_over_day_delta(
      current_combined_total=consolidated_view["total_net_worth_gbp"],
      current_account_totals=current_acc_totals,
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
      f" {datetime.now().strftime('%Y-%m-%d %H:%M')}"
  )
  out(f"🌍 Market Backdrop: {macro_regime.get('risk_climate', '🟢 RISK-ON')} (Macro Score: {macro_regime.get('macro_score', 75)}/100)")
  out(f"🛡️ Crash Defcon: {macro_regime.get('crash_defcon', '🟢 DEFCON 5 — CALM EXPANSION')} (Tail-Crash Prob: {macro_regime.get('crash_probability_pct', 10)}%) | {macro_regime.get('crash_shield_directive', '')}")
  out(f"   VIX: {float(macro_regime.get('vix') or 15.0):.1f} (Term Structure: {float(macro_regime.get('vix_term_structure') or 0.87):.2f}x) | Inst. Distribution: SPY {int(macro_regime.get('spy_dist_days') or 0)}d / QQQ {int(macro_regime.get('qqq_dist_days') or 0)}d ({macro_regime.get('dist_status', 'Low')})")
  out(f"   Real Rates (TIP/IEF): {float(macro_regime.get('real_rate_ratio') or 1.16):.3f} | Oil: ${float(macro_regime.get('crude_oil_price') or 80.0):.1f} | 10Y-2Y: {float(macro_regime.get('yield_curve_spread') or 0.70):+.2f}% | DXY: {float(macro_regime.get('dxy_index') or 100.0):.1f} | Credit: {float(macro_regime.get('credit_spread_ratio') or 0.75):.3f}")
  out("=" * 85)

  # ⚡ 30-SECOND TL;DR AT THE VERY TOP OF TERMINAL
  tldr_summary = generate_tldr_summary(results_by_account)
  out("\n" + "━" * 85)
  out("⚡ 30-SECOND TL;DR — WHAT YOU NEED TO DO TODAY (BY ACCOUNT)")
  out("━" * 85)
  for acc_name, t_info in tldr_summary.items():
    out(f"\n👤 [{acc_name.upper()}'S ACCOUNT] ➔ {t_info['phase_label']}")
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
      out("   ✨ Portfolio balanced & compounding. No new orders required today.")
    out(f"   🛡️ Protected Cash Floor: £{t_info['buffer_reserve']:,.0f} (never touched, safe to withdraw)")
  out("━" * 85)

  # Consolidated Wealth Overview Banner
  active_accounts_label = " & ".join(results_by_account.keys()).upper()
  out(f"\n🌐 PORTFOLIO WEALTH OVERVIEW ({active_accounts_label}):")
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
    out(f"  📈 Day-over-Day: {d_sign}£{d_gbp:,.2f} ({d_pct:+.2f}%) vs yesterday ({dod.get('prior_date')})")
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
        f"  🛡️ 5-Year Horizon Stress-Test & Compounding: 5Y Sortino: {cons_5y.get('current_sortino_5y', 1.4):.2f} |"
        f" 5Y Hist. CAGR: +{cons_5y.get('current_cagr_5y_pct', 20.0):.1f}% |"
        f" Portfolio Beta: {cons_5y.get('current_beta_5y', 1.0):.2f} |"
        f" 2022 Bear Crash Stress: {cons_5y.get('current_stress_2022_pct', -25.0):+.1f}% (£{cons_5y.get('current_stress_2022_gbp', 0.0):,.0f})"
    )
    out(
        f"  🚀 5-Year (2031) Wealth Projection: £{cons_5y.get('projected_5y_conservative_gbp', 0.0):,.0f} (Conservative Risk-Adjusted)"
        f" ➔ £{cons_5y.get('projected_5y_target_gbp', 0.0):,.0f} (Historical 5Y Trajectory) | {cons_5y.get('crash_readiness_grade', '')}"
    )
  out("  🔒 Capital Isolation: Money CANNOT move between accounts. Execute orders within each respective Trading 212 app.")
  out("-" * 85)

  for acc_name, data in results_by_account.items():
    plan = data["rebalancing_plan"]
    blocked_val = plan.get("blocked_cash_gbp", 0.0)
    out(f"\n👤 Portfolio: {acc_name}")
    out(
        f"  💰 Uninvested Cash: £{plan['available_cash_gbp']:,.2f} |"
        f" ⏳ Pending Committed: £{blocked_val:,.2f} |"
        f" 🛡️ Protected Reserve: £{plan['buffer_reserve_gbp']:,.0f} (Safe to withdraw) |"
        f" Reinvesting: £{sum(b['estimated_gbp'] for b in plan['buy_orders']):,.0f} |"
        f" Left in Pocket: £{plan['final_cash_balance_gbp']:,.2f}"
    )
    acc_5y = plan.get("portfolio_5y_risk_profile", {})
    if acc_5y:
      out(
          f"  🛡️ 5-Year Risk & Crash Shield ({acc_5y.get('crash_readiness_grade', '')}):"
          f" 5Y Sortino {acc_5y.get('current_sortino_5y', 1.4):.2f} ➔ {acc_5y.get('post_sortino_5y', 1.5):.2f} |"
          f" 2022 Crash Stress {acc_5y.get('current_stress_2022_pct', -25.0):+.1f}% ➔ {acc_5y.get('post_stress_2022_pct', -20.0):+.1f}% (£{acc_5y.get('post_stress_2022_gbp', 0.0):,.0f}) |"
          f" All-Weather Shield: {acc_5y.get('post_shield_pct', 0.0):.1f}% |"
          f" 2031 Est: £{acc_5y.get('projected_5y_conservative_gbp', 0.0):,.0f}–£{acc_5y.get('projected_5y_target_gbp', 0.0):,.0f}"
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
      out(f"\n  🛡️ STEP {step}: CASH WITHDRAWAL FLOOR SAFE (£{plan['buffer_reserve_gbp']:,.0f})")
      out(f"       Status: You have £{plan['available_cash_gbp']:,.2f} in cash (plus £{blocked_val:,.2f} in pending orders). Your £{plan['buffer_reserve_gbp']:,.0f} floor is 100% ring-fenced & liquid.")
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
        lev_tag = " [8% Max Cap Applied]" if is_lev else ""
        alt_str = f"Altman Z: {bo['altman_z']:.1f}" if bo.get("altman_z") is not None else f"Altman: {bo.get('altman_label', 'N/A')}"
        w_trend = bo.get("weekly_trend_label", "Weekly Stage 2 Confirmed")
        cagr5 = bo.get("cagr_5y_pct", 18.0)
        sort5 = bo.get("sortino_5y", 1.5)
        crash22 = bo.get("crash_2022_ret_pct", -15.0)
        role_lbl = bo.get("crash_role_label", "📈 Core Compounder")
        out(
            f"    {step}. BUY {type_tag} {bo['name']} ({bo['search_symbol']}){lev_tag} ➔"
            f" {bo['app_instruction']}"
        )
        out(f"       5Y Profile: {role_lbl} | 5Y CAGR: {cagr5:+.1f}% | 5Y Sortino: {sort5:.2f} | 2022 Crash: {crash22:+.1f}% | {w_trend} | {alt_str} | R:R: {bo.get('risk_reward_ratio', 2.5)}:1")
        out(f"       Why: {bo['plain_english_why']}")
        step += 1

    if plan.get("hold_items"):
      hold_names = ", ".join(
          [f"{h.get('name', h['ticker'])} ({h['ticker']}: £{h.get('value_gbp', 0.0):,.0f})" for h in plan["hold_items"]]
      )
      out(f"\n  🟡 STEP {step}: EVERYTHING ELSE — DO NOTHING (RELAX):")
      out(f"     Companies to leave alone: {hold_names}")
      out(
          f"     Why: All performing great in upward trends. Sit back and let"
          f" them compound."
      )

  out("\n" + "=" * 85 + "\n")

  # Send email if not dry-run
  if not dry_run:
    today_str = datetime.now().strftime("%Y-%m-%d")
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
            json.dump({"date": today_str, "time": datetime.now().strftime("%H:%M:%S")}, f)
        except Exception:
          pass
      else:
        print("ℹ️ Email not sent (check SMTP credentials in config / env).")
        logger.info("ℹ️ Email not sent (check SMTP credentials in config / env).")
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


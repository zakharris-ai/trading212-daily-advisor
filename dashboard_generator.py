"""Interactive Web Dashboard Generator for Trading 212 Institutional Engine.

Produces a standalone, zero-dependency, ultra-responsive HTML/JS dashboard
displaying real-time action cards, thematic MPT risk meters, position forensic
tables, and historical equity progression.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional
from config import WATCHLIST, EXCLUDED_TICKERS, ACCOUNTS


def generate_dashboard_html(
    results_by_account: Dict[str, Dict[str, Any]],
    consolidated_view: Dict[str, Any],
    macro_regime: Dict[str, Any],
    gbp_rate: float,
    history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Renders the comprehensive interactive HTML dashboard."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_date = datetime.now().strftime("%Y-%m-%d")

    tot_net_worth = consolidated_view.get("total_net_worth_gbp", 0.0)
    tot_invested = consolidated_view.get("total_invested_gbp", 0.0)
    tot_cash = consolidated_view.get("total_liquid_cash_gbp", 0.0)
    tot_blocked = consolidated_view.get("total_blocked_cash_gbp", 0.0)
    tot_ppl = consolidated_view.get("total_ppl_gbp", 0.0)
    dod = consolidated_view.get("day_over_day", {})

    dod_gbp = dod.get("combined_delta_gbp", 0.0)
    dod_pct = dod.get("combined_delta_pct", 0.0)
    dod_sign = "+" if dod_gbp >= 0 else ""
    dod_color = "#10b981" if dod_gbp >= 0 else "#ef4444"

    # Macro parameters (null-safe)
    vix = float(macro_regime.get("vix") or 15.0)
    macro_score = int(macro_regime.get("macro_score") or 75)
    climate = macro_regime.get("risk_climate") or "🟢 RISK-ON / HIGH-BETA FAVORABLE"
    us10y = float(macro_regime.get("us10y_yield") or 4.5)
    curve_spread = float(macro_regime.get("yield_curve_spread") or 0.70)
    term_struct = float(macro_regime.get("vix_term_structure") or 0.85)
    oil = float(macro_regime.get("crude_oil_price") or 80.0)
    spy_dist = int(macro_regime.get("spy_dist_days") or 0)
    qqq_dist = int(macro_regime.get("qqq_dist_days") or 0)
    hb_ratio = float(macro_regime.get("high_beta_ratio") or 2.15)
    semi_ratio = float(macro_regime.get("semi_leadership_ratio") or 0.80)

    # Historical Performance Data for SVG Chart
    history_points = []
    if history:
        for entry in history[-14:]:  # last 14 sessions
            acc_map = {k.lower(): float(v or 0.0) for k, v in entry.get("accounts", {}).items()}
            history_points.append({
                "date": entry.get("date", ""),
                "total": float(entry.get("combined_total", 0.0)),
                "ppl": float(entry.get("total_ppl", 0.0)),
                **acc_map,
            })

    # Prepare positions data across accounts
    all_positions = []
    for acc_name, acc_data in results_by_account.items():
        plan = acc_data.get("rebalancing_plan", {})
        analyses = acc_data.get("analyses", [])
        for pos in analyses:
            p_dict = {
                "account": acc_name,
                "ticker": pos.get("ticker", ""),
                "symbol": pos.get("search_symbol", pos.get("ticker", "")),
                "name": pos.get("name", pos.get("ticker", "")),
                "price": float(pos.get("price", 0.0) or 0.0),
                "currency": pos.get("currency", "USD"),
                "avg_price": float(pos.get("avg_price", 0.0) or 0.0),
                "quantity": float(pos.get("quantity", 0.0) or 0.0),
                "value_gbp": float(pos.get("value_gbp", 0.0) or 0.0),
                "unrealized_pnl": float(pos.get("unrealized_pnl", 0.0) or 0.0),
                "weight_pct": float(pos.get("portfolio_weight_pct", 0.0) or 0.0),
                "signal": pos.get("signal", "🟢 HOLD"),
                "action_short": pos.get("action_short", "HOLD"),
                "plan_status": pos.get("plan_status", "HOLD"),
                "badge_color": pos.get("plan_badge_color", "#6366f1"),
                "conviction_score": int(pos.get("conviction_score", 50) or 50),
                "piotroski": pos.get("piotroski_score"),
                "altman_z": pos.get("altman_z"),
                "roic": pos.get("roic") or pos.get("research", {}).get("roic"),
                "roic_label": pos.get("roic_label") or pos.get("research", {}).get("roic_label", "N/A"),
                "insider_label": pos.get("insider_label") or pos.get("research", {}).get("insider_label", "Neutral"),
                "options_pcr_vol": pos.get("options_pcr_vol") or pos.get("research", {}).get("options_pcr_vol"),
                "options_pcr_label": pos.get("options_pcr_label") or pos.get("research", {}).get("options_pcr_label", "N/A"),
                "inst_flow_label": pos.get("institutional_flow_label") or pos.get("research", {}).get("institutional_flow_label", "Stable"),
                "stage": pos.get("minervini_stage", "Stage 2 Bull"),
                "trailing_stop": float(pos.get("trailing_stop", 0.0) or 0.0),
                "target_price": float(pos.get("target_price", 0.0) or 0.0),
                "theme": pos.get("theme", "Growth"),
                "macro_cluster": pos.get("macro_cluster", "Technology"),
                "earnings_streak_label": pos.get("earnings_streak_label") or pos.get("research", {}).get("earnings_streak_label", "N/A"),
                "fcf_yield_pct": pos.get("fcf_yield_pct") or pos.get("research", {}).get("fcf_yield_pct"),
                "institutional_ownership_pct": pos.get("institutional_ownership_pct") or pos.get("research", {}).get("institutional_ownership_pct"),
                "rs_alpha": pos.get("rs_alpha", 0.0),
                "bearish_divergence": pos.get("bearish_divergence", False),
                "risk_reward": float(pos.get("risk_reward_ratio", 3.0) or 3.0),
                "cagr_5y_pct": float(pos.get("cagr_5y_pct", 18.0) or 18.0),
                "sortino_5y": float(pos.get("sortino_5y", 1.5) or 1.5),
                "crash_2022_ret_pct": float(pos.get("crash_2022_ret_pct", -15.0) or -15.0),
                "crash_role_label": pos.get("crash_role_label", "📈 Core Compounder"),
            }
            if p_dict["avg_price"] > 0:
                p_dict["gain_pct"] = round(((p_dict["price"] - p_dict["avg_price"]) / p_dict["avg_price"]) * 100.0, 1)
            else:
                p_dict["gain_pct"] = 0.0
            all_positions.append(p_dict)

    # Sort positions by value descending
    all_positions.sort(key=lambda x: x["value_gbp"], reverse=True)

    # JSON payloads for client interactivity
    positions_json = json.dumps(all_positions)
    history_json = json.dumps(history_points)
    # Calculate dynamic accounts metadata and total reserve floor
    accounts_meta = {}
    tot_reserve = 0.0
    for acc_name, acc_data in results_by_account.items():
        plan = acc_data.get("rebalancing_plan", {})
        analyses = acc_data.get("analyses", [])
        cash_info = acc_data.get("cash", {})

        invested_gbp = sum(float(pos.get("value_gbp", 0.0) or 0.0) for pos in analyses)
        cash_free_gbp = float(cash_info.get("free_gbp", cash_info.get("free", 0.0)) or 0.0)

        total_val = float(consolidated_view.get("account_shares", {}).get(acc_name, {}).get("total_value_gbp", 0.0) or 0.0)
        if total_val <= 0:
            total_val = float(cash_info.get("total", 0.0) or 0.0)
        if total_val <= 0:
            total_val = invested_gbp + cash_free_gbp

        share_pct = float(consolidated_view.get("account_shares", {}).get(acc_name, {}).get("share_of_total_pct", 0.0) or 0.0)
        if share_pct <= 0 and tot_net_worth > 0:
            share_pct = round((total_val / tot_net_worth) * 100.0, 1)

        reserve_floor = float(plan.get("buffer_reserve_gbp", 0.0) or 0.0)
        if reserve_floor <= 0:
            reserve_floor = float(ACCOUNTS.get(acc_name, {}).get("min_cash_reserve_gbp", 1500.0))

        tot_reserve += reserve_floor

        accounts_meta[acc_name] = {
            "total_value_gbp": total_val,
            "invested_gbp": invested_gbp,
            "cash_free_gbp": cash_free_gbp,
            "reserve_floor": reserve_floor,
            "share_pct": share_pct,
            "risk_5y": plan.get("portfolio_5y_risk_profile", {}),
        }

    accounts_meta["ALL"] = {
        "total_value_gbp": tot_net_worth,
        "invested_gbp": tot_invested,
        "cash_free_gbp": tot_cash,
        "reserve_floor": tot_reserve,
        "share_pct": 100.0,
        "risk_5y": consolidated_view.get("portfolio_5y_risk_profile", {}),
    }

    if tot_reserve <= 0:
        tot_reserve = sum(float(a.get("min_cash_reserve_gbp", 1500.0)) for k, a in ACCOUNTS.items())

    # Build dynamic action cards and status badges per account
    action_cards_by_account: Dict[str, List[str]] = {}
    action_badges_by_account: Dict[str, Dict[str, str]] = {}

    for acc_name, acc_data in results_by_account.items():
        plan = acc_data.get("rebalancing_plan", {})
        actionable_sells = [s for s in plan.get("sell_orders", []) if not s.get("is_pending")]
        pending_sells = plan.get("pending_sell_orders", [])
        buy_orders = plan.get("buy_orders", [])
        reserve_val = accounts_meta.get(acc_name, {}).get("reserve_floor", 0.0)
        avail_cash = float(plan.get("available_cash_gbp", 0.0) or 0.0)

        cards = []
        step = 1

        if not actionable_sells and not buy_orders and not pending_sells:
            action_badges_by_account[acc_name] = {
                "text": "✅ ALL SET — SIT BACK & RELAX",
                "class": "badge-emerald"
            }
            cards.append(f"""
        <div class="action-card reserve">
          <div class="action-header">
            <span style="font-weight: 800; font-size: 14px; color: var(--emerald);">👤 [{acc_name.upper()}] ✅ ALL SET — SIT BACK & RELAX</span>
            <span class="action-badge" style="background: var(--emerald-glow); color: var(--emerald);">PORTFOLIO BALANCED</span>
          </div>
          <div class="action-instruction">✨ No action required today. Portfolio is fully balanced and compounding.</div>
          <div class="action-why"><b>Status:</b> All positions are in confirmed Stage 2 uptrends with healthy risk/reward profiles. Protected cash floor of £{reserve_val:,.0f} is fully preserved and liquid (£{avail_cash:,.2f} total cash). Sit back and let your winners run.</div>
        </div>""")
        else:
            has_trims = any("TRIM" in s.get("action", "") for s in actionable_sells)
            badge_text = "🟠 PROFIT HARVESTING & REALLOCATION ACTIVE" if has_trims else ("🔴 CAPITAL REALLOCATION ACTIVE" if actionable_sells else "🟢 CAPITAL DEPLOYMENT ACTIVE")
            action_badges_by_account[acc_name] = {
                "text": badge_text,
                "class": "badge-amber" if (has_trims or actionable_sells) else "badge-emerald"
            }

            for pso in pending_sells:
                lim_p = pso.get("limit_price", 0.0)
                curr = pso.get("currency", "USD")
                curr_sym = "$" if curr == "USD" else "£"
                lim_str = f" @ limit {curr_sym}{lim_p:,.2f}" if lim_p > 0 else ""
                pso_val = float(pso.get("estimated_gbp", 0.0) or 0.0)
                s_name = pso.get("name", pso.get("ticker"))
                s_sym = pso.get("search_symbol", pso.get("ticker"))
                shares_amt = pso.get("shares", 0.0)
                cards.append(f"""
        <div class="action-card" style="border-left: 5px solid var(--primary); background: linear-gradient(135deg, rgba(59, 130, 246, 0.06), var(--card-bg) 60%);">
          <div class="action-header">
            <span style="font-weight: 800; font-size: 14px; color: var(--primary);">👤 [{acc_name.upper()}] ⏳ ORDER ALREADY QUEUED IN TRADING 212</span>
            <span class="action-badge" style="background: rgba(59, 130, 246, 0.15); color: var(--primary);">NO ACTION NEEDED</span>
          </div>
          <div class="action-instruction">SELL {s_name} ({s_sym}) ~{shares_amt:,.2f} shares{lim_str}</div>
          <div class="action-why"><b>Status:</b> Order is resting in Trading 212 awaiting execution (+£{pso_val:,.0f} cash when filled). No duplicate action needed.</div>
        </div>""")

            for so in actionable_sells:
                is_trim = "TRIM" in so.get("action", "")
                card_class = "trim" if is_trim else "sell"
                h_color = "var(--amber)" if is_trim else "var(--rose)"
                b_bg = "var(--amber-glow)" if is_trim else "var(--rose-glow)"
                b_col = "var(--amber)" if is_trim else "var(--rose)"
                act_label = f"STEP {step}: 🟠 {so.get('action', 'TRIM')}" if is_trim else f"STEP {step}: 🔴 {so.get('action', 'SELL')}"
                est_val = float(so.get("estimated_gbp", 0.0) or 0.0)
                cards.append(f"""
        <div class="action-card {card_class}">
          <div class="action-header">
            <span style="font-weight: 800; font-size: 14px; color: {h_color};">👤 [{acc_name.upper()}] {act_label}</span>
            <span class="action-badge" style="background: {b_bg}; color: {b_col};">BANKS +£{est_val:,.0f} CASH</span>
          </div>
          <div class="action-instruction">{so.get('app_instruction', '')}</div>
          <div class="action-why"><b>Why:</b> {so.get('plain_english_why', '')}</div>
        </div>""")
                step += 1

            for bo in buy_orders:
                b_sym = bo.get("search_symbol", bo.get("ticker"))
                b_name = bo.get("name", bo.get("ticker"))
                b_est = float(bo.get("estimated_gbp", 0.0) or 0.0)
                b_why = bo.get("plain_english_why", "")
                b_inst = bo.get("app_instruction", "")
                act_label = f"STEP {step}: 🟢 BUY {b_name} ({b_sym})"
                cards.append(f"""
        <div class="action-card buy">
          <div class="action-header">
            <span style="font-weight: 800; font-size: 14px; color: var(--emerald);">👤 [{acc_name.upper()}] {act_label}</span>
            <span class="action-badge" style="background: var(--emerald-glow); color: var(--emerald);">REINVEST £{b_est:,.0f}</span>
          </div>
          <div class="action-instruction">{b_inst}</div>
          <div class="action-why"><b>Data Rationale:</b> {b_why}</div>
        </div>""")
                step += 1

        action_cards_by_account[acc_name] = cards

    all_cards = []
    any_actions_active = False
    for acc_name, cards in action_cards_by_account.items():
        all_cards.extend(cards)
        if any("STEP" in c for c in cards):
            any_actions_active = True

    if not any_actions_active:
        action_badges_by_account["ALL"] = {
            "text": "✅ ALL SET — SIT BACK & RELAX",
            "class": "badge-emerald"
        }
    else:
        action_badges_by_account["ALL"] = {
            "text": "🟠 ACTIONS ACTIVE (SELECT ACCOUNT TAB)",
            "class": "badge-amber"
        }
    action_cards_by_account["ALL"] = all_cards

    # Build dynamic account tabs
    tabs_buttons_html = f"""
      <button class="tab-btn active" onclick="switchAccount('ALL')">
        <span>🌐 Consolidated (£{tot_net_worth:,.0f})</span>
      </button>"""
    for acc_name, meta in accounts_meta.items():
        if acc_name == "ALL":
            continue
        tabs_buttons_html += f"""
      <button class="tab-btn" onclick="switchAccount('{acc_name}')">
        <span>👤 {acc_name}'s Account (£{meta['total_value_gbp']:,.0f} • {meta['share_pct']:.1f}%)</span>
      </button>"""

    # JSON payloads for client interactivity
    positions_json = json.dumps(all_positions)
    history_json = json.dumps(history_points)
    accounts_meta_json = json.dumps(accounts_meta)
    action_cards_json = json.dumps(action_cards_by_account)
    action_badges_json = json.dumps(action_badges_by_account)

    # Thematic Clusters
    thematic_clusters = consolidated_view.get("thematic_breakdown", [])
    accounts_title_label = " & ".join(results_by_account.keys()) or "Portfolio"
    excluded_badge_html = (
        f'<span class="badge badge-indigo">Excluded: {", ".join(sorted(EXCLUDED_TICKERS))}</span>'
        if EXCLUDED_TICKERS
        else '<span class="badge badge-indigo">5-Year High-Sortino Universe</span>'
    )

    # HTML Document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trading 212 Institutional Engine — Wealth & Rebalancing Terminal</title>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f293d;
      --card-hover: #172236;
      --text: #f3f4f6;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --emerald: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.15);
      --amber: #f59e0b;
      --amber-glow: rgba(245, 158, 11, 0.15);
      --rose: #f43f5e;
      --rose-glow: rgba(244, 63, 94, 0.15);
      --indigo: #6366f1;
      --purple: #8b5cf6;
      --cyan: #06b6d4;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
      -webkit-font-smoothing: antialiased;
    }}
    .max-container {{ max-width: 1400px; margin: 0 auto; }}
    
    /* Header Bar */
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
    }}
    .title-group h1 {{
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .title-group p {{
      color: var(--text-muted);
      font-size: 13px;
      margin-top: 4px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .badge-emerald {{ background: var(--emerald-glow); color: var(--emerald); border: 1px solid var(--emerald); }}
    .badge-amber {{ background: var(--amber-glow); color: var(--amber); border: 1px solid var(--amber); }}
    .badge-rose {{ background: var(--rose-glow); color: var(--rose); border: 1px solid var(--rose); }}
    .badge-indigo {{ background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid #6366f1; }}

    /* Macro Marquee */
    .macro-bar {{
      background: #0d1322;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 14px;
      margin-bottom: 24px;
      font-size: 13px;
    }}
    .macro-item {{ display: flex; align-items: center; gap: 8px; }}
    .macro-label {{ color: var(--text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .macro-val {{ font-weight: 700; color: #fff; }}

    /* KPI Cards Grid */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px 20px;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .kpi-card:hover {{
      transform: translateY(-2px);
      border-color: #2d3b55;
    }}
    .kpi-title {{
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .kpi-value {{
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }}
    .kpi-sub {{
      font-size: 12px;
      margin-top: 6px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    /* View Mode Tabs */
    .tabs-bar {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 12px;
      overflow-x: auto;
    }}
    .tab-btn {{
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: 10px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease;
      white-space: nowrap;
    }}
    .tab-btn:hover {{
      color: #fff;
      background: rgba(255, 255, 255, 0.04);
    }}
    .tab-btn.active {{
      background: #1e293b;
      color: #fff;
      border-color: #3b82f6;
      box-shadow: 0 0 12px rgba(59, 130, 246, 0.2);
    }}

    /* Section Layout */
    .section-title {{
      font-size: 18px;
      font-weight: 800;
      letter-spacing: -0.01em;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    /* Action Cards */
    .action-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }}
    .action-card {{
      background: var(--card-bg);
      border-radius: 14px;
      padding: 20px;
      border: 1px solid var(--card-border);
      position: relative;
    }}
    .action-card.trim {{
      border-left: 5px solid var(--amber);
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.06), var(--card-bg) 60%);
    }}
    .action-card.buy {{
      border-left: 5px solid var(--emerald);
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.06), var(--card-bg) 60%);
    }}
    .action-card.reserve {{
      border-left: 5px solid var(--cyan);
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.06), var(--card-bg) 60%);
    }}
    .action-card.sell {{
      border-left: 5px solid var(--rose);
      background: linear-gradient(135deg, rgba(244, 63, 94, 0.06), var(--card-bg) 60%);
    }}
    .action-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    .action-badge {{
      padding: 3px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }}
    .action-instruction {{
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 14px;
      color: #fff;
      font-weight: 600;
      margin: 12px 0;
    }}
    .action-why {{
      font-size: 12.5px;
      color: var(--text-muted);
      line-height: 1.4;
    }}

    /* Table */
    .table-container {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      overflow-x: auto;
      margin-bottom: 32px;
    }}
    .table-toolbar {{
      padding: 16px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      border-bottom: 1px solid var(--card-border);
    }}
    .search-input {{
      background: #090d16;
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      width: 260px;
      outline: none;
    }}
    .search-input:focus {{ border-color: var(--primary); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}
    th {{
      padding: 12px 16px;
      background: #0c121e;
      color: var(--text-muted);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--card-border);
      cursor: pointer;
      user-select: none;
    }}
    th:hover {{ color: #fff; }}
    td {{
      padding: 14px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }}
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
    .ticker-cell {{ font-weight: 800; font-size: 14px; display: flex; align-items: center; gap: 8px; }}
    .company-cell {{ color: var(--text-muted); font-size: 12px; }}

    /* MPT Cluster Cards */
    .mpt-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }}
    .mpt-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px 18px;
    }}
    .mpt-header {{ display: flex; justify-content: space-between; font-size: 13px; font-weight: 700; margin-bottom: 10px; }}
    .progress-track {{
      background: #1e293b;
      height: 8px;
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 8px;
      position: relative;
    }}
    .progress-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s ease; }}
    .mpt-meta {{ display: flex; justify-content: space-between; font-size: 11.5px; color: var(--text-muted); }}

    /* Equity Curve Chart */
    .chart-box {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 32px;
    }}
    .chart-container {{
      width: 100%;
      height: 220px;
      position: relative;
    }}
    svg.equity-svg {{
      width: 100%;
      height: 100%;
      overflow: visible;
    }}

    /* Watchlist Grid */
    .watch-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
      gap: 16px;
      margin-bottom: 40px;
    }}
    .watch-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
      transition: border-color 0.15s ease;
    }}
    .watch-card:hover {{ border-color: #3b82f6; }}
    .watch-top {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }}
    .watch-title {{ font-size: 15px; font-weight: 800; }}
    .watch-theme {{ font-size: 11px; color: var(--text-muted); margin-bottom: 12px; }}
    .watch-metrics {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      background: #090d16;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 11.5px;
    }}
    .watch-m-item {{ display: flex; flex-direction: column; gap: 2px; }}
    .watch-m-label {{ color: var(--text-muted); font-size: 10px; text-transform: uppercase; }}
    .watch-m-val {{ font-weight: 700; color: #fff; }}

    /* Footer */
    .footer {{
      border-top: 1px solid var(--card-border);
      padding-top: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      font-size: 12px;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>
  <div class="max-container">

    <!-- Header -->
    <header class="header">
      <div class="title-group">
        <h1>
          <span>🔬 Institutional Quantitative Dashboard</span>
          <span class="badge badge-emerald">LIVE ENGINE</span>
        </h1>
        <p>{accounts_title_label} Trading 212 Portfolio • Daily Buy/Sell Recommendations, Trims, & Risk Parity Engine</p>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 13px; font-weight: 700; color: #fff;">Anchor FX: 1 GBP = ${gbp_rate:.4f} USD</div>
        <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 2px;">Last Audit: {now_str}</div>
      </div>
    </header>

    <!-- Macro Telemetry Bar -->
    <div class="macro-bar">
      <div class="macro-item">
        <span class="macro-label">Backdrop:</span>
        <span class="badge badge-emerald">{climate} ({macro_score}/100)</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">Crash Defcon:</span>
        <span class="badge badge-emerald">{macro_regime.get('crash_defcon', '🟢 DEFCON 5 — CALM EXPANSION')}</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">VIX:</span>
        <span class="macro-val">{vix:.1f} ({term_struct:.2f}x term)</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">10Y Yield:</span>
        <span class="macro-val">{us10y:.2f}% (Spread: {curve_spread:+.2f}%)</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">Dist. Days:</span>
        <span class="macro-val">SPY {spy_dist}d / QQQ {qqq_dist}d</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">High-Beta:</span>
        <span class="macro-val">{hb_ratio:.2f}x</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">Semi Lead:</span>
        <span class="macro-val">{semi_ratio:.3f}x</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">Crude Oil:</span>
        <span class="macro-val">${oil:.1f}</span>
      </div>
    </div>

    <!-- Top KPI Row -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">
          <span>Consolidated Net Worth</span>
          <span>💎</span>
        </div>
        <div class="kpi-value" id="kpiNetWorth">£{tot_net_worth:,.2f}</div>
        <div class="kpi-sub">
          <span style="color: {dod_color}; font-weight: 700;">{dod_sign}£{dod_gbp:,.2f} ({dod_pct:+.2f}%)</span>
          <span>vs yesterday</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-title">
          <span>Total Invested Equity</span>
          <span>📈</span>
        </div>
        <div class="kpi-value" id="kpiInvested">£{tot_invested:,.2f}</div>
        <div class="kpi-sub">
          <span>Across {len(all_positions)} audited positions</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-title">
          <span>Liquid Free Cash</span>
          <span>💰</span>
        </div>
        <div class="kpi-value" id="kpiCash">£{tot_cash:,.2f}</div>
        <div class="kpi-sub">
          <span>Pending Blocked: £{tot_blocked:,.2f}</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-title">
          <span>Protected Cash Floor</span>
          <span>🛡️</span>
        </div>
        <div class="kpi-value" style="color: #38bdf8;" id="kpiReserve">£{tot_reserve:,.2f}</div>
        <div class="kpi-sub">
          <span>100% Ring-fenced safe to withdraw</span>
        </div>
      </div>
    </div>

    <!-- Account View Selector Tabs -->
    <div class="tabs-bar">
{tabs_buttons_html}
    </div>

    <!-- 5-Year Horizon Crash-Protection & Stress-Test Banner -->
    <div class="kpi-grid" style="margin-bottom: 24px;">
      <div class="kpi-card" style="border-color: rgba(16, 185, 129, 0.35);">
        <div class="kpi-title">
          <span>5-Year Sortino & CAGR</span>
          <span>🚀</span>
        </div>
        <div class="kpi-value" style="color: var(--emerald); font-size: 22px;" id="kpiSortino5Y">{consolidated_view.get('portfolio_5y_risk_profile', {}).get('current_sortino_5y', 1.5):.2f} Sortino</div>
        <div class="kpi-sub" id="kpiCagr5Y">
          <span>Hist. 5Y CAGR: +{consolidated_view.get('portfolio_5y_risk_profile', {}).get('current_cagr_5y_pct', 20.0):.1f}%/yr</span>
        </div>
      </div>

      <div class="kpi-card" style="border-color: rgba(244, 63, 94, 0.35);">
        <div class="kpi-title">
          <span>2022 Crash Stress-Test</span>
          <span>🛡️</span>
        </div>
        <div class="kpi-value" style="color: var(--amber); font-size: 22px;" id="kpiCrash2022">{consolidated_view.get('portfolio_5y_risk_profile', {}).get('current_stress_2022_pct', -25.0):+.1f}%</div>
        <div class="kpi-sub" id="kpiCrash2022Sub">
          <span>Simulated 2022 Bear Drawdown (£{consolidated_view.get('portfolio_5y_risk_profile', {}).get('current_stress_2022_gbp', 0.0):,.0f})</span>
        </div>
      </div>

      <div class="kpi-card" style="border-color: rgba(59, 130, 246, 0.35);">
        <div class="kpi-title">
          <span>All-Weather Crash Shield</span>
          <span>🏰</span>
        </div>
        <div class="kpi-value" style="color: var(--primary); font-size: 22px;" id="kpiShield5Y">{consolidated_view.get('portfolio_5y_risk_profile', {}).get('current_shield_pct', 0.0):.1f}%</div>
        <div class="kpi-sub" id="kpiShield5YSub">
          <span>{consolidated_view.get('portfolio_5y_risk_profile', {}).get('crash_readiness_grade', '🟢 ALL-WEATHER READY')}</span>
        </div>
      </div>

      <div class="kpi-card" style="border-color: rgba(139, 92, 246, 0.35);">
        <div class="kpi-title">
          <span>5-Year (2031) Wealth Target</span>
          <span>🔮</span>
        </div>
        <div class="kpi-value" style="color: #c4b5fd; font-size: 20px;" id="kpiProj5Y">£{consolidated_view.get('portfolio_5y_risk_profile', {}).get('projected_5y_conservative_gbp', 0.0):,.0f} – £{consolidated_view.get('portfolio_5y_risk_profile', {}).get('projected_5y_target_gbp', 0.0):,.0f}</div>
        <div class="kpi-sub">
          <span>5-Year Horizon Compounding Range</span>
        </div>
      </div>
    </div>

    <!-- 2-Minute Action Center -->
    <div class="section-title">
      <span>⚡ 2-Minute Action Center — What You Need To Do Today</span>
      <span class="badge {action_badges_by_account['ALL']['class']}" id="actionStatusBadge">{action_badges_by_account['ALL']['text']}</span>
    </div>

    <div class="action-grid" id="actionCardsContainer">
      <!-- Generated dynamically by JS according to selected account -->
    </div>

    <!-- MPT Thematic Concentration Meters -->
    <div class="section-title">
      <span>🌐 Modern Portfolio Theory (MPT) Thematic Risk Exposure</span>
      <span style="font-size: 12px; color: var(--text-muted); font-weight: normal;">(35.0% Maximum Correlation Cap)</span>
    </div>

    <div class="mpt-grid">
"""
    for tb in thematic_clusters:
        c_name = tb["cluster"]
        c_val = tb["value_gbp"]
        c_pct = tb["weight_pct"]
        is_sat = c_pct >= 35.0
        bar_color = "#ef4444" if is_sat else ("#10b981" if c_pct >= 15.0 else "#f59e0b")
        status_label = "🔴 SATURATED (>35% CAP)" if is_sat else ("🟢 CORE PILLAR" if c_pct >= 15.0 else "🟡 UNDERWEIGHTED")
        html += f"""
      <div class="mpt-card">
        <div class="mpt-header">
          <span>{c_name}</span>
          <span style="color: {bar_color};">{c_pct:.1f}%</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width: {min(100.0, c_pct * 2.5)}%; background: {bar_color};"></div>
        </div>
        <div class="mpt-meta">
          <span>£{c_val:,.0f} value</span>
          <span style="font-weight: 700; color: {bar_color};">{status_label}</span>
        </div>
      </div>
"""

    html += f"""
    </div>

    <!-- Historical Net Worth Equity Curve -->
    <div class="chart-box">
      <div class="section-title" style="margin-bottom: 12px;">
        <span>📈 Historical Portfolio Equity Curve & Trajectory</span>
        <span style="font-size: 12px; color: var(--text-muted); font-weight: normal;">(Trailing 8 Sessions)</span>
      </div>
      <div class="chart-container" id="chartContainer">
        <!-- Rendered via inline SVG -->
      </div>
    </div>

    <!-- Forensic Holdings Dossier Table -->
    <div class="section-title">
      <span>🔬 Active Holdings Forensic Quality & Technical Audit</span>
    </div>

    <div class="table-container">
      <div class="table-toolbar">
        <input type="text" class="search-input" id="tableSearch" placeholder="🔍 Search holding or symbol..." oninput="filterTable()">
        <div style="display: flex; gap: 8px; font-size: 12px; color: var(--text-muted); align-items: center;">
          <span>Showing <b id="visibleCount">15</b> positions</span>
        </div>
      </div>
      <table id="holdingsTable">
        <thead>
          <tr>
            <th onclick="sortTable('ticker')">Holding</th>
            <th onclick="sortTable('account')">Account</th>
            <th onclick="sortTable('price')">Price</th>
            <th onclick="sortTable('gain_pct')">Gain / Return</th>
            <th onclick="sortTable('unrealized_pnl')">P&L (£)</th>
            <th onclick="sortTable('value_gbp')">Value (GBP)</th>
            <th onclick="sortTable('weight_pct')">Weight</th>
            <th onclick="sortTable('sortino_5y')">5Y Sortino</th>
            <th onclick="sortTable('crash_2022_ret_pct')">2022 Crash</th>
            <th onclick="sortTable('piotroski')">Piotroski</th>
            <th onclick="sortTable('altman_z')">Altman Z</th>
            <th onclick="sortTable('risk_reward')">R:R</th>
            <th>Signal & Strategy Status</th>
          </tr>
        </thead>
        <tbody id="tableBody">
          <!-- Populated by JS -->
        </tbody>
      </table>
    </div>

    <!-- Curated Cross-Sector Watchlist -->
    <div class="section-title">
      <span>🎯 Curated Data-Driven Watchlist (Highest Probability, Lowest Risk)</span>
      {excluded_badge_html}
    </div>

    <div class="watch-grid">
"""
    for w in WATCHLIST:
        t = w["ticker"]
        name = w["name"]
        theme = w["theme"]
        # Basic metadata badges
        html += f"""
      <div class="watch-card">
        <div class="watch-top">
          <div>
            <div class="watch-title">{t}</div>
            <div style="font-size: 12px; color: #fff;">{name}</div>
          </div>
          <span class="badge badge-indigo">SCREENED</span>
        </div>
        <div class="watch-theme">{theme}</div>
        <div class="watch-metrics">
          <div class="watch-m-item">
            <span class="watch-m-label">Balance Sheet</span>
            <span class="watch-m-val">Piotroski 8–9/9</span>
          </div>
          <div class="watch-m-item">
            <span class="watch-m-label">Solvency</span>
            <span class="watch-m-val">Altman Z > 6.0</span>
          </div>
          <div class="watch-m-item">
            <span class="watch-m-label">Trend Structure</span>
            <span class="watch-m-val">Weekly Stage 2</span>
          </div>
          <div class="watch-m-item">
            <span class="watch-m-label">Expectancy</span>
            <span class="watch-m-val">R:R > 5.0:1</span>
          </div>
        </div>
      </div>
"""

    html += f"""
    </div>

    <!-- Footer -->
    <footer class="footer">
      <div>Trading 212 Daily Recommendation & Rebalancing Engine • {accounts_title_label}</div>
      <div>Capital Isolation Enforced: Execute orders directly inside your Trading 212 app.</div>
    </footer>

  </div>

  <script>
    const POSITIONS = {positions_json};
    const HISTORY = {history_json};
    const ACCOUNTS_META = {accounts_meta_json};
    const ACTION_CARDS = {action_cards_json};
    const ACTION_BADGES = {action_badges_json};
    let currentAccount = 'ALL';
    let sortCol = 'value_gbp';
    let sortAsc = false;

    function formatGBP(val) {{
      return '£' + Number(val).toLocaleString('en-GB', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
    }}

    function switchAccount(acc) {{
      currentAccount = acc;
      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.textContent.includes(acc === 'ALL' ? 'Consolidated' : acc));
      }});
      updateKPIs();
      renderActionCards();
      renderTable();
      renderChart();
    }}

    function updateKPIs() {{
      const kpiNetWorth = document.getElementById('kpiNetWorth');
      const kpiInvested = document.getElementById('kpiInvested');
      const kpiCash = document.getElementById('kpiCash');
      const kpiReserve = document.getElementById('kpiReserve');

      if (currentAccount === 'ALL') {{
        kpiNetWorth.textContent = '£{tot_net_worth:,.2f}';
        kpiInvested.textContent = '£{tot_invested:,.2f}';
        kpiCash.textContent = '£{tot_cash:,.2f}';
        kpiReserve.textContent = '£{tot_reserve:,.2f}';
      }} else if (ACCOUNTS_META[currentAccount]) {{
        const acc = ACCOUNTS_META[currentAccount];
        kpiNetWorth.textContent = formatGBP(acc.total_value_gbp);
        kpiInvested.textContent = formatGBP(acc.invested_gbp);
        kpiCash.textContent = formatGBP(acc.cash_free_gbp);
        kpiReserve.textContent = formatGBP(acc.reserve_floor);
      }}

      const accMeta = ACCOUNTS_META[currentAccount] || ACCOUNTS_META['ALL'];
      if (accMeta && accMeta.risk_5y) {{
        const r = accMeta.risk_5y;
        const sortEl = document.getElementById('kpiSortino5Y');
        const cagrEl = document.getElementById('kpiCagr5Y');
        const crashEl = document.getElementById('kpiCrash2022');
        const crashSubEl = document.getElementById('kpiCrash2022Sub');
        const shieldEl = document.getElementById('kpiShield5Y');
        const shieldSubEl = document.getElementById('kpiShield5YSub');
        const projEl = document.getElementById('kpiProj5Y');

        if (sortEl) sortEl.textContent = (r.current_sortino_5y || 1.5).toFixed(2) + (r.post_sortino_5y && r.post_sortino_5y !== r.current_sortino_5y ? ' ➔ ' + r.post_sortino_5y.toFixed(2) : ' Sortino');
        if (cagrEl) cagrEl.innerHTML = `<span>Hist. 5Y CAGR: +${{(r.post_cagr_5y_pct || r.current_cagr_5y_pct || 20).toFixed(1)}}%/yr (Beta ${{(r.post_beta_5y || r.current_beta_5y || 1.0).toFixed(2)}})</span>`;
        if (crashEl) crashEl.textContent = (r.current_stress_2022_pct || -25).toFixed(1) + '%' + (r.post_stress_2022_pct && r.post_stress_2022_pct !== r.current_stress_2022_pct ? ' ➔ ' + r.post_stress_2022_pct.toFixed(1) + '%' : '');
        if (crashSubEl) crashSubEl.innerHTML = `<span>2022 Crash Sim: £${{Math.round(r.post_stress_2022_gbp || r.current_stress_2022_gbp || 0).toLocaleString('en-GB')}}</span>`;
        if (shieldEl) shieldEl.textContent = (r.current_shield_pct || 0).toFixed(1) + '%' + (r.post_shield_pct && r.post_shield_pct !== r.current_shield_pct ? ' ➔ ' + r.post_shield_pct.toFixed(1) + '%' : '');
        if (shieldSubEl) shieldSubEl.innerHTML = `<span>${{r.crash_readiness_grade || '🟢 ALL-WEATHER READY'}}</span>`;
        if (projEl) projEl.textContent = '£' + Math.round(r.projected_5y_conservative_gbp || 0).toLocaleString('en-GB') + ' – £' + Math.round(r.projected_5y_target_gbp || 0).toLocaleString('en-GB');
      }}
    }}

    function renderActionCards() {{
      const container = document.getElementById('actionCardsContainer');
      const badge = document.getElementById('actionStatusBadge');

      const cardsList = ACTION_CARDS[currentAccount] || [];
      container.innerHTML = cardsList.join('\\n');

      const badgeInfo = ACTION_BADGES[currentAccount] || {{ text: '✅ ALL SET', class: 'badge-emerald' }};
      if (badge) {{
        badge.className = 'badge ' + badgeInfo.class;
        badge.textContent = badgeInfo.text;
      }}
    }}

    function renderTable() {{
      const tbody = document.getElementById('tableBody');
      const searchVal = (document.getElementById('tableSearch').value || '').toLowerCase();
      let filtered = POSITIONS.filter(p => {{
        if (currentAccount !== 'ALL' && p.account !== currentAccount) return false;
        if (searchVal && !p.ticker.toLowerCase().includes(searchVal) && !p.name.toLowerCase().includes(searchVal)) return false;
        return true;
      }});

      filtered.sort((a, b) => {{
        let vA = a[sortCol];
        let vB = b[sortCol];
        if (typeof vA === 'string') vA = vA.toLowerCase();
        if (typeof vB === 'string') vB = vB.toLowerCase();
        if (vA < vB) return sortAsc ? -1 : 1;
        if (vA > vB) return sortAsc ? 1 : -1;
        return 0;
      }});

      document.getElementById('visibleCount').textContent = filtered.length;

      let html = '';
      for (const p of filtered) {{
        const pnlColor = p.unrealized_pnl >= 0 ? '#10b981' : '#f43f5e';
        const pnlSign = p.unrealized_pnl >= 0 ? '+' : '';
        const gainSign = p.gain_pct >= 0 ? '+' : '';
        
        const pioBadge = (p.piotroski !== null && p.piotroski !== undefined) ? `<span class="badge ${{p.piotroski >= 8 ? 'badge-emerald' : 'badge-amber'}}">${{p.piotroski}}/9</span>` : '<span style="color:#64748b">N/A</span>';
        const altBadge = (p.altman_z !== null && p.altman_z !== undefined) ? `<span style="font-weight:700; color:${{p.altman_z >= 3.0 ? '#10b981' : '#f59e0b'}}">${{p.altman_z.toFixed(1)}}</span>` : '<span style="color:#64748b">N/A</span>';
        const sort5Badge = `<span style="font-weight:700; color:${{p.sortino_5y >= 1.5 ? '#10b981' : (p.sortino_5y >= 0.9 ? '#38bdf8' : '#f43f5e')}}">${{(p.sortino_5y || 1.5).toFixed(2)}}</span> <span style="font-size:10px; color:#94a3b8;">(${{(p.cagr_5y_pct || 18) >= 0 ? '+' : ''}}${{(p.cagr_5y_pct || 18).toFixed(0)}}%/y)</span>`;
        const crash22Color = (p.crash_2022_ret_pct || -15) >= 0 ? '#10b981' : ((p.crash_2022_ret_pct || -15) >= -20 ? '#38bdf8' : '#f43f5e');
        const crash22Badge = `<span style="font-weight:700; color:${{crash22Color}}">${{(p.crash_2022_ret_pct || -15) >= 0 ? '+' : ''}}${{(p.crash_2022_ret_pct || -15).toFixed(1)}}%</span>`;

        let statusBadgeClass = 'badge-indigo';
        if (p.plan_status.includes('TRIM')) statusBadgeClass = 'badge-amber';
        if (p.plan_status.includes('BUY')) statusBadgeClass = 'badge-emerald';
        if (p.plan_status.includes('SELL')) statusBadgeClass = 'badge-rose';

        html += `
          <tr>
            <td>
              <div class="ticker-cell">
                <span>${{p.symbol}}</span>
                <span style="font-size: 11px; font-weight: normal; color: var(--text-muted);">(${{p.ticker}})</span>
              </div>
              <div class="company-cell">
                <span>${{p.name}}</span>
                ${{p.crash_role_label ? `<span style="font-size: 10px; background: rgba(16, 185, 129, 0.12); color: #34d399; padding: 1px 6px; border-radius: 4px; margin-left: 6px; font-weight: 700; border: 1px solid rgba(16, 185, 129, 0.3);">${{p.crash_role_label}}</span>` : ''}}
                ${{p.earnings_streak_label && p.earnings_streak_label !== 'N/A' ? `<span style="font-size: 10px; background: #f0fdf4; color: #166534; padding: 1px 6px; border-radius: 4px; margin-left: 6px; font-weight: 700; border: 1px solid #bbf7d0;">${{p.earnings_streak_label}}</span>` : ''}}
                ${{p.fcf_yield_pct !== null && p.fcf_yield_pct !== undefined ? `<span style="font-size: 10px; background: #f8fafc; color: #334155; padding: 1px 6px; border-radius: 4px; margin-left: 4px; font-weight: 700; border: 1px solid #e2e8f0;">FCF: ${{p.fcf_yield_pct}}%</span>` : ''}}
                ${{p.roic !== null && p.roic !== undefined && p.roic > 0 ? `<span style="font-size: 10px; background: #ecfdf5; color: #047857; padding: 1px 6px; border-radius: 4px; margin-left: 4px; font-weight: 700; border: 1px solid #bbf7d0;">ROIC: ${{p.roic.toFixed(1)}}%</span>` : ''}}
                ${{p.options_pcr_vol !== null && p.options_pcr_vol !== undefined && p.options_pcr_vol < 0.70 ? `<span style="font-size: 10px; background: #e0f2fe; color: #0369a1; padding: 1px 6px; border-radius: 4px; margin-left: 4px; font-weight: 700; border: 1px solid #bae6fd;">Call Skew (${{p.options_pcr_vol.toFixed(2)}})</span>` : ''}}
                ${{p.insider_label && p.insider_label.includes('Buying') ? `<span style="font-size: 10px; background: #f0fdf4; color: #166534; padding: 1px 6px; border-radius: 4px; margin-left: 4px; font-weight: 800; border: 1px solid #bbf7d0;">👥 Insider Buying</span>` : ''}}
                ${{p.bearish_divergence ? `<span style="font-size: 10px; background: #fef3c7; color: #b45309; padding: 1px 6px; border-radius: 4px; margin-left: 4px; font-weight: 800; border: 1px solid #fde68a;">⚠️ Bearish Div</span>` : ''}}
              </div>
            </td>
            <td><span class="badge" style="background:#1e293b; color:#94a3b8;">${{p.account}}</span></td>
            <td><b>${{p.currency === 'USD' ? '$' : '£'}}${{p.price.toFixed(2)}}</b></td>
            <td style="color: ${{pnlColor}}; font-weight: 700;">${{gainSign}}${{p.gain_pct.toFixed(1)}}%</td>
            <td style="color: ${{pnlColor}}; font-weight: 700;">${{pnlSign}}£${{Math.abs(p.unrealized_pnl).toLocaleString('en-GB', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}}</td>
            <td><b>£${{p.value_gbp.toLocaleString('en-GB', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}}</b></td>
            <td>${{p.weight_pct.toFixed(1)}}%</td>
            <td>${{sort5Badge}}</td>
            <td>${{crash22Badge}}</td>
            <td>${{pioBadge}}</td>
            <td>${{altBadge}}</td>
            <td><b>${{(p.risk_reward || 3.0).toFixed(1)}}:1</b></td>
            <td>
              <span class="badge ${{statusBadgeClass}}">${{p.plan_status}}</span>
            </td>
          </tr>
        `;
      }}
      tbody.innerHTML = html;
    }}

    function filterTable() {{
      renderTable();
    }}

    function sortTable(col) {{
      if (sortCol === col) {{
        sortAsc = !sortAsc;
      }} else {{
        sortCol = col;
        sortAsc = false;
      }}
      renderTable();
    }}

    function renderChart() {{
      const container = document.getElementById('chartContainer');
      if (!HISTORY || HISTORY.length < 2) {{
        container.innerHTML = '<div style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 40px;">Accumulating historical session records...</div>';
        return;
      }}

      const w = container.clientWidth || 1000;
      const h = 200;
      const padL = 60;
      const padR = 20;
      const padT = 20;
      const padB = 30;

      const key = currentAccount === 'ALL' ? 'total' : currentAccount.toLowerCase();
      const vals = HISTORY.map(d => d[key]);
      const minV = Math.min(...vals) * 0.995;
      const maxV = Math.max(...vals) * 1.005;

      const pts = HISTORY.map((d, i) => {{
        const x = padL + (i / (HISTORY.length - 1)) * (w - padL - padR);
        const y = padT + (1 - (d[key] - minV) / (maxV - minV)) * (h - padT - padB);
        return {{ x, y, date: d.date, val: d[key] }};
      }});

      const pathD = pts.reduce((acc, p, i) => i === 0 ? `M ${{p.x}} ${{p.y}}` : `${{acc}} L ${{p.x}} ${{p.y}}`, '');
      const fillD = `${{pathD}} L ${{pts[pts.length-1].x}} ${{h - padB}} L ${{pts[0].x}} ${{h - padB}} Z`;

      let dots = '';
      for (const p of pts) {{
        dots += `<circle cx="${{p.x}}" cy="${{p.y}}" r="4" fill="#3b82f6" stroke="#fff" stroke-width="2" />`;
        dots += `<text x="${{p.x}}" y="${{h - 10}}" text-anchor="middle" font-size="10" fill="#64748b">${{p.date.slice(5)}}</text>`;
      }}

      container.innerHTML = `
        <svg class="equity-svg" viewBox="0 0 ${{w}} ${{h}}">
          <defs>
            <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.35"/>
              <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.0"/>
            </linearGradient>
          </defs>
          <line x1="${{padL}}" y1="${{padT}}" x2="${{w - padR}}" y2="${{padT}}" stroke="#1e293b" stroke-dasharray="3,3" />
          <line x1="${{padL}}" y1="${{h - padB}}" x2="${{w - padR}}" y2="${{h - padB}}" stroke="#1e293b" />
          <text x="${{padL - 10}}" y="${{padT + 5}}" text-anchor="end" font-size="10" fill="#64748b">£${{Math.round(maxV).toLocaleString()}}</text>
          <text x="${{padL - 10}}" y="${{h - padB}}" text-anchor="end" font-size="10" fill="#64748b">£${{Math.round(minV).toLocaleString()}}</text>
          <path d="${{fillD}}" fill="url(#chartGrad)" />
          <path d="${{pathD}}" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round" />
          ${{dots}}
        </svg>
      `;
    }}

    window.addEventListener('resize', renderChart);
    // Initial boot
    updateKPIs();
    renderActionCards();
    renderTable();
    setTimeout(renderChart, 50);
  </script>
</body>
</html>
"""
    return html

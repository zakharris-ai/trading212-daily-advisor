"""Email formatting and dispatch module designed for simplicity and institutional research depth."""

import logging
import math
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from typing import Any, Dict, List, Optional
from config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, RECIPIENT_EMAILS
from strategy_engine import generate_tldr_summary

logger = logging.getLogger(__name__)


def format_price(ticker: str, price: Optional[float], currency: str = "USD") -> str:
    """Formats price clearly with appropriate currency symbol."""
    if price is None:
        p_val = 0.0
    else:
        try:
            p_val = float(price)
            if math.isnan(p_val):
                p_val = 0.0
        except (ValueError, TypeError):
            p_val = 0.0

    raw_c = str(currency or "USD").strip()
    is_pence = (raw_c == "GBp") or (raw_c.upper() in ("GBX", "PENCE", "GBPX"))
    c = raw_c.upper()
    if is_pence or (c == "GBP" and p_val > 500):
        return f"{p_val:,.1f}p"
    elif c == "GBP":
        return f"£{p_val:,.2f}"
    elif c == "USD":
        return f"${p_val:,.2f}"
    elif c == "EUR":
        return f"€{p_val:,.2f}"
    return f"{p_val:,.2f} {currency}"


def build_tldr_html(tldr_summary: Dict[str, Any]) -> str:
    """Builds a high-visibility, 30-second executive TL;DR action card for the top of the email."""
    if not tldr_summary:
        return ""

    account_cards = []
    for acc_name, t_info in tldr_summary.items():
        phase = t_info.get("phase", "ALL_SET")
        phase_label = t_info.get("phase_label", "✅ ALL SET")
        phase_desc = t_info.get("phase_desc", "")
        net_budget = t_info.get("net_budget", 0.0)
        buffer_reserve = t_info.get("buffer_reserve", 1500.0)
        sells_queued = t_info.get("sells_already_queued", [])
        buys_queued = t_info.get("buys_already_queued", [])
        sells_to_place = t_info.get("sells_to_place", [])
        buys_to_place = t_info.get("buys_to_place", [])

        # Visual styling per phase
        if phase == "BUY_NOW":
            card_border = "#22c55e"
            card_bg = "#f0fdf4"
            badge_bg = "#16a34a"
            badge_text = "#ffffff"
        elif phase == "AWAIT_FILLS_THEN_BUY":
            card_border = "#f59e0b"
            card_bg = "#fffbeb"
            badge_bg = "#d97706"
            badge_text = "#ffffff"
        elif phase == "SELL_FIRST":
            card_border = "#ef4444"
            card_bg = "#fef2f2"
            badge_bg = "#dc2626"
            badge_text = "#ffffff"
        else:
            card_border = "#38bdf8"
            card_bg = "#f0f9ff"
            badge_bg = "#0284c7"
            badge_text = "#ffffff"

        # Queued Sells notice
        queued_sells_html = ""
        if sells_queued:
            queued_items = "".join([f"<li style='margin-bottom: 2px;'>{s}</li>" for s in sells_queued])
            queued_sells_html = f"""
            <div style="background: #ffffff; border: 1px solid #fde68a; border-radius: 8px; padding: 10px 14px; margin-top: 10px; font-size: 12px; color: #92400e;">
              <b>⏳ Sells Already Queued in T212 (Awaiting market fill):</b>
              <ul style="margin: 4px 0 0 16px; padding: 0;">
                {queued_items}
              </ul>
            </div>
            """

        # Queued Buys notice
        queued_buys_html = ""
        if buys_queued:
            queued_b_items = "".join([f"<li style='margin-bottom: 2px;'>{b}</li>" for b in buys_queued])
            queued_buys_html = f"""
            <div style="background: #ffffff; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px 14px; margin-top: 10px; font-size: 12px; color: #166534;">
              <b>🟢 Buys Already Queued in T212:</b>
              <ul style="margin: 4px 0 0 16px; padding: 0;">
                {queued_b_items}
              </ul>
            </div>
            """

        # Sells to place
        sells_to_place_html = ""
        if sells_to_place:
            s_items = "".join([
                f"<li style='margin-bottom: 4px;'><b>SELL {s['name']} ({s['symbol']})</b> ➔ Sell {s['shares']:,.0f} shares (+£{s['estimated_gbp']:,.0f})</li>"
                for s in sells_to_place
            ])
            sells_to_place_html = f"""
            <div style="background: #ffffff; border: 1px solid #fecaca; border-radius: 8px; padding: 12px 14px; margin-top: 10px;">
              <div style="font-weight: 800; color: #991b1b; font-size: 13px; margin-bottom: 6px;">🔴 1. EXECUTE THESE SELLS FIRST:</div>
              <ol style="margin: 0 0 0 18px; padding: 0; font-size: 13px; color: #991b1b; line-height: 1.5;">
                {s_items}
              </ol>
            </div>
            """

        # Buys to place
        buys_to_place_html = ""
        if buys_to_place:
            buy_timing = "2. BUY ORDERS TO PLACE ONCE SELLS FILL:" if phase == "AWAIT_FILLS_THEN_BUY" else "BUY ORDERS TO PLACE TODAY:"
            timing_color = "#92400e" if phase == "AWAIT_FILLS_THEN_BUY" else "#166534"
            timing_border = "#fde68a" if phase == "AWAIT_FILLS_THEN_BUY" else "#bbf7d0"
            b_items = "".join([
                f"<li style='margin-bottom: 6px;'><b>BUY {b['name']} ({b['symbol']})</b> ➔ Enter <b>£{b['amount_gbp']:,.0f}</b></li>"
                for b in buys_to_place
            ])
            buys_to_place_html = f"""
            <div style="background: #ffffff; border: 1px solid {timing_border}; border-radius: 8px; padding: 12px 14px; margin-top: 10px;">
              <div style="font-weight: 800; color: {timing_color}; font-size: 13px; margin-bottom: 6px;">🎯 {buy_timing}</div>
              <ol style="margin: 0 0 0 18px; padding: 0; font-size: 13px; color: #0f172a; line-height: 1.5;">
                {b_items}
              </ol>
            </div>
            """

        all_set_html = ""
        if phase == "ALL_SET":
            all_set_html = f"""
            <div style="background: #ffffff; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px 14px; margin-top: 10px; color: #166534; font-size: 13px;">
              ✨ <b>All Set:</b> All positions are healthy and compounding. No buy or sell orders needed today.
            </div>
            """

        card_html = f"""
        <div style="background: {card_bg}; border: 2px solid {card_border}; border-radius: 12px; padding: 16px; margin-bottom: 14px;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
            <div style="font-size: 16px; font-weight: 800; color: #0f172a;">
              👤 {acc_name}'s Account
            </div>
            <div>
              <span style="background: {badge_bg}; color: {badge_text}; font-size: 12px; font-weight: 800; padding: 4px 12px; border-radius: 6px; letter-spacing: 0.02em;">
                {phase_label}
              </span>
            </div>
          </div>

          <div style="font-size: 13px; color: #334155; line-height: 1.5; margin-bottom: 4px;">
            {phase_desc}
          </div>

          {queued_sells_html}
          {queued_buys_html}
          {sells_to_place_html}
          {buys_to_place_html}
          {all_set_html}

          <div style="margin-top: 10px; font-size: 11px; color: #64748b; border-top: 1px dashed #cbd5e1; padding-top: 8px; display: flex; justify-content: space-between; flex-wrap: wrap;">
            <span>🛡️ <b>Protected Cash Floor:</b> £{buffer_reserve:,.0f} (Ring-fenced & safe)</span>
            <span>💰 <b>Deployable Capital:</b> £{net_budget:,.0f}</span>
          </div>
        </div>
        """
        account_cards.append(card_html)

    all_cards_str = "".join(account_cards)

    return f"""
    <!-- ⚡ 30-SECOND TL;DR — WHAT TO DO TODAY -->
    <div style="background-color: #ffffff; border-radius: 16px; border: 2px solid #0284c7; margin-bottom: 24px; padding: 20px; box-shadow: 0 4px 14px rgba(2, 132, 199, 0.12);">
      <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f1f5f9; padding-bottom: 12px; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
        <div>
          <div style="display: inline-block; background: #e0f2fe; color: #0369a1; font-size: 11px; font-weight: 800; text-transform: uppercase; padding: 3px 8px; border-radius: 4px; margin-bottom: 4px; border: 1px solid #bae6fd;">
            ⚡ 30-SECOND EXECUTIVE TL;DR
          </div>
          <h2 style="margin: 0; font-size: 20px; font-weight: 900; color: #0f172a;">What You Need To Do Today</h2>
        </div>
        <div style="font-size: 12px; color: #64748b; text-align: right;">
          Partitioned by Account • <b style="color: #0284c7;">2-Minute Read</b>
        </div>
      </div>

      <div>
        {all_cards_str}
      </div>
    </div>
    """


def generate_html_digest(
    results_by_account: Dict[str, Dict[str, Any]],
    macro_regime: Dict[str, Any],
    consolidated_view: Optional[Dict[str, Any]] = None
) -> str:
    """Generates a clean, non-trader friendly daily action plan with collapsible PhD-level research dossiers."""
    date_str = datetime.now().strftime("%A, %B %d, %Y")

    # Generate 30-second executive TL;DR summary
    tldr_summary = generate_tldr_summary(results_by_account)
    tldr_section_html = build_tldr_html(tldr_summary)

    if consolidated_view is None:
        consolidated_view = next((v.get("consolidated_view") for v in results_by_account.values() if v.get("consolidated_view")), None)


    consolidated_overview_html = ""
    if consolidated_view and len(results_by_account) > 1:
        total_nw = consolidated_view.get("total_net_worth_gbp", 0.0)
        tot_inv = consolidated_view.get("total_invested_gbp", 0.0)
        tot_liq = consolidated_view.get("total_liquid_cash_gbp", 0.0)
        tot_blk = consolidated_view.get("total_blocked_cash_gbp", 0.0)
        tot_ppl = consolidated_view.get("total_ppl_gbp", 0.0)
        ppl_c = "#16a34a" if tot_ppl >= 0 else "#dc2626"
        ppl_s = "+" if tot_ppl > 0 else ""

        dod = consolidated_view.get("day_over_day", {})
        dod_combined_html = ""
        if dod and dod.get("has_prior"):
            d_gbp = dod.get("combined_delta_gbp", 0.0)
            d_pct = dod.get("combined_delta_pct", 0.0)
            d_color = "#16a34a" if d_gbp >= 0 else "#dc2626"
            d_sign = "+" if d_gbp >= 0 else ""
            dod_combined_html = f"<div style='font-size: 12px; font-weight: 700; color: {d_color}; margin-top: 2px;'>{d_sign}£{d_gbp:,.2f} ({d_pct:+.2f}%) vs yesterday</div>"

        shares = consolidated_view.get("account_shares", {})
        account_chips = ""
        for a_name, s in shares.items():
            acc_dod = dod.get("accounts", {}).get(a_name, {}) if (dod and dod.get("has_prior")) else {}
            dod_acc_txt = ""
            if acc_dod and acc_dod.get("delta_gbp") is not None:
                ad_gbp = acc_dod.get("delta_gbp", 0.0)
                ad_color = "#16a34a" if ad_gbp >= 0 else "#dc2626"
                ad_sign = "+" if ad_gbp >= 0 else ""
                dod_acc_txt = f" <span style='color: {ad_color}; font-size: 11px;'>({ad_sign}£{ad_gbp:,.0f})</span>"
            account_chips += f"<span style='background: #e2e8f0; color: #1e293b; padding: 4px 10px; border-radius: 6px; font-weight: 700; margin-right: 8px;'>👤 {a_name}: £{s['total_value_gbp']:,.0f} ({s['share_of_total_pct']}%){dod_acc_txt}</span>"

        thematic_chips = ""
        for tb in consolidated_view.get("thematic_breakdown", [])[:4]:
            thematic_chips += f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px; display: flex; justify-content: space-between; align-items: center;">
              <span style="font-size: 12px; color: #334155;"><b>{tb['cluster']}</b></span>
              <span style="font-size: 11px; font-weight: 800; background: {tb['badge_bg']}; color: #ffffff; padding: 2px 8px; border-radius: 4px;">{tb['weight_pct']}%</span>
            </div>
            """

        accounts_label = " & ".join(results_by_account.keys()) or "Portfolio"
        consolidated_overview_html = f"""
        <!-- Consolidated Portfolio Dashboard -->
        <div style="background-color: #ffffff; border-radius: 16px; border: 2px solid #3b82f6; margin-bottom: 28px; padding: 24px; box-shadow: 0 4px 12px -2px rgba(59, 130, 246, 0.12);">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 14px;">
            <div>
              <div style="display: inline-block; background: #eff6ff; color: #1d4ed8; font-size: 11px; font-weight: 800; text-transform: uppercase; padding: 3px 8px; border-radius: 4px; margin-bottom: 6px; border: 1px solid #bfdbfe;">
                🌐 Unified Portfolio Intelligence
              </div>
              <h2 style="margin: 0; font-size: 20px; font-weight: 800; color: #0f172a;">Portfolio Overview ({accounts_label})</h2>
              <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                Aggregated holdings across configured account(s) for Modern Portfolio Theory (MPT) risk and concentration modeling.
              </div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Combined Net Worth</div>
              <b style="color: #0f172a; font-size: 24px; font-weight: 900;">£{total_nw:,.2f}</b>
              {dod_combined_html}
            </div>
          </div>

          <!-- Account Distribution Chips -->
          <div style="margin-bottom: 16px; font-size: 12px; display: flex; align-items: center; flex-wrap: wrap; gap: 6px;">
            <span style="color: #64748b; font-weight: 600;">Account Split:</span>
            {account_chips}
          </div>

          <!-- 4-Stat Grid -->
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 18px;">
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
              <span style="color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 700;">Total Invested</span><br>
              <b style="color: #0f172a; font-size: 15px;">£{tot_inv:,.2f}</b>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
              <span style="color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 700;">Liquid Cash</span><br>
              <b style="color: #0284c7; font-size: 15px;">£{tot_liq:,.2f}</b>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
              <span style="color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 700;">Pending Blocked</span><br>
              <b style="color: #f59e0b; font-size: 15px;">£{tot_blk:,.2f}</b>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
              <span style="color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 700;">Combined Profit</span><br>
              <b style="color: {ppl_c}; font-size: 15px;">{ppl_s}£{tot_ppl:,.2f}</b>
            </div>
          </div>

          <!-- Global Thematic Exposure -->
          <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 8px;">
              📊 Consolidated Macro Cluster Allocation
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px;">
              {thematic_chips}
            </div>
          </div>

          <!-- 5-Year Horizon Crash Stress-Test & Compounding Trajectory -->
          <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 12px 16px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px; margin-bottom: 8px;">
              <div style="font-size: 12px; font-weight: 800; color: #166534; text-transform: uppercase;">
                🛡️ 5-Year Horizon All-Weather & Crash Stress-Test (2026–2031)
              </div>
              <span style="background: #16a34a; color: #ffffff; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">
                {macro_regime.get('crash_defcon', '🟢 DEFCON 5 — CALM EXPANSION')}
              </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; font-size: 12px;">
              <div style="background: #ffffff; border: 1px solid #dcfce7; border-radius: 6px; padding: 8px;">
                <span style="color: #64748b; font-size: 10px; text-transform: uppercase; font-weight: 700;">Combined 5Y Sortino</span><br>
                <b style="color: #15803d; font-size: 14px;">{consolidated_view.get('portfolio_5y_risk_profile', {}).get('current_sortino_5y', 1.5):.2f}</b> <span style="font-size: 10px; color: #64748b;">(5Y CAGR: +{consolidated_view.get('portfolio_5y_risk_profile', {}).get('current_cagr_5y_pct', 20.0):.1f}%)</span>
              </div>
              <div style="background: #ffffff; border: 1px solid #dcfce7; border-radius: 6px; padding: 8px;">
                <span style="color: #64748b; font-size: 10px; text-transform: uppercase; font-weight: 700;">2022 Crash Stress-Test</span><br>
                <b style="color: #dc2626; font-size: 14px;">{consolidated_view.get('portfolio_5y_risk_profile', {}).get('current_stress_2022_pct', -25.0):+.1f}%</b> <span style="font-size: 10px; color: #64748b;">(£{consolidated_view.get('portfolio_5y_risk_profile', {}).get('current_stress_2022_gbp', 0.0):,.0f})</span>
              </div>
              <div style="background: #ffffff; border: 1px solid #dcfce7; border-radius: 6px; padding: 8px;">
                <span style="color: #64748b; font-size: 10px; text-transform: uppercase; font-weight: 700;">All-Weather Shield</span><br>
                <b style="color: #0284c7; font-size: 14px;">{consolidated_view.get('portfolio_5y_risk_profile', {}).get('current_shield_pct', 0.0):.1f}%</b> <span style="font-size: 10px; color: #64748b;">(Beta: {consolidated_view.get('portfolio_5y_risk_profile', {}).get('current_beta_5y', 1.0):.2f})</span>
              </div>
              <div style="background: #ffffff; border: 1px solid #dcfce7; border-radius: 6px; padding: 8px;">
                <span style="color: #64748b; font-size: 10px; text-transform: uppercase; font-weight: 700;">2031 5Y Wealth Target</span><br>
                <b style="color: #0f172a; font-size: 13px;">£{consolidated_view.get('portfolio_5y_risk_profile', {}).get('projected_5y_conservative_gbp', 0.0):,.0f} – £{consolidated_view.get('portfolio_5y_risk_profile', {}).get('projected_5y_target_gbp', 0.0):,.0f}</b>
              </div>
            </div>
            <div style="font-size: 11px; color: #166534; margin-top: 6px;">
              💡 <b>5-Year Crash Directive:</b> {macro_regime.get('crash_shield_directive', 'Maintain 15%+ All-Weather Crash Shield (Gold SGLN.L + Cash Fortress BRK-B + Healthcare Compounders LLY/ABBV).')}
            </div>
          </div>

          <!-- Capital Isolation Notice -->
          <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #991b1b; line-height: 1.5;">
            🔒 <b>Capital Boundary:</b> Money cannot be moved between accounts. Instructions below are partitioned separately for each Trading 212 account to ensure safe, isolated execution.
          </div>
        </div>
        """
    
    account_sections_html = ""
    for acc_name, data in results_by_account.items():
        cash_info = data.get("cash", {})
        analyses = data.get("analyses", [])
        rebalance_plan = data.get("rebalancing_plan", {})

        avail_cash = rebalance_plan.get("available_cash_gbp", 0.0)
        blocked_cash = rebalance_plan.get("blocked_cash_gbp", cash_info.get("blocked_gbp", cash_info.get("blocked", 0.0)))
        sells_cash = rebalance_plan.get("cash_from_sells_gbp", 0.0)
        deployable = rebalance_plan.get("total_deployable_gbp", 0.0)
        buffer_cash = rebalance_plan.get("buffer_reserve_gbp", 0.0)
        net_budget = rebalance_plan.get("net_budget_to_deploy_gbp", max(0.0, deployable - buffer_cash))
        ppl = cash_info.get("ppl", 0.0)
        ppl_color = "#16a34a" if ppl >= 0 else "#dc2626"
        ppl_sign = "+" if ppl > 0 else ""

        sell_orders = rebalance_plan.get("sell_orders", [])
        buy_orders = rebalance_plan.get("buy_orders", [])
        hold_items = rebalance_plan.get("hold_items", [])
        table_items = rebalance_plan.get("table_items", analyses)
        has_orders_scope = rebalance_plan.get("has_orders_scope", False)
        pending_buys = rebalance_plan.get("pending_buys", {})
        pending_sells = rebalance_plan.get("pending_sells", {})

        # Build Action Cards
        action_cards_html = ""
        step_number = 1

        # 0. Pending Transactions / Blocked Cash Card
        pending_sell_orders = rebalance_plan.get("pending_sell_orders", [])
        if blocked_cash > 0 or pending_buys or pending_sells or pending_sell_orders:
            pending_items_html = ""
            if has_orders_scope and (pending_buys or pending_sells):
                pending_items_html += "<ul style='margin: 4px 0 0 16px; padding: 0;'>"
                for pt, pb in pending_buys.items():
                    name = pb.get("name", pt)
                    val = pb.get("estimated_value", pb.get("value", 0.0))
                    qty = pb.get("quantity", 0.0)
                    strat = pb.get("strategy", "QUANTITY")
                    if strat == "VALUE" or (val > 0 and qty == 0):
                        pending_items_html += f"<li>🟢 <b>BUY {name} ({pt}):</b> ~£{val:,.0f} value order queued for execution.</li>"
                    else:
                        limit_str = f" @ limit £{pb['limit_price']:,.2f}" if pb.get("limit_price", 0) > 0 else ""
                        val_str = f" (~£{val:,.0f})" if val > 0 else ""
                        pending_items_html += f"<li>🟢 <b>BUY {name} ({pt}):</b> {qty:,.2f} shares{limit_str}{val_str} queued.</li>"

                for pt, ps in pending_sells.items():
                    name = ps.get("name", pt)
                    val = float(ps.get("estimated_value", ps.get("value", 0.0)) or 0.0)
                    qty = float(ps.get("quantity", 0.0) or 0.0)
                    if math.isnan(val) or val <= 0:
                        matching_pso = next((p for p in pending_sell_orders if p.get("ticker") == pt or p.get("search_symbol") == pt), None)
                        if matching_pso:
                            val = float(matching_pso.get("estimated_gbp", 0.0) or 0.0)
                            if math.isnan(val):
                                val = 0.0
                            if qty == 0:
                                qty = matching_pso.get("shares", 0.0)
                    if math.isnan(val):
                        val = 0.0
                    lim_p = ps.get("limit_price", 0.0)
                    curr_sym = "$" if ps.get("currency") == "USD" or "US" in ps.get("raw_ticker", "") else "£"
                    limit_str = f" @ limit {curr_sym}{lim_p:,.2f}" if lim_p > 0 else ""
                    val_str = f" (+£{val:,.0f} cash when filled)" if val > 0 else ""
                    pending_items_html += f"<li>🔴 <b>SELL {name} ({pt}):</b> ~{qty:,.2f} shares{limit_str} queued{val_str}.</li>"
                pending_items_html += "</ul>"
            elif pending_sell_orders:
                pending_items_html = f"<div>👉 <b>Committed Pending Balance:</b> £{blocked_cash:,.2f} in Trading 212.</div>"
                pending_items_html += "<div style='margin-top: 6px;'><b>⏳ Orders Already Queued (No action needed):</b><ul style='margin: 4px 0 0 16px; padding: 0;'>"
                for pso in pending_sell_orders:
                    pso_val = float(pso.get('estimated_gbp', 0.0) or 0.0)
                    if math.isnan(pso_val): pso_val = 0.0
                    pending_items_html += f"<li>🔴 <b>SELL {pso['name']} ({pso['search_symbol']}):</b> ~{pso['shares']:,.0f} shares already submitted & queued (+£{pso_val:,.0f} cash when filled).</li>"
                pending_items_html += "</ul></div>"
            else:
                pending_items_html = (
                    f"👉 <b>Committed Capital:</b> Exactly <b>£{blocked_cash:,.2f}</b> is currently committed to pending transactions or open orders in Trading 212.<br>"
                    f"<span style='font-size: 12px; color: #64748b;'>ℹ️ To display individual pending order tickets here, enable the <i>'Orders'</i> permission toggle in your Trading 212 API key settings.</span>"
                )

            action_cards_html += f"""
            <div style="background-color: #fffbeb; border: 2px solid #fde68a; border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 8px;">
                <div style="font-size: 16px; font-weight: 800; color: #92400e;">
                  STEP {step_number}: ⏳ PENDING TRANSACTIONS (NO ACTION REQUIRED)
                </div>
                <div style="background-color: #f59e0b; color: #ffffff; font-size: 11px; font-weight: 800; padding: 4px 10px; border-radius: 9999px;">
                  ALREADY SUBMITTED
                </div>
              </div>
              <div style="background: #ffffff; border-radius: 8px; padding: 12px 14px; margin: 10px 0; border: 1px solid #fef3c7; font-size: 14px; color: #1e293b; line-height: 1.5;">
                {pending_items_html}
              </div>
              <div style="font-size: 13px; color: #78350f; line-height: 1.4;">
                💡 <b>Strategic Rationale:</b> These orders are already in Trading 212. You do NOT need to tap Sell again. Once executed, proceeds will replenish liquid cash. Your <b>£{buffer_cash:,.0f}</b> reserve floor remains 100% ring-fenced.
              </div>
            </div>
            """
            step_number += 1

        # 0b. Protected Cash Reserve Card (when no new buys or sells are needed)
        non_pending_sells = [s for s in sell_orders if not s.get("is_pending")]
        if not non_pending_sells and not buy_orders:
            action_cards_html += f"""
            <div style="background-color: #f0f9ff; border: 2px solid #bae6fd; border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 8px;">
                <div style="font-size: 16px; font-weight: 800; color: #0369a1;">
                  STEP {step_number}: 🛡️ PROTECT CASH RESERVE (£{buffer_cash:,.0f} SAFE TO WITHDRAW)
                </div>
                <div style="background-color: #0284c7; color: #ffffff; font-size: 11px; font-weight: 800; padding: 4px 10px; border-radius: 9999px;">
                  100% LIQUID & SAFE
                </div>
              </div>
              <div style="background: #ffffff; border-radius: 8px; padding: 12px 14px; margin: 10px 0; border: 1px solid #e0f2fe; font-size: 14px; color: #1e293b; line-height: 1.5;">
                👉 <b>Cash Status:</b> You have <b>{cash_info.get('free_label', f'£{avail_cash:,.2f}')}</b> in uninvested liquid cash (plus <b>£{blocked_cash:,.2f}</b> in pending orders). Your <b>£{buffer_cash:,.0f}</b> safety floor is fully ring-fenced and ready for immediate withdrawal whenever you need it.
              </div>
              <div style="font-size: 13px; color: #075985; line-height: 1.4;">
                💡 <b>Strategic Rationale:</b> Zero new buy orders recommended today so your liquid cash reserve stays completely intact.
              </div>
            </div>
            """
            step_number += 1

        # 1. Sells & Trims (ONLY new, unplaced orders)
        for so in non_pending_sells:
            is_trim = "TRIM" in so.get("action", "")
            action_verb = "TRIM" if is_trim else "SELL"
            card_bg = "#fffbeb" if is_trim else "#fff1f2"
            card_border = "#fde68a" if is_trim else "#fecdd3"
            text_color = "#b45309" if is_trim else "#9f1239"
            badge_bg = "#f59e0b" if is_trim else "#f43f5e"
            badge_text = f"BANKS +£{so['estimated_gbp']:,.0f} CASH" if is_trim else f"GETS YOU +£{so['estimated_gbp']:,.0f} CASH"
            instruction_text = so.get("app_instruction", f"Open Trading 212 ➔ Search '{so['search_symbol']}' ➔ Tap Sell")
            share_instruction = f"Sell <b>{so['shares']:,.0f} shares</b> (banking partial profits)" if is_trim else f"Sell <b>all {so['shares']:,.0f} shares</b>"

            action_cards_html += f"""
            <div style="background-color: {card_bg}; border: 2px solid {card_border}; border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 8px;">
                <div style="font-size: 16px; font-weight: 800; color: {text_color};">
                  STEP {step_number}: 🟠 {action_verb} {so['name']} ({so['search_symbol']})
                </div>
                <div style="background-color: {badge_bg}; color: #ffffff; font-size: 12px; font-weight: 800; padding: 4px 12px; border-radius: 9999px;">
                  {badge_text}
                </div>
              </div>
              
              <div style="background: #ffffff; border-radius: 8px; padding: 12px 14px; margin: 10px 0; border: 1px solid {card_border}; font-size: 14px; color: #1e293b; line-height: 1.5;">
                👉 <b>What to tap in Trading 212:</b><br>
                Open Trading 212 ➔ Search <b>"{so['search_symbol']}"</b> ➔ Tap <b>Sell</b> ➔ {share_instruction}.
              </div>

              <div style="font-size: 13px; color: {text_color}; line-height: 1.4;">
                💡 <b>Why:</b> {so['plain_english_why']}
              </div>
            </div>
            """
            step_number += 1

        # 2. Buys
        for bo in buy_orders:
            p_str = format_price(bo['ticker'], bo['price'], bo['currency'])
            is_new = bo.get("is_new", False)
            tag_label = "NEW ADDITION TO PORTFOLIO" if is_new else "ADD TO EXISTING HOLDING"
            tag_bg = "#8b5cf6" if is_new else "#10b981"
            rr_val = bo.get("risk_reward_ratio", 2.5)
            p_score = bo.get("piotroski_score")
            p_score_badge = f'<span style="background: #ecfdf5; color: #047857; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">🛡️ Piotroski: {p_score}/9</span>' if p_score is not None else ''
            
            alt_val = bo.get("altman_z")
            alt_lbl = bo.get("altman_label")
            if alt_val is not None:
                alt_badge = f'<span style="background: #fef3c7; color: #92400e; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">🏛️ Altman Z: {alt_val:.1f}</span>'
            elif alt_lbl and "N/A" not in alt_lbl:
                alt_badge = f'<span style="background: #fef3c7; color: #92400e; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">🏛️ {alt_lbl}</span>'
            else:
                alt_badge = ''

            roic_val = bo.get("roic")
            roic_badge = f'<span style="background: #ecfdf5; color: #047857; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">💎 ROIC: {roic_val:.1f}%</span>' if (roic_val is not None and roic_val > 0) else ''

            insider_lbl = bo.get("insider_label", "")
            insider_badge = f'<span style="background: #f0fdf4; color: #166534; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">👥 Insider: {insider_lbl.split("(")[0].replace("🟢", "").strip()}</span>' if ("Buying" in insider_lbl or "Selling" in insider_lbl) and "Routine" not in insider_lbl else ''

            options_lbl = bo.get("options_pcr_label", "")
            pcr_vol = bo.get("options_pcr_vol")
            options_badge = f'<span style="background: #e0f2fe; color: #0369a1; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">📊 PCR: {pcr_vol:.2f} (Call Skew)</span>' if (pcr_vol is not None and pcr_vol < 0.70) else ''

            inst_lbl = bo.get("institutional_flow_label", "")
            inst_badge = '<span style="background: #ecfdf5; color: #047857; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">🏛️ 13F Inflows</span>' if "Inflows" in inst_lbl else ''

            e_status = bo.get("earnings_catalyst_status", "✅ Clear")
            w_label = bo.get("weekly_trend_label", "Weekly Stage 2 Confirmed")
            w_color = "#047857" if ("Stage 2" in w_label or "Confirmed" in w_label or "Support" in w_label) else "#b91c1c"
            w_bg = "#ecfdf5" if ("Stage 2" in w_label or "Confirmed" in w_label or "Support" in w_label) else "#fef2f2"

            is_lev = any(k in bo.get("ticker", "") for k in ["QQQ3", "3LUS", "3SEM"])
            lev_badge = '<span style="background: #fef3c7; color: #92400e; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">🔒 Leveraged (8% Max Cap Applied)</span>' if is_lev else ''
            cagr5 = bo.get("cagr_5y_pct", 18.0)
            sort5 = bo.get("sortino_5y", 1.5)
            crash22 = bo.get("crash_2022_ret_pct", -15.0)
            role_lbl = bo.get("crash_role_label", "📈 Core Compounder")
            crash_color = "#047857" if crash22 >= -5.0 else ("#0369a1" if crash22 >= -20.0 else "#b45309")
            crash_bg = "#ecfdf5" if crash22 >= -5.0 else ("#e0f2fe" if crash22 >= -20.0 else "#fef3c7")

            action_cards_html += f"""
            <div style="background-color: #f0fdf4; border: 2px solid #bbf7d0; border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 8px;">
                <div style="font-size: 16px; font-weight: 800; color: #166534;">
                  STEP {step_number}: 🟢 BUY {bo['name']} ({bo['search_symbol']})
                </div>
                <div style="background-color: {tag_bg}; color: #ffffff; font-size: 11px; font-weight: 800; padding: 4px 10px; border-radius: 9999px;">
                  {tag_label}
                </div>
              </div>

              <!-- Institutional Quality Badges -->
              <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px;">
                <span style="background: #dcfce7; color: #166534; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">{role_lbl}</span>
                <span style="background: #ecfdf5; color: #047857; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">🚀 5Y CAGR: {cagr5:+.1f}% (Sortino {sort5:.2f})</span>
                <span style="background: {crash_bg}; color: {crash_color}; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">🛡️ 2022 Crash: {crash22:+.1f}%</span>
                <span style="background: #e0f2fe; color: #0369a1; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">📈 {bo.get('minervini_stage', 'Stage 2 Bull')}</span>
                <span style="background: {w_bg}; color: {w_color}; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">🌊 {w_label}</span>
                <span style="background: #ecfdf5; color: #047857; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">📊 {bo.get('volume_footprint_label', 'Accumulation')}</span>
                <span style="background: #e0f2fe; color: #0369a1; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">📐 {rr_val}:1 Risk/Reward</span>
                {p_score_badge}
                {alt_badge}
                {roic_badge}
                {insider_badge}
                {options_badge}
                {inst_badge}
                <span style="background: #f0fdf4; color: #15803d; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 6px;">🎯 Revisions: {bo.get('eps_revision_label', 'Neutral Consensus')}</span>
                <span style="background: #f1f5f9; color: #334155; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 6px;">📅 Earnings: {e_status}</span>
                {lev_badge}
              </div>

              <div style="background: #ffffff; border-radius: 8px; padding: 12px 14px; margin: 10px 0; border: 1px solid #dcfce7; font-size: 14px; color: #1e293b; line-height: 1.5;">
                👉 <b>What to tap in Trading 212:</b><br>
                Open Trading 212 ➔ Search <b>"{bo['search_symbol']}"</b> ➔ Tap <b>Buy</b> ➔ Enter <b>£{bo['estimated_gbp']:,.0f}</b> (approx {bo['shares']:,.1f} shares at ~{p_str}).
              </div>

              <div style="font-size: 13px; color: #14532d; line-height: 1.4;">
                💡 <b>Why:</b> {bo['plain_english_why']}
              </div>
            </div>
            """
            step_number += 1

        # 3. Hold Card
        hold_names = ", ".join([h.get("name", h["ticker"]) for h in hold_items]) if hold_items else "All other existing holdings"
        action_cards_html += f"""
        <div style="background-color: #f8fafc; border: 2px solid #e2e8f0; border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;">
          <div style="font-size: 16px; font-weight: 800; color: #334155; margin-bottom: 6px;">
            STEP {step_number}: 🟡 EVERYTHING ELSE — DO NOTHING (RELAX)
          </div>
          <div style="font-size: 13px; color: #475569; line-height: 1.5;">
            👉 <b>Companies to leave alone:</b> {hold_names}.<br>
            💡 <b>Why:</b> All of these are performing well in healthy upward compound trends. Sit back and let your profits compound!
          </div>
        </div>
        """

        # Build Thematic Exposure Boxes
        thematic_breakdown = rebalance_plan.get("thematic_breakdown", [])
        thematic_items_html = ""
        for tb in thematic_breakdown:
            thematic_items_html += f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px; display: flex; justify-content: space-between; align-items: center;">
              <div>
                <b style="font-size: 13px; color: #0f172a;">{tb['cluster']}</b>
                <div style="font-size: 11px; color: #64748b;">£{tb['value_gbp']:,.0f} ({tb['weight_pct']}%)</div>
              </div>
              <div>
                <span style="background: {tb['badge_bg']}; color: #ffffff; font-size: 10px; font-weight: 800; padding: 3px 8px; border-radius: 9999px;">
                  {tb['status_tag']}
                </span>
              </div>
            </div>
            """

        thematic_dashboard_html = f"""
        <div style="margin-top: 24px; border-top: 1px solid #e2e8f0; padding-top: 16px;">
          <details style="cursor: pointer;" open>
            <summary style="font-size: 15px; font-weight: 800; color: #1e293b; padding: 6px 0;">
              ⚖️ Modern Portfolio Theory (MPT) Thematic Concentration Dashboard
            </summary>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-top: 12px;">
              {thematic_items_html}
            </div>
          </details>
        </div>
        """

        # Build 100% Harmonized Summary Table Rows
        rows_html = ""
        dossiers_html = ""

        for item in table_items:
            ticker = item.get("ticker", "N/A")
            name = item.get("name", ticker)
            val_gbp = item.get("value_gbp", 0.0)
            item_ppl = item.get("unrealized_pnl", 0.0)
            is_new = item.get("is_new_addition", False)
            res = item.get("research", {})
            theme_tag = item.get("theme", res.get("theme", "Core Holding"))

            status_str = item.get("plan_status", "HOLD")
            badge_color = item.get("plan_badge_color", "#6366f1")
            summary_str = item.get("plan_summary", item.get("plain_english_why", ""))

            val_display = f"£{val_gbp:,.0f}" if not is_new else "<span style='color: #8b5cf6; font-weight: 700;'>New Entry</span>"
            if is_new:
                ppl_display = "—"
                item_ppl_color = "#64748b"
            else:
                item_ppl_color = "#16a34a" if item_ppl >= 0 else "#dc2626"
                item_ppl_sign = "+" if item_ppl > 0 else ""
                ppl_display = f"{item_ppl_sign}£{item_ppl:,.0f}"

            rows_html += f"""
            <tr style="border-bottom: 1px solid #e2e8f0;">
              <td style="padding: 10px 8px; font-weight: 700; color: #0f172a;">{name} ({ticker})</td>
              <td style="padding: 10px 8px; font-size: 11px; color: #64748b;">{theme_tag}</td>
              <td style="padding: 10px 8px; text-align: right; color: #475569;">{val_display}</td>
              <td style="padding: 10px 8px; text-align: right; color: {item_ppl_color}; font-weight: 700;">{ppl_display}</td>
              <td style="padding: 10px 8px; text-align: center;">
                <span style="background-color: {badge_color}; color: #ffffff; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 800; white-space: nowrap;">
                  {status_str}
                </span>
              </td>
              <td style="padding: 10px 8px; font-size: 12px; color: #64748b; line-height: 1.4;">{summary_str}</td>
            </tr>
            """

            # Build PhD-Level Research Dossier Card
            f_pe = f"{res.get('forward_pe', '—')}x" if res.get('forward_pe') else "—"
            peg = f"{res.get('peg_ratio', '—')}" if res.get('peg_ratio') else "—"
            rev_g = f"+{res.get('revenue_growth_yoy', '—')}%" if res.get('revenue_growth_yoy') else "—"
            op_m = f"{res.get('operating_margin_pct', '—')}%" if res.get('operating_margin_pct') else "—"
            roe = f"{res.get('roe_pct', '—')}%" if res.get('roe_pct') else "—"
            fcf = f"${res.get('free_cashflow_b', '—')}B" if res.get('free_cashflow_b') else "—"
            upside = f"+{res.get('implied_upside_pct', 0)}%" if res.get('implied_upside_pct') else "—"
            street_rec = res.get('street_consensus', 'Buy')
            moat = res.get('economic_moat', 'Economic Moat')
            grade = res.get('quality_grade', 'A')

            is_etf = res.get('is_etf', False) or any(e in ticker for e in ["VUAA", "VUSA", "VWRP", "SMGB", "CNDX", "DFND", "QQQ3", "SGLN"])
            p_label = res.get('piotroski_label') or item.get('piotroski_label') or ("N/A (Index ETF Basket)" if is_etf else "N/A (Pending)")
            earn_str = f"{res.get('earnings_date_str', '—')} ({res.get('earnings_catalyst_status', '')})"
            rr_display = f"{item.get('risk_reward_ratio', 2.5)}:1 Asymmetric"
            bull = res.get('bull_thesis', summary_str)
            bear = res.get('bear_risks', 'General macroeconomic tightening or equity multiple contraction.')

            weekly_label = item.get('weekly_trend_label') or res.get('weekly_trend_label', 'Weekly Stage 2 Bull')
            weekly_color = "#10b981" if ("Stage 2" in weekly_label or "Bull" in weekly_label or "Confirmed" in weekly_label or "Support" in weekly_label) else ("#ef4444" if "Stage 4" in weekly_label or "Below" in weekly_label else "#6366f1")

            alt_label = res.get('altman_label') or item.get('altman_label') or ("N/A (Index ETF Basket)" if is_etf else "N/A (Not reported)")
            alt_color = res.get('altman_color') or item.get('altman_color') or "#64748b"

            sloan_label = res.get('sloan_label') or item.get('sloan_label') or ("N/A (Index ETF Basket)" if is_etf else "N/A (Not reported)")
            sloan_color = res.get('sloan_color') or item.get('sloan_color') or "#64748b"

            pead_label = res.get('earnings_streak_label') or item.get('earnings_streak_label') or ("🏛️ Benchmark Basket" if is_etf else "N/A")
            fcf_yield = f"{res.get('fcf_yield_pct')}%" if res.get('fcf_yield_pct') is not None else "—"
            inst_own = f"{res.get('institutional_ownership_pct')}%" if res.get('institutional_ownership_pct') is not None else "—"
            has_div = item.get('bearish_divergence', False)
            div_badge = '<span style="background: #fef3c7; color: #b45309; font-size: 11px; font-weight: 800; padding: 3px 8px; border-radius: 9999px;">⚠️ Bearish Div</span>' if has_div else ''

            item_cagr5 = item.get("cagr_5y_pct", 18.0)
            item_sort5 = item.get("sortino_5y", 1.5)
            item_crash22 = item.get("crash_2022_ret_pct", -15.0)
            item_role = item.get("crash_role_label", "📈 Core Compounder")

            etf_extra_metrics = ""
            if is_etf:
                cat = res.get('category', 'Equity Index ETF')
                aum = f"${res.get('total_assets_b', '—')}B" if res.get('total_assets_b') else "—"
                yld = f"{res.get('yield_pct', '—')}%" if res.get('yield_pct') else "—"
                etf_extra_metrics = f"""
                <div><span style="color: #64748b;">ETF Category:</span> <b style="color: #0f172a;">{cat}</b></div>
                <div><span style="color: #64748b;">Fund AUM:</span> <b style="color: #0f172a;">{aum}</b></div>
                <div><span style="color: #64748b;">Dividend Yield:</span> <b style="color: #0f172a;">{yld}</b></div>
                """

            # Live News Wire formatting
            live_news = res.get("live_news", []) or item.get("live_news", [])
            live_news_html = ""
            if live_news:
                news_items = ""
                for article in live_news[:2]:
                    n_title = article.get("title", "")
                    n_pub = article.get("publisher", "Financial Wire")
                    n_date = article.get("pub_date", "")
                    n_url = article.get("url", "")
                    n_sum = article.get("summary", "")
                    title_elem = f'<a href="{n_url}" target="_blank" style="color: #0284c7; text-decoration: none; font-weight: 700;">{n_title} ↗</a>' if n_url else f'<span style="font-weight: 700; color: #0f172a;">{n_title}</span>'
                    sum_elem = f'<div style="color: #475569; font-size: 11px; margin-top: 2px;">{n_sum[:130]}...</div>' if n_sum else ''
                    news_items += f"""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 10px; margin-top: 6px;">
                      <div style="display: flex; justify-content: space-between; align-items: center; font-size: 10px; color: #64748b; margin-bottom: 2px;">
                        <span>📰 <b>{n_pub}</b></span>
                        <span>{n_date}</span>
                      </div>
                      <div style="font-size: 12px; line-height: 1.35;">{title_elem}</div>
                      {sum_elem}
                    </div>
                    """
                live_news_html = f"""
                <div style="margin-top: 10px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px;">
                  <div style="font-size: 11px; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 0.03em;">
                    📰 Live Market Wires & Real-Time Catalysts
                  </div>
                  {news_items}
                </div>
                """

            dossiers_html += f"""
            <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; padding: 16px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
              <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px; margin-bottom: 10px; flex-wrap: wrap; gap: 6px;">
                <div>
                  <span style="font-size: 15px; font-weight: 800; color: #0f172a;">{name} ({ticker})</span>
                  <span style="background: #e0f2fe; color: #0369a1; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 6px; margin-left: 8px;">{moat}</span>
                </div>
                <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                  {div_badge}
                  <span style="background: #dcfce7; color: #166534; font-size: 11px; font-weight: 800; padding: 3px 8px; border-radius: 9999px;">{item_role}</span>
                  <span style="background: #ecfdf5; color: #047857; font-size: 11px; font-weight: 800; padding: 3px 8px; border-radius: 9999px;">Grade: {grade}</span>
                  <span style="background: {weekly_color}; color: #ffffff; font-size: 11px; font-weight: 800; padding: 3px 8px; border-radius: 9999px;">{weekly_label.split('(')[0].strip()}</span>
                  <span style="background: {badge_color}; color: #ffffff; font-size: 11px; font-weight: 800; padding: 3px 8px; border-radius: 9999px;">{status_str}</span>
                </div>
              </div>

              <!-- Metrics Grid -->
              <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; background: #f8fafc; border-radius: 8px; padding: 12px; margin-bottom: 10px; font-size: 11px;">
                <div><span style="color: #64748b;">5Y Hist. CAGR:</span> <b style="color: #059669;">{item_cagr5:+.1f}%</b></div>
                <div><span style="color: #64748b;">5Y Sortino Ratio:</span> <b style="color: #0284c7;">{item_sort5:.2f}</b></div>
                <div><span style="color: #64748b;">2022 Crash Return:</span> <b style="color: {'#059669' if item_crash22 >= 0 else '#dc2626'};">{item_crash22:+.1f}%</b></div>
                <div><span style="color: #64748b;">Minervini Trend:</span> <b style="color: #059669;">{item.get('minervini_stage', 'Stage 2 Bull')}</b></div>
                <div><span style="color: #64748b;">Weekly 30w Trend:</span> <b style="color: {weekly_color};">{weekly_label}</b></div>
                <div><span style="color: #64748b;">Volume Profile:</span> <b style="color: #0284c7;">{item.get('volume_footprint_label', 'Constructive')}</b></div>
                <div><span style="color: #64748b;">Altman Z Solvency:</span> <b style="color: {alt_color};">{alt_label}</b></div>
                <div><span style="color: #64748b;">Sloan Cash Quality:</span> <b style="color: {sloan_color};">{sloan_label}</b></div>
                <div><span style="color: #64748b;">Piotroski Quality:</span> <b style="color: #059669;">{p_label}</b></div>
                <div><span style="color: #64748b;">Chandelier Stop:</span> <b style="color: #dc2626;">{format_price(ticker, item.get('chandelier_stop', item.get('trailing_stop', 0.0)), item.get('currency', res.get('currency', 'USD')))}</b></div>
                <div><span style="color: #64748b;">Risk/Reward:</span> <b style="color: #0284c7;">{rr_display}</b></div>
                <div><span style="color: #64748b;">Multi-TF Alpha:</span> <b style="color: #16a34a;">{item.get('rs_label', 'Benchmark Neutral')}</b></div>
                <div><span style="color: #64748b;">Earnings Beat Streak:</span> <b style="color: #0284c7;">{pead_label}</b></div>
                <div><span style="color: #64748b;">Next Earnings:</span> <b style="color: #0f172a;">{earn_str}</b></div>
                <div><span style="color: #64748b;">Rule of 40:</span> <b style="color: #0f172a;">{res.get('rule_of_40_label', 'N/A')}</b></div>
                {etf_extra_metrics}
                <div><span style="color: #64748b;">Fwd P/E:</span> <b style="color: #0f172a;">{f_pe}</b></div>
                <div><span style="color: #64748b;">PEG Ratio:</span> <b style="color: #0f172a;">{peg}</b></div>
                <div><span style="color: #64748b;">FCF Yield:</span> <b style="color: #0f172a;">{fcf_yield}</b></div>
                <div><span style="color: #64748b;">Inst Ownership:</span> <b style="color: #0f172a;">{inst_own}</b></div>
                <div><span style="color: #64748b;">YoY Rev Growth:</span> <b style="color: #16a34a;">{rev_g}</b></div>
                <div><span style="color: #64748b;">Operating Margin:</span> <b style="color: #0f172a;">{op_m}</b></div>
                <div><span style="color: #64748b;">ROE:</span> <b style="color: #0f172a;">{roe}</b></div>
                <div><span style="color: #64748b;">FCF (TTM):</span> <b style="color: #0f172a;">{fcf}</b></div>
                <div><span style="color: #64748b;">Consensus Target:</span> <b style="color: #2563eb;">{street_rec} ({upside})</b></div>
                <div><span style="color: #64748b;">Wall St Revisions:</span> <b style="color: #059669;">{res.get('eps_revision_label', 'Neutral Consensus')}</b></div>
                <div><span style="color: #64748b;">ROIC Compounding:</span> <b style="color: {res.get('roic_color', '#059669')};">{res.get('roic_label', 'N/A')}</b></div>
                <div><span style="color: #64748b;">SEC Insider Flow:</span> <b style="color: {res.get('insider_badge_color', '#64748b')};">{res.get('insider_label', 'Neutral Insider Activity')}</b></div>
                <div><span style="color: #64748b;">Options Sentiment:</span> <b style="color: {res.get('options_badge_color', '#64748b')};">{res.get('options_pcr_label', 'N/A')}</b></div>
                <div><span style="color: #64748b;">13F Inst Momentum:</span> <b style="color: {res.get('institutional_flow_color', '#64748b')};">{res.get('institutional_flow_label', 'Stable Institutional Float')}</b></div>
              </div>

              <!-- Qualitative PhD Thesis -->
              <div style="font-size: 12px; line-height: 1.5; color: #334155; margin-bottom: 6px;">
                <b style="color: #15803d;">Institutional Bull Thesis:</b> {bull}
              </div>
              <div style="font-size: 12px; line-height: 1.5; color: #991b1b;">
                <b style="color: #b91c1c;">Risk & Defense Assessment:</b> {bear}
              </div>

              <!-- Live Market News & Real-Time Catalysts -->
              {live_news_html}
            </div>
            """

        acc_5y = rebalance_plan.get("portfolio_5y_risk_profile", {})
        acc_5y_html = ""
        if acc_5y:
            acc_5y_html = f"""
          <!-- 5-Year Horizon Crash-Protection & Stress-Test Panel -->
          <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 14px 18px; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px; margin-bottom: 10px;">
              <div style="font-size: 13px; font-weight: 800; color: #166534;">
                🛡️ {acc_name}'s 5-Year Horizon Crash-Protection & Compounding Profile (2026–2031)
              </div>
              <span style="background: #ffffff; border: 1px solid #86efac; color: #15803d; font-size: 11px; font-weight: 800; padding: 2px 10px; border-radius: 9999px;">
                {acc_5y.get('crash_readiness_grade', '🟢 ALL-WEATHER READY')}
              </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 10px; font-size: 12px;">
              <div style="background: #ffffff; border: 1px solid #dcfce7; border-radius: 8px; padding: 8px 10px;">
                <span style="color: #64748b; font-size: 10px; text-transform: uppercase; font-weight: 700;">5Y Sortino Ratio</span><br>
                <b style="color: #15803d; font-size: 14px;">{acc_5y.get('current_sortino_5y', 1.4):.2f} ➔ {acc_5y.get('post_sortino_5y', 1.5):.2f}</b>
              </div>
              <div style="background: #ffffff; border: 1px solid #dcfce7; border-radius: 8px; padding: 8px 10px;">
                <span style="color: #64748b; font-size: 10px; text-transform: uppercase; font-weight: 700;">2022 Crash Stress-Test</span><br>
                <b style="color: #dc2626; font-size: 14px;">{acc_5y.get('current_stress_2022_pct', -25.0):+.1f}% ➔ {acc_5y.get('post_stress_2022_pct', -20.0):+.1f}%</b>
              </div>
              <div style="background: #ffffff; border: 1px solid #dcfce7; border-radius: 8px; padding: 8px 10px;">
                <span style="color: #64748b; font-size: 10px; text-transform: uppercase; font-weight: 700;">All-Weather Shield</span><br>
                <b style="color: #0284c7; font-size: 14px;">{acc_5y.get('current_shield_pct', 0.0):.1f}% ➔ {acc_5y.get('post_shield_pct', 0.0):.1f}%</b>
              </div>
              <div style="background: #ffffff; border: 1px solid #dcfce7; border-radius: 8px; padding: 8px 10px;">
                <span style="color: #64748b; font-size: 10px; text-transform: uppercase; font-weight: 700;">2031 5Y Wealth Est.</span><br>
                <b style="color: #0f172a; font-size: 13px;">£{acc_5y.get('projected_5y_conservative_gbp', 0.0):,.0f}–£{acc_5y.get('projected_5y_target_gbp', 0.0):,.0f}</b>
              </div>
            </div>
          </div>
            """

        account_sections_html += f"""
        <!-- User Section Container -->
        <div style="background-color: #ffffff; border-radius: 16px; border: 1px solid #cbd5e1; margin-bottom: 28px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
          
          <!-- Section Title & Profit -->
          <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f1f5f9; padding-bottom: 16px; margin-bottom: 20px;">
            <div>
              <div style="display: inline-block; background: #f1f5f9; color: #334155; font-size: 11px; font-weight: 800; text-transform: uppercase; padding: 2px 8px; border-radius: 4px; margin-bottom: 4px; border: 1px solid #cbd5e1;">
                Trading 212 Account: {acc_name} • Capital Ring-Fenced
              </div>
              <h2 style="margin: 0; font-size: 20px; font-weight: 800; color: #0f172a;">👤 {acc_name}'s Account Actions</h2>
              <div style="font-size: 13px; color: #64748b; margin-top: 3px;">Execute instructions strictly within {acc_name}'s Trading 212 account</div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Total Profit to Date</div>
              <b style="color: {ppl_color}; font-size: 22px; font-weight: 900;">{ppl_sign}£{ppl:,.0f}</b>
            </div>
          </div>

          <!-- Cash Math Box -->
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px;">
            <div>
              <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Uninvested Cash</div>
              <div style="font-size: 17px; font-weight: 800; color: #0f172a; margin-top: 2px;">{cash_info.get('free_label', f'£{avail_cash:,.2f}')}</div>
            </div>
            <div>
              <div style="font-size: 11px; text-transform: uppercase; color: #d97706; font-weight: 700;">⏳ Pending Committed</div>
              <div style="font-size: 17px; font-weight: 800; color: #b45309; margin-top: 2px;">£{blocked_cash:,.2f}</div>
            </div>
            <div>
              <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Cash From Selling</div>
              <div style="font-size: 17px; font-weight: 800; color: #dc2626; margin-top: 2px;">+£{sells_cash:,.0f}</div>
            </div>
            <div>
              <div style="font-size: 11px; text-transform: uppercase; color: #0284c7; font-weight: 700;">🛡️ Protected Reserve</div>
              <div style="font-size: 17px; font-weight: 800; color: #0369a1; margin-top: 2px;">£{buffer_cash:,.0f} <span style="font-size: 10px; color: #0284c7; font-weight: 600;">(Withdraw Safe)</span></div>
            </div>
            <div>
              <div style="font-size: 11px; text-transform: uppercase; color: #16a34a; font-weight: 700;">Available to Deploy</div>
              <div style="font-size: 17px; font-weight: 800; color: #16a34a; margin-top: 2px;">£{net_budget:,.0f}</div>
            </div>
          </div>

          {acc_5y_html}

          <!-- 2-Minute Action Cards -->
          <h3 style="margin: 0 0 14px 0; font-size: 17px; font-weight: 800; color: #0f172a;">⚡ What To Do in Trading 212 Today</h3>
          {action_cards_html}

          <!-- Thematic Concentration Dashboard -->
          {thematic_dashboard_html}

          <!-- Harmonized Summary Table (Collapsible) -->
          <div style="margin-top: 28px; border-top: 1px solid #e2e8f0; padding-top: 20px;">
            <details style="cursor: pointer;" open>
              <summary style="font-size: 15px; font-weight: 800; color: #1e293b; padding: 8px 0;">
                📊 Portfolio Overview & Harmonized Decision Table ({len(table_items)} assets)
              </summary>
              <div style="overflow-x: auto; margin-top: 12px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: left;">
                  <thead>
                    <tr style="background-color: #f8fafc; color: #64748b; font-size: 11px; text-transform: uppercase;">
                      <th style="padding: 8px;">Company</th>
                      <th style="padding: 8px;">Theme</th>
                      <th style="padding: 8px; text-align: right;">Total Value</th>
                      <th style="padding: 8px; text-align: right;">Profit</th>
                      <th style="padding: 8px; text-align: center;">Decision</th>
                      <th style="padding: 8px;">Strategic Rationale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows_html}
                  </tbody>
                </table>
              </div>
            </details>
          </div>

          <!-- Institutional PhD Research Dossiers (Collapsible) -->
          <div style="margin-top: 24px; border-top: 1px solid #e2e8f0; padding-top: 20px;">
            <details style="cursor: pointer;">
              <summary style="font-size: 15px; font-weight: 800; color: #0f766e; padding: 8px 0;">
                🔬 Institutional Deep Research Dossiers & PhD Factor Models (Tap to view full analysis)
              </summary>
              <div style="margin-top: 14px;">
                {dossiers_html}
              </div>
            </details>
          </div>

        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>What You Need To Do Today</title>
    </head>
    <body style="margin: 0; padding: 20px 12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f1f5f9; color: #0f172a;">
      <div style="max-width: 760px; margin: 0 auto;">
        
        <!-- Header Banner -->
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #ffffff; padding: 24px; border-radius: 16px; margin-bottom: 20px;">
          <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; color: #38bdf8; font-weight: 800;">TRADING 212 DAILY ACTION GUIDE</div>
          <h1 style="margin: 4px 0 6px 0; font-size: 24px; font-weight: 900; color: #ffffff;">⚡ What You Need To Do Today</h1>
          <div style="font-size: 13px; color: #94a3b8;">📅 {date_str} • {" & ".join(results_by_account.keys()) or "Portfolio"}</div>
        </div>

        <!-- ⚡ 30-SECOND TL;DR — WHAT TO DO TODAY -->
        {tldr_section_html}

        <!-- Consolidated Portfolio Dashboard -->
        {consolidated_overview_html}

        <!-- Account Section -->
        {account_sections_html}

        <!-- Global Macro & Intermarket Barometer (Collapsible reference) -->
        <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 14px; padding: 16px 20px; margin-bottom: 22px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
          <details style="cursor: pointer;">
            <summary style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 15px; font-weight: 800; color: #0f172a; padding: 4px 0;">
              <span>🌍 Global Macro & Intermarket Barometer (Tap to view details)</span>
              <span style="background: #ecfdf5; color: #047857; font-size: 12px; font-weight: 800; padding: 4px 12px; border-radius: 9999px;">
                Macro Score: {macro_regime.get('macro_score', 75)}/100 • {macro_regime.get('risk_climate', '🟢 RISK-ON')}
              </span>
            </summary>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin: 14px 0 12px 0; border-top: 1px solid #f1f5f9; padding-top: 12px; font-size: 12px;">
              <div style="background: #f8fafc; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0;">
                <span style="color: #64748b; font-size: 11px;">VIX Volatility</span><br>
                <b style="color: #0f172a; font-size: 14px;">{float(macro_regime.get('vix') or 15.0):.1f}</b> <span style="font-size: 11px; color: #059669;">({macro_regime.get('vix_status', 'Low')})</span>
              </div>
              <div style="background: #f8fafc; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0;">
                <span style="color: #64748b; font-size: 11px;">Yield Curve (10Y-2Y)</span><br>
                <b style="color: #0f172a; font-size: 14px;">{float(macro_regime.get('yield_curve_spread') or 0.70):+.2f}%</b> <span style="font-size: 11px; color: #0284c7;">({macro_regime.get('yield_curve_status', 'Normal')})</span>
              </div>
              <div style="background: #f8fafc; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0;">
                <span style="color: #64748b; font-size: 11px;">US Dollar Index (DXY)</span><br>
                <b style="color: #0f172a; font-size: 14px;">{float(macro_regime.get('dxy_index') or 100.0):.1f}</b> <span style="font-size: 11px; color: #059669;">({macro_regime.get('dxy_status', 'Neutral')})</span>
              </div>
              <div style="background: #f8fafc; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0;">
                <span style="color: #64748b; font-size: 11px;">Credit Risk (HYG/LQD)</span><br>
                <b style="color: #0f172a; font-size: 14px;">{float(macro_regime.get('credit_spread_ratio') or 0.75):.3f}</b> <span style="font-size: 11px; color: #059669;">({macro_regime.get('credit_status', 'Tight Spreads')})</span>
              </div>
              <div style="background: #f8fafc; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0;">
                <span style="color: #64748b; font-size: 11px;">Copper/Gold Barometer</span><br>
                <b style="color: #0f172a; font-size: 14px;">{float(macro_regime.get('copper_gold_ratio') or 1.50):.2f}</b> <span style="font-size: 11px; color: #0284c7;">({macro_regime.get('copper_gold_status', 'Expansion')})</span>
              </div>
              <div style="background: #f8fafc; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0;">
                <span style="color: #64748b; font-size: 11px;">Market Breadth (RSP/SPY)</span><br>
                <b style="color: #0f172a; font-size: 14px;">{float(macro_regime.get('breadth_ratio') or 0.28):.3f}</b> <span style="font-size: 11px; color: #059669;">({macro_regime.get('breadth_status', 'Broad')})</span>
              </div>
              <div style="background: #f8fafc; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0;">
                <span style="color: #64748b; font-size: 11px;">VIX Term Structure</span><br>
                <b style="color: #0f172a; font-size: 14px;">{float(macro_regime.get('vix_term_structure') or 0.87):.2f}x</b> <span style="font-size: 11px; color: #059669;">({macro_regime.get('vix_term_structure_status', 'Contango').split('—')[0].strip()})</span>
              </div>
              <div style="background: #f8fafc; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0;">
                <span style="color: #64748b; font-size: 11px;">Real Rates (TIP/IEF)</span><br>
                <b style="color: #0f172a; font-size: 14px;">{float(macro_regime.get('real_rate_ratio') or 1.16):.3f}</b> <span style="font-size: 11px; color: #059669;">({macro_regime.get('real_rate_status', 'Easing').split('—')[0].strip()})</span>
              </div>
              <div style="background: #f8fafc; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0;">
                <span style="color: #64748b; font-size: 11px;">Inst. Distribution (SPY/QQQ)</span><br>
                <b style="color: #0f172a; font-size: 14px;">{int(macro_regime.get('spy_dist_days') or 0)}d / {int(macro_regime.get('qqq_dist_days') or 0)}d</b> <span style="font-size: 11px; color: #059669;">({macro_regime.get('dist_status', 'Low')})</span>
              </div>
              <div style="background: #f8fafc; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0;">
                <span style="color: #64748b; font-size: 11px;">High-Beta / Low-Vol</span><br>
                <b style="color: #0f172a; font-size: 14px;">{float(macro_regime.get('high_beta_ratio') or 2.15):.2f}x</b> <span style="font-size: 11px; color: #059669;">({macro_regime.get('high_beta_status', 'Risk-On').split('—')[0].replace('🟢', '').replace('🟡', '').replace('🔴', '').strip()})</span>
              </div>
              <div style="background: #f8fafc; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0;">
                <span style="color: #64748b; font-size: 11px;">Semi Canary (SMH/SPY)</span><br>
                <b style="color: #0f172a; font-size: 14px;">{float(macro_regime.get('semi_leadership_ratio') or 0.80):.3f}x</b> <span style="font-size: 11px; color: #059669;">({macro_regime.get('semi_leadership_status', 'Leading').split('—')[0].replace('🟢', '').replace('🟡', '').replace('🔴', '').strip()})</span>
              </div>
            </div>

            <div style="font-size: 13px; color: #334155; line-height: 1.5; background: #f1f5f9; border-radius: 8px; padding: 10px 14px;">
              💡 <b>Regime Synthesis:</b> {macro_regime.get('regime_summary', 'Favorable conditions to compound money.')}
            </div>
          </details>
        </div>

        <!-- Footer -->
        <div style="text-align: center; margin-top: 24px; font-size: 12px; color: #94a3b8; line-height: 1.5;">
          Automated Trading 212 Daily Recommendation & Rebalancing Advisor.<br>
          Institutional modeling distilled into clean, 2-minute daily execution.
        </div>
      </div>
    </body>
    </html>
    """
    return html


def send_email_digest(
    html_content: str,
    subject_prefix: str = "⚡ What You Need To Do Today",
    plain_text_summary: Optional[str] = None
) -> bool:
    """Dispatches the daily digest to all configured alert emails via SMTP with full RFC MIME compliance."""
    if not SMTP_USER or not SMTP_PASS or not RECIPIENT_EMAILS:
        logger.warning("SMTP credentials or recipient emails not configured. Email skipped.")
        return False

    now_dt = datetime.now()
    now_str = now_dt.strftime("%d %b %Y")
    subject = f"{subject_prefix} — {now_str}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Trading 212 Assistant <{SMTP_USER}>"
    msg["To"] = ", ".join(RECIPIENT_EMAILS)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="gmail.com")
    msg["Reply-To"] = SMTP_USER
    msg["X-Mailer"] = "Trading212-Notifier/2.0"
    msg["Auto-Submitted"] = "auto-generated"

    # 1. Plain-text alternative (essential for deliverability and spam avoidance)
    if not plain_text_summary:
        plain_text_summary = (
            f"⚡ WHAT YOU NEED TO DO TODAY — {now_str}\n\n"
            "Please view this email in an HTML-compatible client for full interactive tables and research dossiers."
        )

    part_text = MIMEText(plain_text_summary, "plain", "utf-8")
    part_html = MIMEText(html_content, "html", "utf-8")

    # RFC 2046 standard: plain text part first, HTML part second
    msg.attach(part_text)
    msg.attach(part_html)

    try:
        logger.info("Connecting to SMTP server %s:%s...", SMTP_HOST, SMTP_PORT)
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20)
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, RECIPIENT_EMAILS, msg.as_string())
        server.quit()
        logger.info("Daily digest successfully dispatched to: %s", ", ".join(RECIPIENT_EMAILS))
        return True
    except Exception as e:
        logger.error("Failed to dispatch daily digest email: %s", e)
        return False

#!/usr/bin/env python3
"""
Simple withdrawal + federal tax estimator for 2026 (single filers).

Input CSV columns (header):
 ticker, account_type, shares, price, market_value, cost_basis, cap_gain_method,
 annual_yield_pct, dividend_type

Notes:
 - account_type: one of 'deferred', 'brokerage', 'roth'
 - You must provide either (shares and price) OR market_value.
 - annual_yield_pct: e.g. 3.5 for 3.5% annual yield
 - dividend_type (only used for brokerage): 'qualified', 'ordinary', or 'interest'
   (qualified dividends are taxed at capital gains preferential rates)
"""

import csv
from pathlib import Path
from typing import List, Dict

# ---------- 2026 single-filer tax parameters (IRS Rev Proc 2025-32) ----------
STANDARD_DEDUCTION = 16100.0  # single filer for 2026. Source: IRS Rev Proc. :contentReference[oaicite:2]{index=2}

# Ordinary tax brackets for single filers (2026) as (upper_bound, marginal_rate)
# Note: upper_bound uses the bracket top; last bracket uses None for infinity.
ORDINARY_BRACKETS = [
    (12400.0, 0.10),
    (50400.0, 0.12),
    (105700.0, 0.22),
    (201775.0, 0.24),
    (256225.0, 0.32),
    (640600.0, 0.35),
    (None, 0.37),
]

# Long-term capital gains / qualified dividend thresholds (single filers) for 2026.
# The thresholds are taxable-income amounts where 0% and 15% caps end.
# Source: NerdWallet / IRS Rev Proc summaries. :contentReference[oaicite:3]{index=3}
CG_0pct_limit_single = 49450.0
CG_15pct_limit_single = 545500.0

# ---------- Helper functions ----------
def compute_market_value(row: Dict[str,str]) -> float:
    mv = row.get("market_value", "").strip()
    if mv:
        return float(mv)
    shares = row.get("shares", "").strip()
    price = row.get("price", "").strip()
    if shares and price:
        return float(shares) * float(price)
    raise ValueError(f"Row for ticker {row.get('ticker','?')} must contain market_value or shares+price.")

def parse_portfolio_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def ordinary_tax_on_amount(taxable_amount: float) -> float:
    """
    Compute ordinary tax using ORDINARY_BRACKETS.
    taxable_amount assumed to be >= 0 (already after standard deduction and other adjustments).
    """
    if taxable_amount <= 0:
        return 0.0
    tax = 0.0
    lower = 0.0
    for upper, rate in ORDINARY_BRACKETS:
        if upper is None:
            taxable_here = taxable_amount - lower
        else:
            taxable_here = max(0.0, min(taxable_amount, upper) - lower)
        if taxable_here > 0:
            tax += taxable_here * rate
        if upper is None or taxable_amount <= (upper if upper is not None else taxable_amount):
            break
        lower = upper
    return tax

def capital_gains_tax_for_qualified(q_amount: float, ordinary_income_after_deduction: float) -> float:
    """
    Compute tax on qualified dividends (q_amount) using capital gains brackets.
    The algorithm fills the 0% band first (capacity = CG_0pct_limit_single - ordinary_income_after_deduction),
    then the 15% band (capacity = CG_15pct_limit_single - (ordinary + already_filled)),
    remainder taxed at 20%.
    """
    if q_amount <= 0:
        return 0.0

    tax = 0.0
    remaining = q_amount

    # Determine how much of the 0% band is available given ordinary income that occupies part of taxable income.
    cap0_space = max(0.0, CG_0pct_limit_single - ordinary_income_after_deduction)
    take0 = min(remaining, cap0_space)
    tax += take0 * 0.0
    remaining -= take0

    # Now 15% band
    # The top of the 15% band is CG_15pct_limit_single. The already occupied amount is ordinary_income_after_deduction + take0
    already = ordinary_income_after_deduction + take0
    cap15_space = max(0.0, CG_15pct_limit_single - already)
    take15 = min(remaining, cap15_space)
    tax += take15 * 0.15
    remaining -= take15

    # Remainder at 20%
    if remaining > 0:
        tax += remaining * 0.20

    return tax

# ---------- Main estimator ----------
def estimate_from_rows(rows: List[Dict]) -> Dict:
    """
    rows: list of dicts as parsed from CSV (string values)
    Returns a dict with annual and monthly totals and estimated federal tax.
    """
    # Accumulators
    annual_by_account = {"deferred": 0.0, "brokerage": 0.0, "roth": 0.0}
    brokerage_qualified = 0.0
    brokerage_ordinary = 0.0  # non-qualified dividends and interest from brokerage
    # We'll treat deferred yields as ordinary taxable income when withdrawn

    for r in rows:
        ticker = r.get("ticker","?")
        account = r.get("account_type","").strip().lower()
        if account not in ("deferred","brokerage","roth"):
            raise ValueError(f"account_type must be one of deferred, brokerage, roth. Problem row ticker={ticker}")

        mv = compute_market_value(r)
        try:
            annual_yield_pct = float(r.get("annual_yield_pct","0") or 0.0)
        except:
            annual_yield_pct = 0.0
        annual_income = mv * (annual_yield_pct / 100.0)

        annual_by_account[account] += annual_income

        if account == "brokerage":
            dtype = r.get("dividend_type","ordinary").strip().lower()
            if dtype == "qualified":
                brokerage_qualified += annual_income
            else:
                # treat 'ordinary' or 'interest' or other as taxable ordinary income
                brokerage_ordinary += annual_income

    # Totals
    total_annual = sum(annual_by_account.values())
    total_monthly = total_annual / 12.0

    # Taxable ordinary income: deferred + brokerage_ordinary
    taxable_ordinary = annual_by_account["deferred"] + brokerage_ordinary

    # Step 1: apply standard deduction against ordinary income first
    ordinary_after_deduction = max(0.0, taxable_ordinary - STANDARD_DEDUCTION)

    # The standard deduction reduces ordinary income; if it completely wipes ordinary income, the remaining deduction
    # is NOT carried to reduce qualified dividends in typical ordering (qualified dividends use the remaining standard deduction
    # capacity effectively via the ordinary income reduction). Our approach (ordinary_after_deduction) matches that effect.

    # Step 2: ordinary tax on ordinary_after_deduction
    ordinary_tax = ordinary_tax_on_amount(ordinary_after_deduction)

    # Step 3: compute capital gains (qualified dividends) tax.
    # Note that qualified dividends are included in taxable income for determining thresholds,
    # but the preferential rate is applied as allocated based on ordinary_after_deduction as done in capital_gains_tax_for_qualified.
    capg_tax = capital_gains_tax_for_qualified(brokerage_qualified, ordinary_after_deduction)

    # Combined federal tax (approx)
    federal_tax = ordinary_tax + capg_tax

    # Effective tax rate on taxable portion of withdrawals (excluding roth which is tax-free)
    taxable_withdrawals_annual = annual_by_account["deferred"] + brokerage_ordinary + brokerage_qualified
    effective_rate = (federal_tax / taxable_withdrawals_annual) if taxable_withdrawals_annual > 0 else 0.0

    return {
        "annual_by_account": annual_by_account,
        "brokerage_qualified": brokerage_qualified,
        "brokerage_ordinary": brokerage_ordinary,
        "total_annual_withdrawal": total_annual,
        "total_monthly_withdrawal": total_monthly,
        "taxable_withdrawals_annual": taxable_withdrawals_annual,
        "ordinary_tax": ordinary_tax,
        "capital_gains_tax": capg_tax,
        "estimated_federal_tax": federal_tax,
        "effective_tax_rate_on_taxable_withdrawals": effective_rate,
        "standard_deduction_used": min(STANDARD_DEDUCTION, taxable_ordinary),
        "ordinary_after_deduction": ordinary_after_deduction,
    }

# ---------- CLI example usage ----------
def print_report(res: Dict):
    print("\n=== Withdrawal & Federal Tax Estimate (2026, single filer) ===\n")
    print(f"Annual withdrawals by account:")
    for a,v in res["annual_by_account"].items():
        print(f"  {a:9s}: ${v:,.2f}")
    print(f"\nTotal annual withdrawal (all accounts): ${res['total_annual_withdrawal']:,.2f}")
    print(f"Total monthly withdrawal: ${res['total_monthly_withdrawal']:,.2f}")
    print(f"\nTaxable annual withdrawals (deferred + brokerage): ${res['taxable_withdrawals_annual']:,.2f}")
    print(f"  Brokerage: qualified = ${res['brokerage_qualified']:,.2f}, ordinary/interest = ${res['brokerage_ordinary']:,.2f}")
    print(f"\nStandard deduction (single, 2026): ${STANDARD_DEDUCTION:,.2f}")
    print(f"Ordinary income after standard deduction: ${res['ordinary_after_deduction']:,.2f}")
    print(f"Estimated ordinary tax: ${res['ordinary_tax']:,.2f}")
    print(f"Estimated tax on qualified dividends (cap gains rates): ${res['capital_gains_tax']:,.2f}")
    print(f"\nEstimated federal tax on the withdrawals (annual): ${res['estimated_federal_tax']:,.2f}")
    print(f"Effective federal tax rate on taxable withdrawals: {res['effective_tax_rate_on_taxable_withdrawals']*100:.2f}%")
    print("\n(Notes: roth withdrawals are assumed tax-free; capital gains thresholds used for qualified dividends.\nThis is an estimate only; consult a tax professional for detailed planning.)\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Estimate withdrawal amounts from portfolio yields and 2026 federal tax (single).")
    p.add_argument("csvfile", help="CSV file with portfolio rows (see header description in script).")
    args = p.parse_args()

    rows = parse_portfolio_csv(args.csvfile)
    res = estimate_from_rows(rows)
    print_report(res)


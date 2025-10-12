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
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

import yfinance as yf

CACHE_FILENAME = "ticker_data_cache.json"
CACHE_TTL = 7 * 24 * 3600  # seconds — how “fresh” a cache entry may be (one week)

def load_cache() -> Dict[str, Dict]:
    if Path(CACHE_FILENAME).exists():
        with open(CACHE_FILENAME, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_cache(cache, filename="cache.json"):
    def convert(o):
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, dict):
            return {str(k): convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        return o

    with open(filename, "w") as f:
        json.dump(convert(cache), f, indent=2)

def fetch_ticker_data(ticker: str) -> Dict:
    """
    Fetch price, yield, dividend history, etc. for a ticker, via yfinance.
    Returns a dict with keys: price, trailing_yield_pct, dividend_history (pandas Series or list), ...
    """
    tk = yf.Ticker(ticker)
    info = tk.info  # dictionary of metadata
    # price: try to get current price
    price = info.get("regularMarketPrice") or info.get("previousClose") or info.get("currentPrice")

    # trailing annual dividend yield (fraction)
    # yfinance’s `info` may include 'trailingAnnualDividendYield' (a fraction, e.g. 0.015)
    trailing_yield_frac = info.get("trailingAnnualDividendYield")
    trailing_yield_pct = None
    if trailing_yield_frac is not None:
        trailing_yield_pct = trailing_yield_frac * 100.0

    # dividend history (past dividends)
    # Use tk.dividends, which is a pandas Series indexed by date → dividend per share
    divhist = None
    try:
        divhist = tk.dividends  # a pandas Series
    except Exception as e:
        divhist = None

    return {
        "price": price,
        "trailing_yield_pct": trailing_yield_pct,
        "dividend_history": divhist.to_dict() if divhist is not None else None,
        "fetched_time": time.time(),
    }

def get_ticker_info(ticker: str, cache: Dict[str, Dict], force_refresh: bool = False) -> Dict:
    """
    Return the ticker info, using cache or fetch if needed.
    """
    rec = cache.get(ticker)
    if rec and not force_refresh:
        age = time.time() - rec.get("fetched_time", 0)
        if age < CACHE_TTL:
            return rec
    # Otherwise fetch fresh
    new = fetch_ticker_data(ticker)
    cache[ticker] = new
    return new

def infer_dividend_type(divhist: Optional[Dict]) -> str:
    """
    Very naive inference: if historical dividends exist, assume 'qualified' by default.
    If none or erratic, fallback to 'ordinary'.
    You could strengthen this by checking fund/ticker documentation or whether dividends are ordinary-income sources.
    """
    if divhist:
        return "qualified"
    return "ordinary"
	

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
    with open(path, newline="") as f:
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
def estimate_with_lookup(rows: List[Dict], cache: Dict[str, Dict], force_refresh: bool = False):
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
        ticker = r["ticker"].strip().upper()
        acct = r["account_type"].strip().lower()
        shares = float(r["shares"])
        cost_basis = float(r["cost_basis"])
        # For simplicity ignoring cap_gain_method here except store it

        info = get_ticker_info(ticker, cache, force_refresh=force_refresh)
        price = r.get("override_price")
        if price is not None and price.strip() != "":
            price = float(price)
        else:
            price = info.get("price")
        if price is None:
            raise RuntimeError(f"Could not fetch price for {ticker}")

        yield_pct = None
        y_over = r.get("override_yield_pct")
        if y_over is not None and y_over.strip() != "":
            yield_pct = float(y_over)
        else:
            yield_pct = info.get("trailing_yield_pct")
        if yield_pct is None:
            # fallback: if you have dividend_history, you could approximate: sum(dividends) / price
            divhist = info.get("dividend_history")
            if divhist:
                total_div = sum(divhist.values())
                yield_pct = (total_div / price) * 100.0
            else:
                yield_pct = 0.0

        # Infer dividend type unless overridden
        div_type = r.get("override_dividend_type")
        if div_type is None or div_type.strip() == "":
            div_type = infer_dividend_type(info.get("dividend_history"))

        mv = shares * price
        annual_income = mv * (yield_pct / 100.0)
        annual_by_account[acct] = annual_by_account.get(acct, 0.0) + annual_income

        if acct == "brokerage":
            if div_type.strip().lower() == "qualified":
                brokerage_qualified += annual_income
            else:
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
        "cache_used": True,
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

    parser = argparse.ArgumentParser(description="Withdrawal estimator with auto price/yield lookup + cache.")
    parser.add_argument("csvfile", help="CSV file of holdings (ticker,account_type,shares,cost_basis,cap_gain_method, optional overrides)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force refresh of all ticker data (ignore cache).")
    args = parser.parse_args()

    rows = parse_portfolio_csv(args.csvfile)
    cache = load_cache()
    res = estimate_with_lookup(rows, cache, force_refresh=args.refresh)
    save_cache(cache)
	
    print_report(res)


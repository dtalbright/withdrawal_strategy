#!/usr/bin/env python3
"""
Withdrawal and Federal Tax Estimator (2026, Single Filer)

Features:
 - Reads holdings from a JSON file
 - Auto-fetches price and yield via yfinance (with cache)
 - Estimates monthly withdrawals (dividends/interest)
 - Calculates approximate federal tax for 2026 single filers
 - Optionally includes monthly Social Security income

JSON input format:
[
  {
    "ticker": "VTI",
    "account_type": "brokerage",
    "shares": 100,
    "cost_basis": 20000,
    "cap_gain_method": "fifo"
  },
  {
    "ticker": "VBTLX",
    "account_type": "deferred",
    "shares": 300,
    "cost_basis": 30000,
    "cap_gain_method": "average"
  }
]

Account types: "deferred", "brokerage", "roth"
"""

import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import yfinance as yf

CACHE_FILENAME = "ticker_data_cache.json"
CACHE_TTL = 7 * 24 * 3600  # 1 week


# ---------- Cache helpers ----------
def load_cache() -> Dict[str, Dict]:
    if Path(CACHE_FILENAME).exists():
        with open(CACHE_FILENAME, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_cache(cache, filename=CACHE_FILENAME):
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


# ---------- Data fetching ----------
def fetch_ticker_data(ticker: str) -> Dict:
    tk = yf.Ticker(ticker)
    info = tk.info
    price = info.get("regularMarketPrice") or info.get("previousClose") or info.get("currentPrice")

    trailing_yield_frac = info.get("trailingAnnualDividendYield")
    trailing_yield_pct = trailing_yield_frac * 100.0 if trailing_yield_frac is not None else None

    try:
        divhist = tk.dividends
    except Exception:
        divhist = None

    return {
        "price": price,
        "trailing_yield_pct": trailing_yield_pct,
        "dividend_history": divhist.to_dict() if divhist is not None else None,
        "fetched_time": time.time(),
    }

def get_ticker_info(ticker: str, cache: Dict[str, Dict], force_refresh: bool = False) -> Dict:
    rec = cache.get(ticker)
    if rec and not force_refresh:
        age = time.time() - rec.get("fetched_time", 0)
        if age < CACHE_TTL:
            return rec
    new = fetch_ticker_data(ticker)
    cache[ticker] = new
    return new


# ---------- Dividend inference ----------
def infer_dividend_type(divhist: Optional[Dict]) -> str:
    if divhist:
        return "qualified"
    return "ordinary"


# ---------- 2026 single-filer tax parameters ----------
STANDARD_DEDUCTION = 16100.0  # IRS Rev. Proc. 2025-32

ORDINARY_BRACKETS = [
    (12400.0, 0.10),
    (50400.0, 0.12),
    (105700.0, 0.22),
    (201775.0, 0.24),
    (256225.0, 0.32),
    (640600.0, 0.35),
    (None, 0.37),
]

CG_0pct_limit_single = 49450.0
CG_15pct_limit_single = 545500.0


# ---------- Tax computation helpers ----------
def ordinary_tax_on_amount(taxable_amount: float) -> float:
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
        if upper is None or taxable_amount <= (upper or taxable_amount):
            break
        lower = upper
    return tax


def capital_gains_tax_for_qualified(q_amount: float, ordinary_income_after_deduction: float) -> float:
    if q_amount <= 0:
        return 0.0
    tax = 0.0
    remaining = q_amount

    cap0_space = max(0.0, CG_0pct_limit_single - ordinary_income_after_deduction)
    take0 = min(remaining, cap0_space)
    remaining -= take0

    already = ordinary_income_after_deduction + take0
    cap15_space = max(0.0, CG_15pct_limit_single - already)
    take15 = min(remaining, cap15_space)
    tax += take15 * 0.15
    remaining -= take15

    if remaining > 0:
        tax += remaining * 0.20

    return tax


def taxable_social_security(ordinary_income: float, ss_annual: float) -> float:
    """Approximate taxable portion of Social Security for single filers."""
    if ss_annual <= 0:
        return 0.0
    provisional = ordinary_income + 0.5 * ss_annual
    if provisional <= 25000:
        taxable = 0.0
    elif provisional <= 34000:
        taxable = 0.5 * (provisional - 25000)
    else:
        taxable = 0.85 * ss_annual - max(0.0, 0.85 * (34000 - 25000))
        # Adjust for partial band; clamp to max 85%
        taxable = min(taxable, 0.85 * ss_annual)
    return taxable


# ---------- Portfolio parsing ----------
def parse_portfolio_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Portfolio JSON must be a list of holdings")
    return data


# ---------- Main estimator ----------
def estimate_with_lookup(rows: List[Dict], cache: Dict[str, Dict],
                         force_refresh: bool = False,
                         social_security_monthly: float = 0.0):
    annual_by_account = {"deferred": 0.0, "brokerage": 0.0, "roth": 0.0}
    brokerage_qualified = 0.0
    brokerage_ordinary = 0.0

    for r in rows:
        ticker = r["ticker"].strip().upper()
        acct = r["account_type"].strip().lower()
        shares = float(r["shares"])

        info = get_ticker_info(ticker, cache, force_refresh=force_refresh)
        price = info.get("price")
        if price is None:
            raise RuntimeError(f"Could not fetch price for {ticker}")

        yield_pct = info.get("trailing_yield_pct")
        if yield_pct is None:
            divhist = info.get("dividend_history")
            if divhist:
                total_div = sum(divhist.values())
                yield_pct = (total_div / price) * 100.0
            else:
                yield_pct = 0.0

        div_type = infer_dividend_type(info.get("dividend_history"))
        mv = shares * price
        annual_income = mv * (yield_pct / 100.0)
        print(f'Ticker: {ticker}, Income: {annual_income}')
        annual_by_account[acct] += annual_income

        if acct == "brokerage":
            if div_type == "qualified":
                brokerage_qualified += annual_income
            else:
                brokerage_ordinary += annual_income

    total_annual = sum(annual_by_account.values())
    total_monthly = total_annual / 12.0

    taxable_ordinary = annual_by_account["deferred"] + brokerage_ordinary

    ss_annual = social_security_monthly * 12.0
    ss_taxable = taxable_social_security(taxable_ordinary, ss_annual)
    taxable_ordinary += ss_taxable

    ordinary_after_deduction = max(0.0, taxable_ordinary - STANDARD_DEDUCTION)
    ordinary_tax = ordinary_tax_on_amount(ordinary_after_deduction)
    capg_tax = capital_gains_tax_for_qualified(brokerage_qualified, ordinary_after_deduction)
    federal_tax = ordinary_tax + capg_tax

    taxable_withdrawals_annual = (
        annual_by_account["deferred"] + brokerage_ordinary + brokerage_qualified + ss_taxable
    )
    effective_rate = (federal_tax / taxable_withdrawals_annual) if taxable_withdrawals_annual > 0 else 0.0

    return {
        "annual_by_account": annual_by_account,
        "brokerage_qualified": brokerage_qualified,
        "brokerage_ordinary": brokerage_ordinary,
        "social_security_monthly": social_security_monthly,
        "social_security_taxable_annual": ss_taxable,
        "total_annual_withdrawal": total_annual + ss_annual,
        "total_monthly_withdrawal": total_monthly + social_security_monthly,
        "estimated_federal_tax": federal_tax,
        "effective_tax_rate_on_taxable_withdrawals": effective_rate,
        "ordinary_after_deduction": ordinary_after_deduction,
        "ordinary_tax": ordinary_tax,
        "capital_gains_tax": capg_tax,
        "standard_deduction_used": min(STANDARD_DEDUCTION, taxable_ordinary),
    }


# ---------- CLI ----------
def print_report(res: Dict):
    print("\n=== Withdrawal & Federal Tax Estimate (2026, single filer) ===\n")
    print("Annual withdrawals by account:")
    for a, v in res["annual_by_account"].items():
        print(f"  {a:9s}: ${v:,.2f}")
    print(f"\nTotal annual withdrawal (incl. Social Security): ${res['total_annual_withdrawal']:,.2f}")
    print(f"Total monthly withdrawal: ${res['total_monthly_withdrawal']:,.2f}")
    print(f"\nTaxable Social Security (annual): ${res['social_security_taxable_annual']:,.2f}")
    print(f"\nOrdinary income after standard deduction: ${res['ordinary_after_deduction']:,.2f}")
    print(f"Estimated ordinary tax: ${res['ordinary_tax']:,.2f}")
    print(f"Estimated tax on qualified dividends: ${res['capital_gains_tax']:,.2f}")
    print(f"\nEstimated federal tax (annual): ${res['estimated_federal_tax']:,.2f}")
    print(f"Effective tax rate on taxable withdrawals: {res['effective_tax_rate_on_taxable_withdrawals']*100:.2f}%")
    print("\n(Notes: Roth withdrawals tax-free; SS taxed using provisional income rules.)\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Withdrawal estimator (JSON input, 2026 single filer).")
    parser.add_argument("jsonfile", help="JSON file of holdings")
    parser.add_argument("--refresh", action="store_true", help="Force refresh of ticker data")
    parser.add_argument("--social-security", type=float, default=0.0,
                        help="Monthly Social Security income (optional)")
    args = parser.parse_args()

    rows = parse_portfolio_json(args.jsonfile)
    cache = load_cache()
    res = estimate_with_lookup(rows, cache,
                               force_refresh=args.refresh,
                               social_security_monthly=args.social_security)
    save_cache(cache)
    print_report(res)


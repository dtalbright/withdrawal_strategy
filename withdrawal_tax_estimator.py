#!/usr/bin/env python3
"""
withdrawal_tax_estimator.py

Simple withdrawal + federal tax estimator for 2026 (single filers).

Usage:
    python3 withdrawal_tax_estimator.py portfolio.json [--refresh] [--social-security MONTHLY] [--capital_gains AMOUNT] [--breakdown]

Options:
    --refresh               Force refresh of ticker data (ignore cache)
    --social-security/-s    Monthly Social Security amount (default 0)
    --capital_gains/-c      Total long-term capital gains to include in tax calc (default 0)
    --breakdown             Show detailed bracket breakdown including Social Security tax allocation

Account types: "deferred", "brokerage", "roth"
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
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
    """
    Fetch price and dividend history using yfinance.
    For tickers ending in 'XX' (money market naming heuristic) we default price=1.0 and no dividends.
    """
    if ticker.endswith("XX"):
        price = 1.0
        divhist = None
    else:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        price = info.get("regularMarketPrice") or info.get("previousClose") or info.get("currentPrice") or None

        try:
            divhist = tk.dividends
        except Exception:
            divhist = None

    return {
        "price": price,
        "dividend_history": divhist.to_dict() if divhist is not None else None,
        "fetched_time": time.time(),
    }


def get_ticker_info(ticker: str, cache: Dict[str, Dict], force_refresh: bool = False) -> Dict:
    """
    Return ticker info using cache if available and fresh, otherwise fetch.
    """
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
    """
    Default: if historical dividends exist, treat distributions as 'qualified'.
    Otherwise treat as 'ordinary'.
    """
    if divhist:
        return "qualified"
    return "ordinary"


# ---------- 2026 single-filer tax parameters ----------
STANDARD_DEDUCTION = 16100.0  # single filer for 2026 (IRS Rev Proc 2025-32)

# Ordinary tax brackets for single filers (2026) as (upper_bound, marginal_rate)
ORDINARY_BRACKETS = [
    (12400.0, 0.10),
    (50400.0, 0.12),
    (105700.0, 0.22),
    (201775.0, 0.24),
    (256225.0, 0.32),
    (640600.0, 0.35),
    (None, 0.37),
]

# Updated 2026 capital gains/QDI thresholds for single filers (IRS Rev Proc 2025-32)
CG_0pct_limit_single = 49450.0
CG_15pct_limit_single = 545500.0


# ---------- Tax computation helpers ----------
def ordinary_tax_on_amount(taxable_amount: float) -> float:
    """
    Compute ordinary income tax using ORDINARY_BRACKETS.
    taxable_amount assumed >= 0.
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
    Compute tax on qualified dividends / LTCG given ordinary income after deduction.
    Fills 0% band first (capacity = CG_0pct_limit_single - ordinary_income_after_deduction),
    then 15% band, remainder at 20%.
    """
    if q_amount <= 0:
        return 0.0

    tax = 0.0
    remaining = q_amount

    # 0% band
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
    """
    Approximate taxable portion of Social Security for single filers.
    Formula: provisional = ordinary_income + 0.5 * SS
    - provisional <= 25k -> 0 taxable
    - 25k < provisional <= 34k -> up to 50% taxable on the excess over 25k
    - provisional > 34k -> up to 85% taxable (approx)
    """
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
                         additional_ltcg: float = 0.0,
                         social_security_monthly: float = 0.0) -> Dict:
    """
    Returns a dict with key values used in reporting. additional_ltcg is a flat LT capital gain amount
    that will be added into the brokerage capital gains bucket and into the brokerage withdrawals.
    """
    annual_by_account = {"deferred": 0.0, "brokerage": 0.0, "roth": 0.0}
    brokerage_qualified = 0.0
    brokerage_ordinary = 0.0
    brokerage_capital_gains = float(additional_ltcg or 0.0)

    for r in rows:
        ticker = r["ticker"].strip().upper()
        acct = r["account_type"].strip().lower()
        shares = float(r.get("shares", 0.0))

        info = get_ticker_info(ticker, cache, force_refresh=force_refresh)
        price = info.get("price")
        if price is None:
            raise RuntimeError(f"Could not fetch price for {ticker}")

        # --- Determine yield_pct (simplified logic) ---
        yield_override = r.get("override_yield_pct")
        price = info.get("price") or 1.0

        if yield_override is not None:
            # Explicit override for money market funds (or any manual entry)
            yield_pct = float(yield_override)
            print(f"[INFO] Using override yield for {ticker}: {yield_pct:.2f}%")

        else:
            # Always compute yield based on trailing 12-month actual dividends
            divhist = info.get("dividend_history")
            if not divhist:
                yield_pct = 0.0
                print(f"[INFO] No dividend history for {ticker}, yield set to 0")
            else:
                # Keep only the last 2 years of dividend data for cache compactness
                div_series = pd.Series(divhist)
                div_series.index = pd.to_datetime(div_series.index, utc=True, errors="coerce")
                div_series = div_series[~div_series.index.isna()]  # remove bad timestamps
                two_years_ago = pd.Timestamp.now(tz=div_series.index.tz) - pd.DateOffset(years=2)
                div_series = div_series[div_series.index > two_years_ago]

                # Update the cache with trimmed history
                cache[ticker]["dividend_history"] = div_series.to_dict()

                # Now compute trailing 12-month yield
                one_year_ago = pd.Timestamp.now(tz=div_series.index.tz) - pd.DateOffset(years=1)
                recent_divs = div_series[div_series.index > one_year_ago]

                if not recent_divs.empty and price > 0:
                    total_div = recent_divs.sum()
                    yield_pct = (total_div / price) * 100.0
                    print(f"[INFO] TTM dividend yield for {ticker}: {yield_pct:.2f}%")
                else:
                    yield_pct = 0.0
                    print(f"[INFO] No dividends in the last 12 months for {ticker}, yield set to 0")

        # --- Determine dividend tax type ---
        dividend_tax_type = r.get("dividend_tax_type")
        if yield_override is not None:
            # Money markets or manual yields → ordinary income
            dividend_tax_type = (dividend_tax_type or "ordinary").lower()
        else:
            # Default for any dividend-paying ETF/stock → qualified
            dividend_tax_type = (dividend_tax_type or "qualified").lower()

        mv = shares * price
        annual_income = mv * (yield_pct / 100.0)

        # --- Accumulate by account type ---
        if acct not in annual_by_account:
            raise ValueError(f"Unknown account type: {acct}")
        annual_by_account[acct] += annual_income

        if acct == "brokerage":
            if dividend_tax_type == "qualified":
                brokerage_qualified += annual_income
            elif dividend_tax_type == "capital_gains":
                brokerage_capital_gains += annual_income
            else:
                brokerage_ordinary += annual_income
        if acct == "deferred":
            dividend_tax_type = "ordinary"

        print(f"[CALC] {ticker}: MV=${mv:,.2f}, Income=${annual_income:,.2f}, Type={dividend_tax_type}")

    # --- Totals & ensure additional LTCG is reflected in brokerage withdrawals and totals ---
    # add additional_ltcg into brokerage withdrawals via brokerage_capital_gains earlier
    # total_annual should include all account withdrawals plus SS annual
    total_annual_without_ss = (annual_by_account["deferred"]
                               + annual_by_account["roth"]
                               + (annual_by_account["brokerage"] + brokerage_capital_gains))
    ss_annual = social_security_monthly * 12.0
    total_annual = total_annual_without_ss + ss_annual
    total_monthly = total_annual / 12.0

    taxable_ordinary = annual_by_account["deferred"] + brokerage_ordinary
    print(f"[CALC] Taxable ordinary: ${taxable_ordinary:,.2f}")
    print(f"[CALC] Brokerage qualifed: ${brokerage_qualified:,.2f}")
    print(f"[CALC] Brokerage cap gains: ${brokerage_capital_gains:,.2f}")
    print(f"[CALC] Annual social security: ${ss_annual:,.2f}")

    # Taxable portion of Social Security
    ss_taxable = taxable_social_security(taxable_ordinary, ss_annual)
    # include SS taxable in ordinary taxable amount for bracket calculation
    taxable_ordinary_with_ss = taxable_ordinary + ss_taxable

    # MAGI = ordinary income before SS taxation rules
    # = (taxable ordinary income + adjustments such as SS non-taxable portion)
    magi = taxable_ordinary + brokerage_qualified + brokerage_capital_gains

    # Ordinary income after deduction
    ordinary_after_deduction = max(0.0, taxable_ordinary_with_ss - STANDARD_DEDUCTION)
    ordinary_tax = ordinary_tax_on_amount(ordinary_after_deduction)

    # Capital gains tax calculation:
    # Qualified dividends first (fill 0%/15% bands), then capital gains stacking after qualified dividends.
    qualified_dividend_tax = capital_gains_tax_for_qualified(
        brokerage_qualified, ordinary_after_deduction
    )
    ordinary_plus_q = ordinary_after_deduction + brokerage_qualified
    capital_gains_tax_amount = capital_gains_tax_for_qualified(
        brokerage_capital_gains, ordinary_plus_q
    )
    capg_tax = qualified_dividend_tax + capital_gains_tax_amount

    federal_tax = ordinary_tax + capg_tax

    taxable_withdrawals_annual = (
        annual_by_account["deferred"]
        + brokerage_ordinary
        + brokerage_qualified
        + brokerage_capital_gains
        + ss_taxable
    )
    effective_rate = federal_tax / taxable_withdrawals_annual if taxable_withdrawals_annual > 0 else 0.0

    return {
        "annual_by_account": annual_by_account,
        "brokerage_qualified": brokerage_qualified,
        "brokerage_ordinary": brokerage_ordinary,
        "brokerage_capital_gains": brokerage_capital_gains,
        "social_security_monthly": social_security_monthly,
        "social_security_taxable_annual": ss_taxable,
        "magi": magi,
        "total_annual_withdrawal": total_annual,
        "total_monthly_withdrawal": total_monthly,
        "estimated_federal_tax": federal_tax,
        "effective_tax_rate_on_taxable_withdrawals": effective_rate,
        "ordinary_after_deduction": ordinary_after_deduction,
        "ordinary_tax": ordinary_tax,
        "capital_gains_tax": capg_tax,
        "standard_deduction_used": min(STANDARD_DEDUCTION, taxable_ordinary_with_ss),
    }


def print_tax_bracket_breakdown(res: Dict):
    """
    Print a detailed bracket breakdown:
     - ordinary income taxed bracket-by-bracket (uses ordinary_after_deduction)
     - qualified dividends + LTCG taxed across 0/15/20 bands
     - show taxable Social Security and estimate tax attributable to SS portion
    """
    ordinary_taxable = res["ordinary_after_deduction"]
    qualified_divs = res["brokerage_qualified"]
    capg = res["brokerage_capital_gains"]
    ss_taxable = res["social_security_taxable_annual"]
    ordinary_tax = res["ordinary_tax"]

    print("\n--- Ordinary Income Tax Breakdown ---")
    remaining = ordinary_taxable
    lower = 0.0
    computed_ord_tax = 0.0
    for upper, rate in ORDINARY_BRACKETS:
        if remaining <= 0:
            break
        if upper is None:
            taxable_here = remaining
        else:
            taxable_here = max(0.0, min(remaining, upper - lower))
        tax_slice = taxable_here * rate
        if taxable_here > 0:
            print(f"  ${lower:,.0f} – ${lower + taxable_here:,.0f} @ {rate*100:.0f}%: ${tax_slice:,.2f}")
        computed_ord_tax += tax_slice
        remaining -= taxable_here
        lower = upper if upper is not None else lower

    print(f"Total ordinary taxable (after deduction): ${ordinary_taxable:,.2f} → Ordinary tax: ${computed_ord_tax:,.2f}")

    # Qualified dividends + LTCG
    cg_total = qualified_divs + capg
    print("\n--- Qualified Dividends & Long-Term Capital Gains Breakdown ---")
    print(f"  Base taxable ordinary (for CG thresholds): ${ordinary_taxable:,.2f}")
    # space in 0% band
    space0 = max(0.0, CG_0pct_limit_single - ordinary_taxable)
    cg0 = min(cg_total, space0)
    remain = cg_total - cg0
    print(f"    {cg0:,.2f} taxed @ 0% → $0.00")
    tax_cg = 0.0
    if remain > 0:
        # 15% band capacity after ordinary+cg0
        already = ordinary_taxable + cg0
        cap15 = max(0.0, CG_15pct_limit_single - already)
        take15 = min(remain, cap15)
        tax_15 = take15 * 0.15
        print(f"    {take15:,.2f} taxed @ 15% → ${tax_15:,.2f}")
        tax_cg += tax_15
        remain -= take15
        if remain > 0:
            tax_20 = remain * 0.20
            print(f"    {remain:,.2f} taxed @ 20% → ${tax_20:,.2f}")
            tax_cg += tax_20

    print(f"Total qualified + LTCG: ${cg_total:,.2f} → CG tax: ${tax_cg:,.2f}")

    # Sanity print vs computed values
    print(f"\n(Computed ordinary tax: ${computed_ord_tax:,.2f}, script ordinary tax: ${ordinary_tax:,.2f})")
    print(f"(Computed CG tax: ${tax_cg:,.2f}, script CG tax: ${res['capital_gains_tax']:,.2f})")

    # Social Security tax allocation estimate
    print("\n--- Social Security Taxation ---")
    print(f"Taxable Social Security (annual): ${ss_taxable:,.2f}")
    denom = ordinary_taxable if ordinary_taxable > 0 else (ss_taxable if ss_taxable > 0 else 1.0)
    ss_share = min(1.0, ss_taxable / denom) if denom > 0 else 0.0
    ss_tax_est = ordinary_tax * ss_share
    print(f"Estimated portion of ordinary tax attributable to SS: ${ss_tax_est:,.2f} (approx)")

    print("\n--- Summary ---")
    print(f"Ordinary tax: ${ordinary_tax:,.2f}")
    print(f"Capital gains tax: ${res['capital_gains_tax']:,.2f}")
    print(f"Total federal tax: ${res['estimated_federal_tax']:,.2f}")

# ---------- CLI ----------
def print_report(res: Dict):
    print("\n=== Withdrawal & Federal Tax Estimate (2026, single filer) ===\n")
    print("Annual withdrawals by account:")
    print(f"  deferred : ${res['annual_by_account']['deferred']:,.2f}")
    print(f"  brokerage: ${res['annual_by_account']['brokerage'] + res['brokerage_capital_gains']:,.2f}")
    print(f"  roth     : ${res['annual_by_account']['roth']:,.2f}")
    print(f"\nTotal annual withdrawal (incl. Social Security): ${res['total_annual_withdrawal']:,.2f}")
    print(f"Total monthly withdrawal: ${res['total_monthly_withdrawal']:,.2f}")
    print(f"\nTaxable Social Security (annual): ${res['social_security_taxable_annual']:,.2f}")
    print(f"MAGI (Provisional Income calc basis): ${res['magi']:,.2f}")
    print(f"\nOrdinary income after standard deduction: ${res['ordinary_after_deduction']:,.2f}")
    print(f"Estimated ordinary tax: ${res['ordinary_tax']:,.2f}")
    print(f"Estimated capital gains tax: ${res['capital_gains_tax']:,.2f}")
    print(f"\nEstimated federal tax (annual): ${res['estimated_federal_tax']:,.2f}")
    print(f"Effective tax rate on taxable withdrawals: {res['effective_tax_rate_on_taxable_withdrawals']*100:.2f}%")
    print("\n(Notes: Roth withdrawals tax-free; SS taxed using provisional income rules.)\n")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Withdrawal estimator (JSON input, 2026 single filer).")
    parser.add_argument("jsonfile", help="JSON file of holdings")
    parser.add_argument("-r", "--refresh", action="store_true", help="Force refresh of ticker data")
    parser.add_argument("-s", "--social-security", type=float, default=0.0,
                        help="Monthly Social Security income (optional)")
    parser.add_argument("-c", "--capital_gains", type=float, default=0.0,
                        help="Total long-term capital gains to include in the tax calculation")
    parser.add_argument("-b", "--breakdown", action="store_true",
                        help="Show detailed tax bracket breakdown including Social Security allocation")
    args = parser.parse_args()

    rows = parse_portfolio_json(args.jsonfile)
    cache = load_cache()
    res = estimate_with_lookup(rows, cache,
                               force_refresh=args.refresh,
                               additional_ltcg=args.capital_gains,
                               social_security_monthly=args.social_security)
    save_cache(cache)
    print_report(res)

    if args.breakdown:
        print_tax_bracket_breakdown(res)

    print('\n')


#!/usr/bin/env python3

"""
portfolio_beta.py

Calculate weighted beta of a portfolio defined in a JSON file (flat list).

Usage:
    python3 portfolio_beta.py portfolio.json [--refresh]

Positional arguments:
    portfolio.json   Portfolio JSON file (flat list of holdings)

Options:
    --refresh        Refresh price & beta data via Yahoo Finance

Notes:
- JSON entries must include: ticker, account_type, shares
- Values are computed as shares × price
- Missing beta defaults to 1.0 and is flagged
- Beta by account type and overall portfolio is reported
"""

import argparse
import json
import os
from typing import Dict, List
import yfinance as yf

CACHE_FILE = "beta_cache.json"


def load_portfolio(json_path: str) -> List[Dict]:
    with open(json_path, "r") as f:
        return json.load(f)


def load_cache() -> Dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache: Dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_ticker_info(ticker: str) -> Dict:
    """Fetch current price and beta from Yahoo Finance."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        price = info.get("regularMarketPrice") or info.get("previousClose") or 1.0
        beta = info.get("beta3Year")
        if beta is None:
            print(f"Warning: No beta found for {ticker}, using 1.0")
            beta = 1.0
        return {"price": float(price), "beta": float(beta)}
    except Exception:
        print(f"Error fetching info for {ticker}, using price=1.0, beta=1.0")
        return {"price": 1.0, "beta": 1.0}


def compute_weighted_beta(holdings: List[Dict], refresh: bool):
    cache = load_cache()
    total_value = 0.0
    per_type_totals = {"brokerage": 0.0, "deferred": 0.0, "roth": 0.0}
    per_type_beta_sum = {"brokerage": 0.0, "deferred": 0.0, "roth": 0.0}

    for h in holdings:
        ticker = h["ticker"].upper()
        acct_type = h["account_type"].lower()
        shares = float(h["shares"])

        if refresh or ticker not in cache:
            cache[ticker] = fetch_ticker_info(ticker)

        price = cache[ticker]["price"]
        beta = cache[ticker]["beta"]

        value = shares * price
        total_value += value

        if acct_type in per_type_totals:
            per_type_totals[acct_type] += value
            per_type_beta_sum[acct_type] += value * beta

        print(f"{ticker}: Shares={shares}, Price={price:.2f}, Value={value:.2f}, Beta={beta:.2f}")

    save_cache(cache)

    overall_beta = sum(per_type_beta_sum.values()) / total_value if total_value > 0 else 0.0

    beta_by_account = {}
    for acct in per_type_totals:
        if per_type_totals[acct] > 0:
            beta_by_account[acct] = per_type_beta_sum[acct] / per_type_totals[acct]
        else:
            beta_by_account[acct] = None  # no assets in this account type

    return overall_beta, beta_by_account


def main():
    parser = argparse.ArgumentParser(description="Calculate weighted portfolio beta")
    parser.add_argument("portfolio_file", help="Portfolio JSON file")
    parser.add_argument("--refresh", action="store_true", help="Refresh beta & price data via Yahoo Finance")
    args = parser.parse_args()

    if not os.path.exists(args.portfolio_file):
        raise FileNotFoundError(f"Portfolio file not found: {args.portfolio_file}")

    holdings = load_portfolio(args.portfolio_file)
    if not holdings:
        print("Portfolio is empty!")
        return

    overall_beta, beta_by_account = compute_weighted_beta(holdings, args.refresh)

    print("\n================ Portfolio Beta =================")
    print(f"Overall Portfolio Beta: {overall_beta:.3f}")
    print("Beta by Account Type:")
    for acct, beta in beta_by_account.items():
        if beta is not None:
            print(f"  {acct}: {beta:.3f}")
        else:
            print(f"  {acct}: No holdings")
    print("===============================================")


if __name__ == "__main__":
    main()


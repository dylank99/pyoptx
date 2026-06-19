#!/usr/bin/env python3
"""Implied-volatility smile / surface from a real option chain.

Pulls an option chain with yfinance, computes the Black-Scholes implied
volatility for each quote using ``pyoptx.implied_volatility``, and plots the
volatility smile (per maturity) and a 3-D volatility surface.

If yfinance or the network is unavailable, pass ``--synthetic`` to generate a
realistic smile/surface from a parametric vol model so the plotting path can be
exercised offline.

Usage:
    python scripts/calibration_demo.py --ticker AAPL
    python scripts/calibration_demo.py --synthetic        # offline demo
"""

import argparse
import datetime as dt
import math

import numpy as np

import pyoptx


# --------------------------------------------------------------------------- #
# Data acquisition
# --------------------------------------------------------------------------- #
def fetch_chain(ticker: str, max_expiries: int, r: float):
    """Return (spot, [(T_years, strikes[], mid_prices[], opt_type)], r) via yfinance."""
    import yfinance as yf  # imported lazily so --synthetic needs no network

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    today = dt.date.today()

    rows = []
    for expiry in tk.options[:max_expiries]:
        exp_date = dt.date.fromisoformat(expiry)
        t_years = max((exp_date - today).days, 1) / 365.0
        chain = tk.option_chain(expiry)
        calls = chain.calls
        # Keep liquid, near-the-money quotes with a real bid/ask.
        calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
        mid = ((calls["bid"] + calls["ask"]) / 2.0).to_numpy()
        strikes = calls["strike"].to_numpy()
        moneyness = strikes / spot
        mask = (moneyness > 0.85) & (moneyness < 1.15)
        if mask.sum() >= 3:
            rows.append((t_years, strikes[mask], mid[mask], "call"))
    return spot, rows, r


def synthetic_chain(r: float):
    """A parametric smile (vol rises away from ATM) for offline demos."""
    spot = 100.0
    rows = []
    for t_years in (0.08, 0.25, 0.5, 1.0):
        strikes = np.linspace(80.0, 120.0, 17)
        prices = []
        for k in strikes:
            m = math.log(k / spot)
            # Smile: base vol + curvature in log-moneyness + mild term effect.
            sigma = 0.18 + 0.6 * m * m + 0.02 * math.sqrt(t_years)
            prices.append(pyoptx.bs_price(spot, float(k), t_years, r, sigma, "call"))
        rows.append((t_years, strikes, np.array(prices), "call"))
    return spot, rows, r


# --------------------------------------------------------------------------- #
# IV computation + plotting
# --------------------------------------------------------------------------- #
def compute_ivs(spot, rows, r):
    """Return list of (T, strikes[], ivs[]) dropping points that fail to solve."""
    out = []
    for t_years, strikes, prices, opt in rows:
        ks, ivs = [], []
        for k, p in zip(strikes, prices):
            try:
                iv = pyoptx.implied_volatility(float(p), spot, float(k), t_years, r, opt)
            except (ValueError, RuntimeError):
                continue  # outside no-arbitrage bounds or non-convergent
            if 1e-3 < iv < 5.0:
                ks.append(float(k))
                ivs.append(iv)
        if ks:
            out.append((t_years, np.array(ks), np.array(ivs)))
    return out


def plot(spot, surface, ticker, outfile):
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    fig = plt.figure(figsize=(13, 5.5))

    # (1) Smile per maturity
    ax1 = fig.add_subplot(1, 2, 1)
    for t_years, ks, ivs in surface:
        ax1.plot(ks / spot, ivs * 100, marker="o", ms=3, label=f"T={t_years*365:.0f}d")
    ax1.axvline(1.0, color="grey", ls="--", lw=0.8)
    ax1.set_xlabel("moneyness  K / S")
    ax1.set_ylabel("implied vol (%)")
    ax1.set_title(f"{ticker} volatility smile")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # (2) Surface
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for t_years, ks, ivs in surface:
        ax2.plot(ks / spot, np.full_like(ks, t_years), ivs * 100, marker="o", ms=2)
    ax2.set_xlabel("K / S")
    ax2.set_ylabel("T (years)")
    ax2.set_zlabel("IV (%)")
    ax2.set_title(f"{ticker} volatility surface")

    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    print(f"saved plot -> {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--rate", type=float, default=0.04, help="risk-free rate")
    ap.add_argument("--max-expiries", type=int, default=6)
    ap.add_argument("--synthetic", action="store_true", help="offline parametric data")
    ap.add_argument("--outfile", default="vol_surface.png")
    args = ap.parse_args()

    if args.synthetic:
        ticker = "SYNTHETIC"
        spot, rows, r = synthetic_chain(args.rate)
    else:
        ticker = args.ticker
        try:
            spot, rows, r = fetch_chain(args.ticker, args.max_expiries, args.rate)
        except Exception as exc:  # network/yfinance issues -> fall back gracefully
            print(f"yfinance fetch failed ({exc!r}); falling back to --synthetic data")
            ticker, (spot, rows, r) = "SYNTHETIC", (synthetic_chain(args.rate))

    print(f"{ticker}: spot={spot:.2f}, {len(rows)} maturities")
    surface = compute_ivs(spot, rows, r)
    total = sum(len(ks) for _, ks, _ in surface)
    print(f"computed {total} implied vols across {len(surface)} maturities")
    plot(spot, surface, ticker, args.outfile)


if __name__ == "__main__":
    main()

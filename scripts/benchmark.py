#!/usr/bin/env python3
"""Benchmark the Rust core against an equivalent pure-Python implementation.

Compares Black-Scholes pricing (throughput over many scalar calls) and Monte
Carlo pricing (single large run), and prints the speedup. The pure-Python code
below is a faithful, dependency-free port of the same formulas used in the Rust
core, so the comparison is apples-to-apples.

Usage:
    python scripts/benchmark.py [--bs-iters N] [--mc-paths N]
"""

import argparse
import math
import random
import time

import pyoptx

SQRT_2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Pure-Python reference implementations
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    # math.erf is C-backed but called per-evaluation; this mirrors Phi(x).
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def py_bs_price(s, k, t, r, sigma, opt="call", q=0.0):
    if t == 0.0:
        intrinsic = max(s - k, 0.0) if opt == "call" else max(k - s, 0.0)
        return intrinsic
    vol_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    dq, dr = math.exp(-q * t), math.exp(-r * t)
    if opt == "call":
        return s * dq * _norm_cdf(d1) - k * dr * _norm_cdf(d2)
    return k * dr * _norm_cdf(-d2) - s * dq * _norm_cdf(-d1)


def py_mc_price(s, k, t, r, sigma, opt="call", n_paths=100_000, seed=0, q=0.0):
    rng = random.Random(seed)
    disc = math.exp(-r * t)
    drift = (r - q - 0.5 * sigma * sigma) * t
    vol = sigma * math.sqrt(t)
    total = 0.0
    total_sq = 0.0
    for _ in range(n_paths):
        z = rng.gauss(0.0, 1.0)
        st = s * math.exp(drift + vol * z)
        payoff = max(st - k, 0.0) if opt == "call" else max(k - st, 0.0)
        y = disc * payoff
        total += y
        total_sq += y * y
    mean = total / n_paths
    var = (total_sq - n_paths * mean * mean) / (n_paths - 1)
    return mean, math.sqrt(var / n_paths)


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #
def time_it(fn, *args, repeat=1, **kwargs):
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best, out


def bench_black_scholes(iters):
    # A spread of parameters so the branch predictor / cache see realistic work.
    params = [(100.0, 90.0 + (i % 40), 1.0, 0.05, 0.2) for i in range(1000)]

    def run_rust():
        acc = 0.0
        for _ in range(iters // 1000):
            for s, k, t, r, sig in params:
                acc += pyoptx.bs_price(s, k, t, r, sig, "call")
        return acc

    def run_py():
        acc = 0.0
        for _ in range(iters // 1000):
            for s, k, t, r, sig in params:
                acc += py_bs_price(s, k, t, r, sig, "call")
        return acc

    rust_t, _ = time_it(run_rust, repeat=3)
    py_t, _ = time_it(run_py, repeat=3)
    return rust_t, py_t


def bench_monte_carlo(paths):
    rust_t, rust_out = time_it(
        pyoptx.mc_price, 100, 100, 1.0, 0.05, 0.2, "call",
        n_paths=paths, seed=1, antithetic=False, control_variate=False,
    )
    py_t, py_out = time_it(py_mc_price, 100, 100, 1.0, 0.05, 0.2, "call", n_paths=paths, seed=1)
    return rust_t, py_t, rust_out[0], py_out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs-iters", type=int, default=1_000_000)
    ap.add_argument("--mc-paths", type=int, default=500_000)
    args = ap.parse_args()

    print(f"pyoptx {pyoptx.__version__} — Rust core vs pure Python\n")

    bs_rust, bs_py = bench_black_scholes(args.bs_iters)
    print("Black-Scholes pricing")
    print(f"  iterations     : {args.bs_iters:,}")
    print(f"  rust core      : {bs_rust*1e3:8.2f} ms")
    print(f"  pure python    : {bs_py*1e3:8.2f} ms")
    print(f"  speedup        : {bs_py / bs_rust:8.1f}x\n")

    mc_rust, mc_py, rust_price, py_price = bench_monte_carlo(args.mc_paths)
    print("Monte Carlo pricing")
    print(f"  paths          : {args.mc_paths:,}")
    print(f"  rust core      : {mc_rust*1e3:8.2f} ms   (price {rust_price:.4f})")
    print(f"  pure python    : {mc_py*1e3:8.2f} ms   (price {py_price:.4f})")
    print(f"  speedup        : {mc_py / mc_rust:8.1f}x")


if __name__ == "__main__":
    main()

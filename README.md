# pyoptx

**Fast options pricing and risk library: Black-Scholes, binomial trees, and Monte Carlo with full Greeks, built on a Rust core with Python bindings.**

[![CI](https://github.com/dylank99/pyoptx/actions/workflows/ci.yml/badge.svg)](https://github.com/dylank99/pyoptx/actions/workflows/ci.yml)

`pyoptx` pairs a small, well-tested Rust numerical core (`pyoptx_core`) with an ergonomic, typed Python API. The heavy math runs in compiled Rust via [PyO3](https://pyo3.rs); Python gets a clean surface with `.pyi` type stubs.

---

## Features

- **Black-Scholes-Merton** European call/put pricing with continuous dividend yield, and the full set of first-order Greeks — delta, gamma, vega, theta, rho.
- **Binomial trees** — Cox-Ross-Rubinstein for European *and* American options with a configurable number of steps; provably converges to Black-Scholes for European options.
- **Monte Carlo** — GBM simulation with **antithetic variates** and a **control variate**, a reported **standard error**, and a **seedable RNG** for reproducibility.
- **Implied volatility** — robust Newton-Raphson with an automatic bisection fallback; non-convergence and arbitrage violations are reported, not swallowed.
- **Numerical rigor** — `f64` throughout, a double-precision normal CDF, sensible handling of edge cases (`T=0`, `σ=0`, deep ITM/OTM), and no panics in library paths: errors surface as proper Python exceptions.
- **Typed API** — `pyoptx/_core.pyi` stubs and a `py.typed` marker.

## Architecture

```
pyoptx/                  Python package (clean, typed re-exports)
├── __init__.py          public API
├── _core.pyi            type stubs for the compiled module
└── py.typed
src/                     Rust core crate `pyoptx_core`
├── lib.rs               module wiring + PyO3 bindings (exposed as pyoptx._core)
├── distributions.rs     normal PDF / CDF (West 2009 high-accuracy CDF)
├── black_scholes.rs     pricing + Greeks
├── binomial.rs          CRR European / American
├── monte_carlo.rs       GBM MC + variance reduction
├── implied_vol.rs       Newton + bisection solver
└── error.rs             error type -> Python exceptions
```

The Rust math modules are **pure** (no PyO3 dependency); the PyO3 glue is compiled only behind the `extension-module` feature. This means `cargo test` runs the numerical unit tests with zero Python linkage, while `maturin` builds the extension with the bindings enabled.

## Installation

### From source (development)

Requires a [Rust toolchain](https://rustup.rs/) and Python ≥ 3.8.

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release          # builds the Rust extension and installs pyoptx editable
```

Or a plain editable install (maturin is the build backend):

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[test]"           # pytest
pip install -e ".[demo]"           # yfinance, numpy, matplotlib, pandas (calibration demo)
```

## Quick start

```python
import pyoptx

# --- Black-Scholes price + Greeks ---
price = pyoptx.bs_price(s=100, k=100, t=1.0, r=0.05, sigma=0.20, option_type="call")
# 10.4506

g = pyoptx.bs_greeks(100, 100, 1.0, 0.05, 0.20, "call")
# {'price': 10.45, 'delta': 0.6368, 'gamma': 0.0188, 'vega': 37.52, 'theta': -6.41, 'rho': 53.23}

# --- Binomial: European converges to BS; American captures early exercise ---
pyoptx.binomial_price(100, 100, 1.0, 0.05, 0.20, "call", "european", steps=1000)   # ~10.4506
pyoptx.binomial_price(100, 110, 1.0, 0.08, 0.30, "put",  "american", steps=1000)   # > European put

# --- Monte Carlo with standard error (reproducible via seed) ---
mc_price, std_err = pyoptx.mc_price(100, 100, 1.0, 0.05, 0.20, "call",
                                    n_paths=200_000, seed=42)
print(f"{mc_price:.4f} ± {std_err:.4f}")

# --- Implied volatility ---
iv = pyoptx.implied_volatility(market_price=10.4506, s=100, k=100, t=1.0, r=0.05,
                               option_type="call")          # ~0.2000
```

All functions take `q` (continuous dividend yield, default `0.0`) as a trailing argument. `option_type` is `"call"` or `"put"`.

## The math

**Notation:** spot `S`, strike `K`, time to expiry `T` (years), risk-free rate `r`, volatility `σ`, dividend yield `q`. `Φ` and `φ` are the standard normal CDF and PDF.

### Black-Scholes-Merton

$$d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

$$C = S e^{-qT}\Phi(d_1) - K e^{-rT}\Phi(d_2), \qquad P = K e^{-rT}\Phi(-d_2) - S e^{-qT}\Phi(-d_1)$$

First-order Greeks (vega per `1.0` of vol, theta per year, rho per `1.0` of rate):

| Greek | Call | Put |
|-------|------|-----|
| Delta | $e^{-qT}\Phi(d_1)$ | $-e^{-qT}\Phi(-d_1)$ |
| Gamma | $\dfrac{e^{-qT}\varphi(d_1)}{S\sigma\sqrt{T}}$ | same |
| Vega  | $S e^{-qT}\varphi(d_1)\sqrt{T}$ | same |
| Theta | $-\dfrac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}\Phi(d_2) + qSe^{-qT}\Phi(d_1)$ | $-\dfrac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt{T}} + rKe^{-rT}\Phi(-d_2) - qSe^{-qT}\Phi(-d_1)$ |
| Rho   | $KTe^{-rT}\Phi(d_2)$ | $-KTe^{-rT}\Phi(-d_2)$ |

The normal CDF uses the Hart algorithm as presented by Graeme West (2009), accurate to roughly machine precision — important for stable Greeks and implied-vol round-trips.

### Binomial tree (Cox-Ross-Rubinstein)

With `n` steps of size `Δt = T/n`:

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = \tfrac{1}{u}, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

Terminal payoffs are discounted back through the tree at $e^{-r\Delta t}$ per step. For **American** options each node is maxed against immediate exercise (intrinsic value). As `n → ∞` the European price converges to Black-Scholes.

### Monte Carlo (GBM)

Under the risk-neutral measure the terminal spot is

$$S_T = S\,\exp\!\Big[\big(r - q - \tfrac{1}{2}\sigma^2\big)T + \sigma\sqrt{T}\,Z\Big], \quad Z \sim \mathcal{N}(0,1)$$

and the price is $e^{-rT}\,\mathbb{E}[\text{payoff}(S_T)]$, estimated by sampling. Two variance-reduction techniques are available:

- **Antithetic variates** — each draw `Z` is paired with `−Z`, cancelling much of the sampling noise.
- **Control variate** — the discounted terminal spot $e^{-rT}S_T$ has the analytically known mean $S e^{-qT}$ (from the GBM/Black-Scholes model). Regressing the payoff on this control with the optimal coefficient $c^\* = \mathrm{Cov}(Y,X)/\mathrm{Var}(X)$ removes the variance it explains.

The estimator's **standard error** is always returned so you can judge the Monte Carlo noise.

### Implied volatility

Price is strictly increasing in `σ`, so the inverse is well defined between the no-arbitrage bounds. We seed Newton-Raphson with the Brenner-Subrahmanyam approximation $\sigma_0 \approx \sqrt{2\pi/T}\cdot \text{price}/S$ and fall back to **bisection** whenever vega is too small, a step leaves the bracket, or Newton fails to converge. Prices outside the no-arbitrage bounds raise `ValueError`; genuine non-convergence raises `RuntimeError`.

## Benchmarks

Rust core vs. an equivalent dependency-free pure-Python implementation of the same formulas (`scripts/benchmark.py`, single core):

| Workload | Pure Python | Rust core | Speedup |
|----------|------------:|----------:|--------:|
| Black-Scholes — 1,000,000 scalar prices | 571 ms | 161 ms | **3.6×** |
| Monte Carlo — 500,000 paths (1 option)  | 391 ms | 11 ms  | **35.5×** |

Scalar Black-Scholes is dominated by per-call FFI marshalling (and Python's BS uses C-backed `math.erf`), so the win is modest. The advantage compounds in tight numerical loops like Monte Carlo, where the entire path simulation stays in Rust. Reproduce with:

```bash
python scripts/benchmark.py
```

*(Numbers above were measured on the development machine; absolute timings vary by hardware.)*

## Calibration demo: the volatility smile

`scripts/calibration_demo.py` pulls a real option chain with `yfinance`, computes implied vols across strikes and maturities with `pyoptx.implied_volatility`, and plots the smile and a 3-D vol surface.

```bash
pip install -e ".[demo]"
python scripts/calibration_demo.py --ticker AAPL        # live data
python scripts/calibration_demo.py --synthetic          # offline parametric demo
```

The `--synthetic` mode needs no network and generates a realistic smile, so the IV-solver and plotting paths can be exercised anywhere.

## Testing

```bash
cargo test                          # Rust unit tests (pure-math core)
cargo clippy --all-targets -- -D warnings
maturin develop --release           # build the extension
pytest -q                           # Python tests (Greeks vs FD, IV round-trips, API)
```

Rust tests cover textbook prices, put-call parity, binomial→Black-Scholes convergence, and Monte Carlo within a few standard errors of the analytical price. Python tests check analytical Greeks against finite differences, implied-vol round-trips, and the public API surface.

## License

MIT — see [LICENSE](LICENSE).

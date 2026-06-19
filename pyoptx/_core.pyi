"""Type stubs for the compiled Rust extension ``pyoptx._core``.

These signatures mirror the ``#[pyfunction]`` definitions in ``src/lib.rs``.
All numeric arguments are ``float`` unless noted. ``option_type`` is
``"call"`` or ``"put"``; ``exercise`` is ``"european"`` or ``"american"``.
"""

from typing import Dict, Literal, Tuple

OptionType = Literal["call", "put", "c", "p"]
Exercise = Literal["european", "american", "euro", "amer"]

def bs_price(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    option_type: OptionType = ...,
    q: float = ...,
) -> float:
    """Black-Scholes-Merton price of a European option."""

def bs_delta(
    s: float, k: float, t: float, r: float, sigma: float,
    option_type: OptionType = ..., q: float = ...,
) -> float:
    """Delta — ∂price/∂S."""

def bs_gamma(
    s: float, k: float, t: float, r: float, sigma: float,
    option_type: OptionType = ..., q: float = ...,
) -> float:
    """Gamma — ∂²price/∂S² (identical for calls and puts)."""

def bs_vega(
    s: float, k: float, t: float, r: float, sigma: float,
    option_type: OptionType = ..., q: float = ...,
) -> float:
    """Vega — ∂price/∂σ, per 1.0 of volatility."""

def bs_theta(
    s: float, k: float, t: float, r: float, sigma: float,
    option_type: OptionType = ..., q: float = ...,
) -> float:
    """Theta — ∂price/∂t, per year."""

def bs_rho(
    s: float, k: float, t: float, r: float, sigma: float,
    option_type: OptionType = ..., q: float = ...,
) -> float:
    """Rho — ∂price/∂r, per 1.0 of the rate."""

def bs_greeks(
    s: float, k: float, t: float, r: float, sigma: float,
    option_type: OptionType = ..., q: float = ...,
) -> Dict[str, float]:
    """Price plus all five Greeks: keys ``price, delta, gamma, vega, theta, rho``."""

def binomial_price(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    option_type: OptionType = ...,
    exercise: Exercise = ...,
    steps: int = ...,
    q: float = ...,
) -> float:
    """Cox-Ross-Rubinstein binomial price (European or American)."""

def mc_price(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    option_type: OptionType = ...,
    n_paths: int = ...,
    seed: int = ...,
    q: float = ...,
    antithetic: bool = ...,
    control_variate: bool = ...,
) -> Tuple[float, float]:
    """Monte Carlo price of a European option. Returns ``(price, std_error)``."""

def implied_volatility(
    market_price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    option_type: OptionType = ...,
    q: float = ...,
) -> float:
    """Black-Scholes implied volatility from a market price (Newton + bisection)."""

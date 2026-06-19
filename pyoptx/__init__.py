"""pyoptx — fast options pricing and risk.

Black-Scholes, binomial trees, and Monte Carlo with full Greeks and implied
volatility, implemented in a Rust core (``pyoptx._core``) and re-exported here
with a clean, typed API.

Quick start
-----------
>>> import pyoptx
>>> pyoptx.bs_price(100, 100, 1.0, 0.05, 0.2, "call")
10.450583572185565
>>> g = pyoptx.bs_greeks(100, 100, 1.0, 0.05, 0.2, "call")
>>> round(g["delta"], 4)
0.6368
>>> price, std_err = pyoptx.mc_price(100, 100, 1.0, 0.05, 0.2, "call", seed=1)
>>> iv = pyoptx.implied_volatility(10.45, 100, 100, 1.0, 0.05, "call")

Conventions
-----------
* ``s`` spot, ``k`` strike, ``t`` time to expiry in years, ``r`` risk-free rate,
  ``sigma`` volatility, ``q`` continuous dividend yield (default 0).
* ``option_type`` is ``"call"`` or ``"put"``.
* Greeks: vega is per 1.0 of vol (per 100 vol points), theta is per year, rho is
  per 1.0 of the rate.
"""

from ._core import (
    bs_price,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
    bs_greeks,
    binomial_price,
    mc_price,
    implied_volatility,
)

__version__ = "0.1.0"

__all__ = [
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_rho",
    "bs_greeks",
    "binomial_price",
    "mc_price",
    "implied_volatility",
    "__version__",
]

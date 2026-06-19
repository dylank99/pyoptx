"""Python test suite for pyoptx.

Covers:
  * analytical Greeks vs central finite-difference approximations,
  * implied-vol round-trips (price -> IV -> price),
  * a basic API / error-handling smoke test.

Run with:  pytest -q
"""

import math

import pytest

import pyoptx

# A representative, well-conditioned parameter set.
S, K, T, R, SIGMA, Q = 100.0, 105.0, 0.75, 0.03, 0.22, 0.01


def price(s=S, k=K, t=T, r=R, sigma=SIGMA, opt="call", q=Q):
    return pyoptx.bs_price(s, k, t, r, sigma, opt, q)


# --------------------------------------------------------------------------- #
# Analytical Greeks vs finite differences
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("opt", ["call", "put"])
def test_delta_matches_finite_difference(opt):
    h = 1e-4
    fd = (price(s=S + h, opt=opt) - price(s=S - h, opt=opt)) / (2 * h)
    assert pyoptx.bs_delta(S, K, T, R, SIGMA, opt, Q) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_gamma_matches_second_difference(opt):
    h = 1e-2
    fd = (price(s=S + h, opt=opt) - 2 * price(opt=opt) + price(s=S - h, opt=opt)) / (h * h)
    assert pyoptx.bs_gamma(S, K, T, R, SIGMA, opt, Q) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_vega_matches_finite_difference(opt):
    h = 1e-4
    fd = (price(sigma=SIGMA + h, opt=opt) - price(sigma=SIGMA - h, opt=opt)) / (2 * h)
    assert pyoptx.bs_vega(S, K, T, R, SIGMA, opt, Q) == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_rho_matches_finite_difference(opt):
    h = 1e-5
    fd = (price(r=R + h, opt=opt) - price(r=R - h, opt=opt)) / (2 * h)
    assert pyoptx.bs_rho(S, K, T, R, SIGMA, opt, Q) == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_theta_matches_finite_difference(opt):
    # theta = -d(price)/dt
    h = 1e-5
    fd = -(price(t=T + h, opt=opt) - price(t=T - h, opt=opt)) / (2 * h)
    assert pyoptx.bs_theta(S, K, T, R, SIGMA, opt, Q) == pytest.approx(fd, abs=1e-3)


def test_bs_greeks_bundle_matches_individual():
    g = pyoptx.bs_greeks(S, K, T, R, SIGMA, "call", Q)
    assert g["price"] == pytest.approx(pyoptx.bs_price(S, K, T, R, SIGMA, "call", Q))
    assert g["delta"] == pytest.approx(pyoptx.bs_delta(S, K, T, R, SIGMA, "call", Q))
    assert g["gamma"] == pytest.approx(pyoptx.bs_gamma(S, K, T, R, SIGMA, "call", Q))
    assert g["vega"] == pytest.approx(pyoptx.bs_vega(S, K, T, R, SIGMA, "call", Q))
    assert g["theta"] == pytest.approx(pyoptx.bs_theta(S, K, T, R, SIGMA, "call", Q))
    assert g["rho"] == pytest.approx(pyoptx.bs_rho(S, K, T, R, SIGMA, "call", Q))


# --------------------------------------------------------------------------- #
# Put-call parity
# --------------------------------------------------------------------------- #
def test_put_call_parity():
    c = pyoptx.bs_price(S, K, T, R, SIGMA, "call", Q)
    p = pyoptx.bs_price(S, K, T, R, SIGMA, "put", Q)
    lhs = c - p
    rhs = S * math.exp(-Q * T) - K * math.exp(-R * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


# --------------------------------------------------------------------------- #
# Implied-vol round trips
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("k", [80.0, 95.0, 100.0, 110.0, 125.0])
@pytest.mark.parametrize("sigma", [0.12, 0.25, 0.45, 0.8])
def test_implied_vol_round_trip(opt, k, sigma):
    p = pyoptx.bs_price(S, k, T, R, sigma, opt, Q)
    if p < 1e-6:
        pytest.skip("price negligible; IV ill-conditioned")
    iv = pyoptx.implied_volatility(p, S, k, T, R, opt, Q)
    assert iv == pytest.approx(sigma, abs=1e-4)
    # And the recovered vol reprices to the original price.
    assert pyoptx.bs_price(S, k, T, R, iv, opt, Q) == pytest.approx(p, abs=1e-8)


def test_implied_vol_rejects_arbitrage_violation():
    # A call worth more than spot is impossible.
    with pytest.raises(ValueError):
        pyoptx.implied_volatility(200.0, 100.0, 100.0, 1.0, 0.05, "call")


# --------------------------------------------------------------------------- #
# Cross-model consistency
# --------------------------------------------------------------------------- #
def test_binomial_converges_to_black_scholes():
    bs = pyoptx.bs_price(S, K, T, R, SIGMA, "call", Q)
    tree = pyoptx.binomial_price(S, K, T, R, SIGMA, "call", "european", 3000, Q)
    assert tree == pytest.approx(bs, abs=2e-3)


def test_american_put_at_least_european():
    a = pyoptx.binomial_price(100, 110, 1.0, 0.08, 0.3, "put", "american", 800)
    e = pyoptx.binomial_price(100, 110, 1.0, 0.08, 0.3, "put", "european", 800)
    assert a >= e


def test_monte_carlo_within_standard_errors():
    bs = pyoptx.bs_price(S, K, T, R, SIGMA, "call", Q)
    price_mc, se = pyoptx.mc_price(S, K, T, R, SIGMA, "call", n_paths=200_000, seed=7, q=Q)
    assert se > 0.0
    assert abs(price_mc - bs) < 4.0 * se + 1e-9


def test_monte_carlo_reproducible():
    a = pyoptx.mc_price(S, K, T, R, SIGMA, "call", n_paths=20_000, seed=123, q=Q)
    b = pyoptx.mc_price(S, K, T, R, SIGMA, "call", n_paths=20_000, seed=123, q=Q)
    assert a == b


# --------------------------------------------------------------------------- #
# API / edge-case smoke tests
# --------------------------------------------------------------------------- #
def test_public_api_surface():
    for name in [
        "bs_price", "bs_delta", "bs_gamma", "bs_vega", "bs_theta", "bs_rho",
        "bs_greeks", "binomial_price", "mc_price", "implied_volatility",
    ]:
        assert hasattr(pyoptx, name), f"missing {name}"
    assert isinstance(pyoptx.__version__, str)


def test_intrinsic_at_expiry():
    assert pyoptx.bs_price(110, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(10.0)
    assert pyoptx.bs_price(90, 100, 0.0, 0.05, 0.2, "put") == pytest.approx(10.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        pyoptx.bs_price(-1, 100, 1, 0.05, 0.2, "call")
    with pytest.raises(ValueError):
        pyoptx.bs_price(100, 100, 1, 0.05, 0.2, "banana")
    with pytest.raises(ValueError):
        pyoptx.binomial_price(100, 100, 1, 0.05, 0.2, "call", "european", 0)

//! Implied volatility: recover `σ` from an observed option price.
//!
//! The price is strictly increasing in volatility, so the inverse is well
//! defined between the no-arbitrage bounds. We use Newton-Raphson (fast,
//! quadratic convergence) seeded with the Brenner-Subrahmanyam approximation,
//! and fall back to bisection whenever Newton misbehaves — small/zero vega,
//! a step outside the bracket, or simply failing to converge. Non-convergence
//! is reported as an error rather than returning a bogus number.

use crate::black_scholes::{self, OptionType};
use crate::error::{PyoptxError, Result};

const MAX_NEWTON_ITERS: usize = 100;
const MAX_BISECT_ITERS: usize = 200;
// Tight price tolerance: where vega is tiny (deep ITM/OTM) the price is
// insensitive to sigma, so we must drive the price residual very low to pin
// down sigma to a useful precision.
const PRICE_TOL: f64 = 1e-10;
const VOL_LO: f64 = 1e-9;
const VOL_HI: f64 = 10.0; // 1000% vol upper bracket

/// Solve for the Black-Scholes implied volatility given a market price.
#[allow(clippy::too_many_arguments)]
pub fn implied_vol(
    market_price: f64,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    opt: OptionType,
    q: f64,
) -> Result<f64> {
    if !market_price.is_finite() || market_price < 0.0 {
        return Err(PyoptxError::InvalidInput(
            "market price must be finite and >= 0".to_string(),
        ));
    }
    if s <= 0.0 || k <= 0.0 {
        return Err(PyoptxError::InvalidInput("S and K must be > 0".to_string()));
    }
    if t <= 0.0 {
        return Err(PyoptxError::InvalidInput(
            "T must be > 0 to imply a volatility".to_string(),
        ));
    }

    // No-arbitrage bounds. The price must lie between the intrinsic (discounted
    // forward) lower bound and the asset/strike upper bound.
    let disc_q = (-q * t).exp();
    let disc_r = (-r * t).exp();
    let lower = match opt {
        OptionType::Call => (s * disc_q - k * disc_r).max(0.0),
        OptionType::Put => (k * disc_r - s * disc_q).max(0.0),
    };
    let upper = match opt {
        OptionType::Call => s * disc_q,
        OptionType::Put => k * disc_r,
    };
    // Allow a tiny tolerance for prices sitting exactly on a bound.
    if market_price < lower - 1e-10 || market_price > upper + 1e-10 {
        return Err(PyoptxError::InvalidInput(format!(
            "price {market_price} outside no-arbitrage bounds [{lower:.6}, {upper:.6}]"
        )));
    }

    let f = |sigma: f64| -> Result<f64> {
        Ok(black_scholes::price(s, k, t, r, sigma, opt, q)? - market_price)
    };

    // --- Newton-Raphson, seeded by Brenner-Subrahmanyam ---
    // σ₀ ≈ √(2π/T) · price / S  is a good at-the-money starting guess.
    let mut sigma =
        ((2.0 * std::f64::consts::PI / t).sqrt() * market_price / s).clamp(VOL_LO, VOL_HI);

    for _ in 0..MAX_NEWTON_ITERS {
        let diff = f(sigma)?;
        if diff.abs() < PRICE_TOL {
            return Ok(sigma);
        }
        let v = black_scholes::vega(s, k, t, r, sigma, opt, q)?;
        if v.abs() < 1e-12 {
            break; // vega too small — hand off to bisection
        }
        let next = sigma - diff / v;
        if !next.is_finite() || next <= VOL_LO || next >= VOL_HI {
            break; // stepped out of the valid bracket — hand off to bisection
        }
        sigma = next;
    }

    // --- Bisection fallback over [VOL_LO, VOL_HI] ---
    let mut lo = VOL_LO;
    let mut hi = VOL_HI;
    let mut f_lo = f(lo)?;
    let f_hi = f(hi)?;
    // Price is monotonic in sigma, so the root is bracketed iff the signs differ.
    if f_lo.signum() == f_hi.signum() {
        return Err(PyoptxError::Convergence(format!(
            "could not bracket a root in [{VOL_LO}, {VOL_HI}] for price {market_price}"
        )));
    }
    for _ in 0..MAX_BISECT_ITERS {
        let mid = 0.5 * (lo + hi);
        let f_mid = f(mid)?;
        if f_mid.abs() < PRICE_TOL || (hi - lo) < 1e-12 {
            return Ok(mid);
        }
        if f_mid.signum() == f_lo.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }

    Err(PyoptxError::Convergence(
        "implied vol did not converge within iteration budget".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_call() {
        let (s, k, t, r, sigma, q) = (100.0, 100.0, 1.0, 0.05, 0.2, 0.0);
        let p = black_scholes::price(s, k, t, r, sigma, OptionType::Call, q).unwrap();
        let iv = implied_vol(p, s, k, t, r, OptionType::Call, q).unwrap();
        assert!((iv - sigma).abs() < 1e-6, "iv={iv} sigma={sigma}");
    }

    #[test]
    fn round_trip_many_strikes_and_vols() {
        let (s, t, r) = (100.0, 0.5, 0.03);
        for &k in &[70.0, 90.0, 100.0, 110.0, 130.0] {
            for &sigma in &[0.1, 0.2, 0.35, 0.6, 0.9] {
                for &opt in &[OptionType::Call, OptionType::Put] {
                    let p = black_scholes::price(s, k, t, r, sigma, opt, 0.0).unwrap();
                    if p < 1e-6 {
                        continue; // negligible price — IV is ill-conditioned
                    }
                    let iv = implied_vol(p, s, k, t, r, opt, 0.0).unwrap();
                    assert!(
                        (iv - sigma).abs() < 1e-4,
                        "k={k} sigma={sigma} opt={opt:?} -> iv={iv}"
                    );
                }
            }
        }
    }

    #[test]
    fn deep_itm_uses_bisection_fallback() {
        // Deep ITM: tiny vega, Newton struggles, bisection should still recover it.
        let (s, k, t, r, sigma) = (200.0, 100.0, 1.0, 0.05, 0.25);
        let p = black_scholes::price(s, k, t, r, sigma, OptionType::Call, 0.0).unwrap();
        let iv = implied_vol(p, s, k, t, r, OptionType::Call, 0.0).unwrap();
        assert!((iv - sigma).abs() < 1e-4, "iv={iv}");
    }

    #[test]
    fn rejects_price_outside_bounds() {
        // A call cannot be worth more than the (dividend-adjusted) spot.
        assert!(implied_vol(150.0, 100.0, 100.0, 1.0, 0.05, OptionType::Call, 0.0).is_err());
        // A call below its intrinsic (discounted forward) lower bound is impossible:
        // S=120, K=100, r=0.05, T=1 -> lower bound ≈ 120 - 95.12 = 24.88.
        assert!(implied_vol(10.0, 120.0, 100.0, 1.0, 0.05, OptionType::Call, 0.0).is_err());
    }

    #[test]
    fn rejects_nonpositive_t() {
        assert!(implied_vol(5.0, 100.0, 100.0, 0.0, 0.05, OptionType::Call, 0.0).is_err());
    }
}

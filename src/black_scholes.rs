//! Black-Scholes-Merton European option pricing and first-order Greeks.
//!
//! All formulas include a continuous dividend yield `q` (set `q = 0` for a
//! non-dividend-paying underlying). Prices and Greeks are expressed per unit of
//! their respective inputs:
//!   * vega is per `1.0` change in volatility (i.e. per 100 vol points),
//!   * theta is per year,
//!   * rho is per `1.0` change in the rate.
//!
//! Edge cases are handled rather than panicking:
//!   * `T == 0`  -> the option is worth its intrinsic value,
//!   * `sigma == 0` -> the payoff is deterministic (discounted forward intrinsic).

use crate::distributions::{norm_cdf, norm_pdf};
use crate::error::{PyoptxError, Result};

/// Call or put.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    /// `+1` for a call, `-1` for a put — handy in the symmetric formulas below.
    #[inline]
    fn sign(self) -> f64 {
        match self {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        }
    }

    /// Parse from a string, accepting common spellings.
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "call" | "c" => Ok(OptionType::Call),
            "put" | "p" => Ok(OptionType::Put),
            other => Err(PyoptxError::InvalidInput(format!(
                "option_type must be 'call' or 'put', got '{other}'"
            ))),
        }
    }
}

/// The five first-order Greeks bundled together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

/// Validate the common set of inputs shared by every routine.
fn validate(s: f64, k: f64, t: f64, sigma: f64) -> Result<()> {
    if !(s.is_finite() && k.is_finite() && t.is_finite() && sigma.is_finite()) {
        return Err(PyoptxError::InvalidInput(
            "inputs must be finite".to_string(),
        ));
    }
    if s <= 0.0 {
        return Err(PyoptxError::InvalidInput("spot S must be > 0".to_string()));
    }
    if k <= 0.0 {
        return Err(PyoptxError::InvalidInput(
            "strike K must be > 0".to_string(),
        ));
    }
    if t < 0.0 {
        return Err(PyoptxError::InvalidInput(
            "time to expiry T must be >= 0".to_string(),
        ));
    }
    if sigma < 0.0 {
        return Err(PyoptxError::InvalidInput(
            "volatility sigma must be >= 0".to_string(),
        ));
    }
    Ok(())
}

/// `d1` and `d2` of the Black-Scholes formula. Only valid for `T > 0`, `sigma > 0`.
#[inline]
fn d1_d2(s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> (f64, f64) {
    let vol_sqrt_t = sigma * t.sqrt();
    // d1 = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;
    (d1, d2)
}

/// Intrinsic value of the option (its value at expiry, undiscounted).
#[inline]
fn intrinsic(s: f64, k: f64, opt: OptionType) -> f64 {
    match opt {
        OptionType::Call => (s - k).max(0.0),
        OptionType::Put => (k - s).max(0.0),
    }
}

/// European option price under Black-Scholes-Merton.
///
/// `call = S·e^{-qT}·Φ(d1) − K·e^{-rT}·Φ(d2)`
/// `put  = K·e^{-rT}·Φ(−d2) − S·e^{-qT}·Φ(−d1)`
pub fn price(s: f64, k: f64, t: f64, r: f64, sigma: f64, opt: OptionType, q: f64) -> Result<f64> {
    validate(s, k, t, sigma)?;

    // At expiry the option is worth exactly its intrinsic value.
    if t == 0.0 {
        return Ok(intrinsic(s, k, opt));
    }
    // Zero volatility: the terminal spot is the deterministic forward
    // F = S·e^{(r−q)T}; the option is the discounted intrinsic of that forward.
    if sigma == 0.0 {
        let fwd = s * ((r - q) * t).exp();
        let payoff = intrinsic(fwd, k, opt);
        return Ok((-r * t).exp() * payoff);
    }

    let (d1, d2) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    let disc_r = (-r * t).exp();
    let sign = opt.sign();
    // Unified call/put formula using the sign convention.
    let price = sign * (s * disc_q * norm_cdf(sign * d1) - k * disc_r * norm_cdf(sign * d2));
    Ok(price)
}

/// Delta — ∂price/∂S.
///
/// `call = e^{-qT}·Φ(d1)`, `put = −e^{-qT}·Φ(−d1)`
pub fn delta(s: f64, k: f64, t: f64, r: f64, sigma: f64, opt: OptionType, q: f64) -> Result<f64> {
    validate(s, k, t, sigma)?;
    if t == 0.0 || sigma == 0.0 {
        // Degenerate: delta is the step function of the (forward) moneyness.
        let fwd = s * ((r - q) * t).exp();
        let itm = match opt {
            OptionType::Call => fwd > k,
            OptionType::Put => fwd < k,
        };
        let disc_q = (-q * t).exp();
        return Ok(if itm { opt.sign() * disc_q } else { 0.0 });
    }
    let (d1, _) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    Ok(opt.sign() * disc_q * norm_cdf(opt.sign() * d1))
}

/// Gamma — ∂²price/∂S². Identical for calls and puts.
///
/// `e^{-qT}·φ(d1) / (S·σ·√T)`
pub fn gamma(s: f64, k: f64, t: f64, r: f64, sigma: f64, _opt: OptionType, q: f64) -> Result<f64> {
    validate(s, k, t, sigma)?;
    if t == 0.0 || sigma == 0.0 {
        return Ok(0.0);
    }
    let (d1, _) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    Ok(disc_q * norm_pdf(d1) / (s * sigma * t.sqrt()))
}

/// Vega — ∂price/∂σ, per unit (1.0) of volatility. Identical for calls and puts.
///
/// `S·e^{-qT}·φ(d1)·√T`
pub fn vega(s: f64, k: f64, t: f64, r: f64, sigma: f64, _opt: OptionType, q: f64) -> Result<f64> {
    validate(s, k, t, sigma)?;
    if t == 0.0 || sigma == 0.0 {
        return Ok(0.0);
    }
    let (d1, _) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    Ok(s * disc_q * norm_pdf(d1) * t.sqrt())
}

/// Theta — ∂price/∂t, per year (negative for most long options).
pub fn theta(s: f64, k: f64, t: f64, r: f64, sigma: f64, opt: OptionType, q: f64) -> Result<f64> {
    validate(s, k, t, sigma)?;
    if t == 0.0 || sigma == 0.0 {
        return Ok(0.0);
    }
    let (d1, d2) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    let disc_r = (-r * t).exp();
    // Common term: time decay of the option's "optionality".
    let term1 = -(s * disc_q * norm_pdf(d1) * sigma) / (2.0 * t.sqrt());
    let sign = opt.sign();
    // sign·[ q·S·e^{-qT}·Φ(sign·d1) − r·K·e^{-rT}·Φ(sign·d2) ]
    let term2 = sign * q * s * disc_q * norm_cdf(sign * d1);
    let term3 = -sign * r * k * disc_r * norm_cdf(sign * d2);
    Ok(term1 + term2 + term3)
}

/// Rho — ∂price/∂r, per unit (1.0) of the interest rate.
///
/// `call = K·T·e^{-rT}·Φ(d2)`, `put = −K·T·e^{-rT}·Φ(−d2)`
pub fn rho(s: f64, k: f64, t: f64, r: f64, sigma: f64, opt: OptionType, q: f64) -> Result<f64> {
    validate(s, k, t, sigma)?;
    if t == 0.0 || sigma == 0.0 {
        return Ok(0.0);
    }
    let (_, d2) = d1_d2(s, k, t, r, q, sigma);
    let disc_r = (-r * t).exp();
    let sign = opt.sign();
    Ok(sign * k * t * disc_r * norm_cdf(sign * d2))
}

/// Compute price and all five Greeks in one pass.
pub fn greeks(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    opt: OptionType,
    q: f64,
) -> Result<Greeks> {
    Ok(Greeks {
        delta: delta(s, k, t, r, sigma, opt, q)?,
        gamma: gamma(s, k, t, r, sigma, opt, q)?,
        vega: vega(s, k, t, r, sigma, opt, q)?,
        theta: theta(s, k, t, r, sigma, opt, q)?,
        rho: rho(s, k, t, r, sigma, opt, q)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hull, "Options, Futures, and Other Derivatives": S=42, K=40, r=0.10,
    // sigma=0.20, T=0.5 gives a call of 4.76 and a put of 0.81.
    #[test]
    fn hull_textbook_values() {
        let c = price(42.0, 40.0, 0.5, 0.10, 0.20, OptionType::Call, 0.0).unwrap();
        let p = price(42.0, 40.0, 0.5, 0.10, 0.20, OptionType::Put, 0.0).unwrap();
        assert!((c - 4.7594).abs() < 1e-3, "call={c}");
        assert!((p - 0.8086).abs() < 1e-3, "put={p}");
    }

    #[test]
    fn put_call_parity() {
        // C − P = S·e^{-qT} − K·e^{-rT}
        let (s, k, t, r, sigma, q) = (100.0, 95.0, 0.75, 0.05, 0.25, 0.02);
        let c = price(s, k, t, r, sigma, OptionType::Call, q).unwrap();
        let p = price(s, k, t, r, sigma, OptionType::Put, q).unwrap();
        let lhs = c - p;
        let rhs = s * (-q * t).exp() - k * (-r * t).exp();
        assert!((lhs - rhs).abs() < 1e-10, "parity off: {lhs} vs {rhs}");
    }

    #[test]
    fn intrinsic_at_expiry() {
        let c = price(110.0, 100.0, 0.0, 0.05, 0.2, OptionType::Call, 0.0).unwrap();
        let p = price(90.0, 100.0, 0.0, 0.05, 0.2, OptionType::Put, 0.0).unwrap();
        assert_eq!(c, 10.0);
        assert_eq!(p, 10.0);
    }

    #[test]
    fn zero_vol_is_discounted_forward_intrinsic() {
        // sigma = 0: deterministic forward F = S·e^{(r-q)T}.
        let (s, k, t, r) = (100.0, 100.0, 1.0, 0.05);
        let c = price(s, k, t, r, 0.0, OptionType::Call, 0.0).unwrap();
        let fwd = s * (r * t).exp();
        let expected = (-r * t).exp() * (fwd - k).max(0.0);
        assert!((c - expected).abs() < 1e-12);
    }

    #[test]
    fn delta_bounds() {
        // Call delta in (0, e^{-qT}); put delta in (-e^{-qT}, 0).
        let cd = delta(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call, 0.0).unwrap();
        let pd = delta(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Put, 0.0).unwrap();
        assert!(cd > 0.0 && cd < 1.0);
        assert!(pd < 0.0 && pd > -1.0);
        // delta_call - delta_put = e^{-qT} = 1 here
        assert!((cd - pd - 1.0).abs() < 1e-10);
    }

    #[test]
    fn vega_gamma_nonnegative_and_shared() {
        let vc = vega(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call, 0.0).unwrap();
        let vp = vega(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Put, 0.0).unwrap();
        let gc = gamma(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call, 0.0).unwrap();
        let gp = gamma(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Put, 0.0).unwrap();
        assert!(vc > 0.0 && gc > 0.0);
        assert!((vc - vp).abs() < 1e-12); // identical for call/put
        assert!((gc - gp).abs() < 1e-12);
    }

    #[test]
    fn greeks_match_finite_difference() {
        let (s, k, t, r, sigma, q) = (100.0, 105.0, 0.5, 0.03, 0.22, 0.01);
        let opt = OptionType::Call;
        let f = |s, t, r, sigma| price(s, k, t, r, sigma, opt, q).unwrap();

        let h = 1e-4;
        let fd_delta = (f(s + h, t, r, sigma) - f(s - h, t, r, sigma)) / (2.0 * h);
        let fd_vega = (f(s, t, r, sigma + h) - f(s, t, r, sigma - h)) / (2.0 * h);
        let fd_rho = (f(s, t, r + h, sigma) - f(s, t, r - h, sigma)) / (2.0 * h);
        // theta = -∂price/∂t
        let fd_theta = -(f(s, t + h, r, sigma) - f(s, t - h, r, sigma)) / (2.0 * h);

        assert!((delta(s, k, t, r, sigma, opt, q).unwrap() - fd_delta).abs() < 1e-5);
        assert!((vega(s, k, t, r, sigma, opt, q).unwrap() - fd_vega).abs() < 1e-4);
        assert!((rho(s, k, t, r, sigma, opt, q).unwrap() - fd_rho).abs() < 1e-4);
        assert!((theta(s, k, t, r, sigma, opt, q).unwrap() - fd_theta).abs() < 1e-3);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(price(-1.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call, 0.0).is_err());
        assert!(price(100.0, 0.0, 1.0, 0.05, 0.2, OptionType::Call, 0.0).is_err());
        assert!(price(100.0, 100.0, -1.0, 0.05, 0.2, OptionType::Call, 0.0).is_err());
    }
}

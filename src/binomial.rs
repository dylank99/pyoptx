//! Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.
//!
//! The tree uses `n` time steps of length `dt = T/n`. Over each step the spot
//! moves up by `u = e^{σ√dt}` or down by `d = 1/u`, with risk-neutral up
//! probability `p = (e^{(r−q)dt} − d) / (u − d)`. Values are rolled back through
//! the tree, discounting by `e^{−r·dt}` per step; for American options each node
//! is maxed against the immediate exercise (intrinsic) value.
//!
//! As `n → ∞` the European price converges to Black-Scholes (verified in tests).

use crate::black_scholes::OptionType;
use crate::error::{PyoptxError, Result};

/// Exercise style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exercise {
    European,
    American,
}

impl Exercise {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "european" | "euro" | "eu" => Ok(Exercise::European),
            "american" | "amer" | "us" => Ok(Exercise::American),
            other => Err(PyoptxError::InvalidInput(format!(
                "exercise must be 'european' or 'american', got '{other}'"
            ))),
        }
    }
}

#[inline]
fn intrinsic(s: f64, k: f64, opt: OptionType) -> f64 {
    match opt {
        OptionType::Call => (s - k).max(0.0),
        OptionType::Put => (k - s).max(0.0),
    }
}

/// Price an option on a CRR binomial tree with `n` steps.
#[allow(clippy::too_many_arguments)]
pub fn price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    opt: OptionType,
    exercise: Exercise,
    n: usize,
    q: f64,
) -> Result<f64> {
    if !(s.is_finite() && k.is_finite() && t.is_finite() && sigma.is_finite()) {
        return Err(PyoptxError::InvalidInput(
            "inputs must be finite".to_string(),
        ));
    }
    if s <= 0.0 || k <= 0.0 {
        return Err(PyoptxError::InvalidInput("S and K must be > 0".to_string()));
    }
    if t < 0.0 {
        return Err(PyoptxError::InvalidInput("T must be >= 0".to_string()));
    }
    if n == 0 {
        return Err(PyoptxError::InvalidInput(
            "steps n must be >= 1".to_string(),
        ));
    }
    // Degenerate cases: no panics, return a sensible value.
    if t == 0.0 {
        return Ok(intrinsic(s, k, opt));
    }
    if sigma <= 0.0 {
        return Err(PyoptxError::InvalidInput(
            "sigma must be > 0 for the binomial tree".to_string(),
        ));
    }

    let dt = t / n as f64;
    let u = (sigma * dt.sqrt()).exp();
    let d = 1.0 / u;
    let disc = (-r * dt).exp();
    let p = (((r - q) * dt).exp() - d) / (u - d);

    // No-arbitrage requires p in [0, 1]; this can fail if dt is too large for
    // the given sigma. Surface it instead of returning a nonsense price.
    if !(0.0..=1.0).contains(&p) {
        return Err(PyoptxError::InvalidInput(format!(
            "risk-neutral probability p={p:.4} outside [0,1]; increase steps n"
        )));
    }

    // Terminal layer: spot at step n with j up-moves is S·u^j·d^(n-j).
    let mut values: Vec<f64> = (0..=n)
        .map(|j| {
            let st = s * u.powi(j as i32) * d.powi((n - j) as i32);
            intrinsic(st, k, opt)
        })
        .collect();

    // Roll back through the tree.
    for step in (0..n).rev() {
        for j in 0..=step {
            // Discounted risk-neutral expectation of the two child nodes.
            let cont = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
            values[j] = match exercise {
                Exercise::European => cont,
                Exercise::American => {
                    let st = s * u.powi(j as i32) * d.powi((step - j) as i32);
                    cont.max(intrinsic(st, k, opt))
                }
            };
        }
    }

    Ok(values[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::black_scholes;

    #[test]
    fn european_converges_to_black_scholes() {
        let (s, k, t, r, sigma, q) = (100.0, 100.0, 1.0, 0.05, 0.2, 0.0);
        let bs = black_scholes::price(s, k, t, r, sigma, OptionType::Call, q).unwrap();
        let tree = price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            Exercise::European,
            2000,
            q,
        )
        .unwrap();
        assert!((tree - bs).abs() < 1e-2, "tree={tree} bs={bs}");
    }

    #[test]
    fn convergence_improves_with_steps() {
        let (s, k, t, r, sigma) = (100.0, 100.0, 1.0, 0.05, 0.2);
        let bs = black_scholes::price(s, k, t, r, sigma, OptionType::Call, 0.0).unwrap();
        let coarse = (price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            Exercise::European,
            10,
            0.0,
        )
        .unwrap()
            - bs)
            .abs();
        let fine = (price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            Exercise::European,
            1000,
            0.0,
        )
        .unwrap()
            - bs)
            .abs();
        assert!(fine < coarse, "fine={fine} should be < coarse={coarse}");
    }

    #[test]
    fn american_call_no_dividends_equals_european() {
        // Without dividends an American call is never exercised early, so it
        // equals the European call.
        let (s, k, t, r, sigma) = (100.0, 95.0, 1.0, 0.05, 0.25);
        let amer = price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            Exercise::American,
            500,
            0.0,
        )
        .unwrap();
        let euro = price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            Exercise::European,
            500,
            0.0,
        )
        .unwrap();
        assert!((amer - euro).abs() < 1e-6, "amer={amer} euro={euro}");
    }

    #[test]
    fn american_put_premium_over_european() {
        // An American put is worth at least its European counterpart.
        let (s, k, t, r, sigma) = (100.0, 110.0, 1.0, 0.08, 0.3);
        let amer = price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            Exercise::American,
            500,
            0.0,
        )
        .unwrap();
        let euro = price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            Exercise::European,
            500,
            0.0,
        )
        .unwrap();
        assert!(amer >= euro - 1e-9, "amer={amer} euro={euro}");
        assert!(
            amer > euro,
            "expect early-exercise premium: amer={amer} euro={euro}"
        );
    }

    #[test]
    fn rejects_zero_steps() {
        assert!(price(
            100.0,
            100.0,
            1.0,
            0.05,
            0.2,
            OptionType::Call,
            Exercise::European,
            0,
            0.0
        )
        .is_err());
    }
}

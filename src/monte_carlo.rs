//! Monte Carlo pricing of European options under geometric Brownian motion.
//!
//! Under the risk-neutral measure the terminal spot is
//! `S_T = S·exp[(r − q − σ²/2)·T + σ·√T·Z]`, with `Z ~ N(0,1)`.
//! The option value is the discounted expectation `e^{−rT}·E[payoff(S_T)]`,
//! estimated by averaging over simulated paths.
//!
//! Two variance-reduction techniques are available and can be combined:
//!   * **Antithetic variates** — each draw `Z` is paired with `−Z`, halving the
//!     number of normal draws and cancelling much of the sampling noise.
//!   * **Control variate** — the discounted terminal spot `e^{−rT}·S_T` has the
//!     analytically known mean `S·e^{−qT}` (from the GBM/Black-Scholes model).
//!     Regressing the payoff on this control removes the variance it explains.
//!
//! The estimator's standard error is always reported so callers can gauge the
//! Monte Carlo noise. A seedable RNG makes every run reproducible.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::black_scholes::OptionType;
use crate::error::{PyoptxError, Result};

/// Result of a Monte Carlo pricing run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McResult {
    /// Point estimate of the option price.
    pub price: f64,
    /// Standard error of the estimate (≈ how far the estimate may be from truth).
    pub std_error: f64,
}

#[inline]
fn payoff(st: f64, k: f64, opt: OptionType) -> f64 {
    match opt {
        OptionType::Call => (st - k).max(0.0),
        OptionType::Put => (k - st).max(0.0),
    }
}

/// Price a European option by Monte Carlo simulation.
///
/// `n_paths` is the number of independent estimator samples. With antithetic
/// variates each sample averages a `Z`/`−Z` pair (i.e. two GBM draws).
#[allow(clippy::too_many_arguments)]
pub fn price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    opt: OptionType,
    n_paths: usize,
    seed: u64,
    q: f64,
    antithetic: bool,
    control_variate: bool,
) -> Result<McResult> {
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
    if sigma < 0.0 {
        return Err(PyoptxError::InvalidInput("sigma must be >= 0".to_string()));
    }
    if n_paths == 0 {
        return Err(PyoptxError::InvalidInput(
            "n_paths must be >= 1".to_string(),
        ));
    }

    // Deterministic payoff at expiry / zero vol: no sampling noise.
    if t == 0.0 || sigma == 0.0 {
        let fwd = s * ((r - q) * t).exp();
        let price = (-r * t).exp() * payoff(fwd, k, opt);
        return Ok(McResult {
            price,
            std_error: 0.0,
        });
    }

    let disc = (-r * t).exp();
    let drift = (r - q - 0.5 * sigma * sigma) * t;
    let vol = sigma * t.sqrt();
    // Known mean of the control X = e^{−rT}·S_T  ⇒  E[X] = S·e^{−qT}.
    let control_mean = s * (-q * t).exp();

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = StandardNormal;

    // Collect the per-sample estimator value `y` and (optionally) control `x`.
    let mut ys: Vec<f64> = Vec::with_capacity(n_paths);
    let mut xs: Vec<f64> = if control_variate {
        Vec::with_capacity(n_paths)
    } else {
        Vec::new()
    };

    for _ in 0..n_paths {
        let z: f64 = normal.sample(&mut rng);
        if antithetic {
            let st_p = s * (drift + vol * z).exp();
            let st_m = s * (drift - vol * z).exp();
            // Average the +Z / −Z pair into a single estimator sample.
            let y = disc * 0.5 * (payoff(st_p, k, opt) + payoff(st_m, k, opt));
            ys.push(y);
            if control_variate {
                let x = disc * 0.5 * (st_p + st_m);
                xs.push(x);
            }
        } else {
            let st = s * (drift + vol * z).exp();
            ys.push(disc * payoff(st, k, opt));
            if control_variate {
                xs.push(disc * st);
            }
        }
    }

    let n = ys.len() as f64;
    let mean_y = ys.iter().sum::<f64>() / n;

    // Without a control variate, this is the plain sample mean.
    let (estimate, sample_var) = if control_variate {
        let mean_x = xs.iter().sum::<f64>() / n;
        // Optimal coefficient c* = Cov(Y, X) / Var(X), estimated from samples.
        let mut cov = 0.0;
        let mut var_x = 0.0;
        for i in 0..ys.len() {
            let dy = ys[i] - mean_y;
            let dx = xs[i] - mean_x;
            cov += dy * dx;
            var_x += dx * dx;
        }
        let c = if var_x > 0.0 { cov / var_x } else { 0.0 };
        // Adjusted samples: Y' = Y − c·(X − E[X]).
        let adj: Vec<f64> = (0..ys.len())
            .map(|i| ys[i] - c * (xs[i] - control_mean))
            .collect();
        let mean_adj = adj.iter().sum::<f64>() / n;
        let var_adj = if n > 1.0 {
            adj.iter().map(|v| (v - mean_adj).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };
        (mean_adj, var_adj)
    } else {
        let var_y = if n > 1.0 {
            ys.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };
        (mean_y, var_y)
    };

    let std_error = (sample_var / n).sqrt();
    Ok(McResult {
        price: estimate,
        std_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::black_scholes;

    #[test]
    fn mc_within_a_few_standard_errors_of_black_scholes() {
        let (s, k, t, r, sigma, q) = (100.0, 100.0, 1.0, 0.05, 0.2, 0.0);
        let bs = black_scholes::price(s, k, t, r, sigma, OptionType::Call, q).unwrap();
        let res = price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            200_000,
            42,
            q,
            true,
            false,
        )
        .unwrap();
        // The estimate should sit within ~4 standard errors of the true price.
        assert!(
            (res.price - bs).abs() < 4.0 * res.std_error + 1e-9,
            "mc={} bs={} se={}",
            res.price,
            bs,
            res.std_error
        );
    }

    #[test]
    fn control_variate_reduces_standard_error() {
        let (s, k, t, r, sigma) = (100.0, 100.0, 1.0, 0.05, 0.2);
        let plain = price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            50_000,
            7,
            0.0,
            false,
            false,
        )
        .unwrap();
        let cv = price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            50_000,
            7,
            0.0,
            false,
            true,
        )
        .unwrap();
        assert!(
            cv.std_error < plain.std_error,
            "cv se={} should be < plain se={}",
            cv.std_error,
            plain.std_error
        );
    }

    #[test]
    fn reproducible_with_same_seed() {
        let a = price(
            100.0,
            100.0,
            1.0,
            0.05,
            0.2,
            OptionType::Call,
            10_000,
            123,
            0.0,
            true,
            true,
        )
        .unwrap();
        let b = price(
            100.0,
            100.0,
            1.0,
            0.05,
            0.2,
            OptionType::Call,
            10_000,
            123,
            0.0,
            true,
            true,
        )
        .unwrap();
        assert_eq!(a.price, b.price);
        assert_eq!(a.std_error, b.std_error);
    }

    #[test]
    fn put_also_prices_correctly() {
        let (s, k, t, r, sigma) = (100.0, 105.0, 0.5, 0.03, 0.25);
        let bs = black_scholes::price(s, k, t, r, sigma, OptionType::Put, 0.0).unwrap();
        let res = price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            200_000,
            99,
            0.0,
            true,
            true,
        )
        .unwrap();
        assert!(
            (res.price - bs).abs() < 4.0 * res.std_error + 1e-3,
            "mc={} bs={}",
            res.price,
            bs
        );
    }
}

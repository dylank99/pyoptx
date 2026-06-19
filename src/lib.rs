//! `pyoptx_core` — the Rust numerical core behind the `pyoptx` Python library.
//!
//! Fast options pricing and risk: Black-Scholes, binomial trees, and Monte
//! Carlo with full Greeks and implied volatility. The numerical modules are
//! pure Rust (no Python dependency) so they can be unit-tested directly with
//! `cargo test`. The PyO3 bindings at the bottom of this file are compiled only
//! when the `extension-module` feature is enabled (which maturin does when it
//! builds the wheel), and are exposed to Python as `pyoptx._core`.

pub mod binomial;
pub mod black_scholes;
pub mod distributions;
pub mod error;
pub mod implied_vol;
pub mod monte_carlo;

// ---------------------------------------------------------------------------
// Python bindings (only built for the extension module).
// ---------------------------------------------------------------------------
#[cfg(feature = "extension-module")]
mod python_bindings {
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use crate::binomial::{self, Exercise};
    use crate::black_scholes::{self, OptionType};
    use crate::error::PyoptxError;
    use crate::implied_vol;
    use crate::monte_carlo;

    /// Surface core errors as idiomatic Python exceptions.
    impl From<PyoptxError> for PyErr {
        fn from(e: PyoptxError) -> Self {
            match e {
                PyoptxError::InvalidInput(msg) => PyValueError::new_err(msg),
                PyoptxError::Convergence(msg) => PyRuntimeError::new_err(msg),
            }
        }
    }

    fn parse_opt(s: &str) -> PyResult<OptionType> {
        Ok(OptionType::parse(s)?)
    }

    /// Black-Scholes-Merton price of a European option.
    #[pyfunction]
    #[pyo3(signature = (s, k, t, r, sigma, option_type="call", q=0.0))]
    fn bs_price(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        option_type: &str,
        q: f64,
    ) -> PyResult<f64> {
        Ok(black_scholes::price(
            s,
            k,
            t,
            r,
            sigma,
            parse_opt(option_type)?,
            q,
        )?)
    }

    #[pyfunction]
    #[pyo3(signature = (s, k, t, r, sigma, option_type="call", q=0.0))]
    fn bs_delta(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        option_type: &str,
        q: f64,
    ) -> PyResult<f64> {
        Ok(black_scholes::delta(
            s,
            k,
            t,
            r,
            sigma,
            parse_opt(option_type)?,
            q,
        )?)
    }

    #[pyfunction]
    #[pyo3(signature = (s, k, t, r, sigma, option_type="call", q=0.0))]
    fn bs_gamma(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        option_type: &str,
        q: f64,
    ) -> PyResult<f64> {
        Ok(black_scholes::gamma(
            s,
            k,
            t,
            r,
            sigma,
            parse_opt(option_type)?,
            q,
        )?)
    }

    #[pyfunction]
    #[pyo3(signature = (s, k, t, r, sigma, option_type="call", q=0.0))]
    fn bs_vega(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        option_type: &str,
        q: f64,
    ) -> PyResult<f64> {
        Ok(black_scholes::vega(
            s,
            k,
            t,
            r,
            sigma,
            parse_opt(option_type)?,
            q,
        )?)
    }

    #[pyfunction]
    #[pyo3(signature = (s, k, t, r, sigma, option_type="call", q=0.0))]
    fn bs_theta(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        option_type: &str,
        q: f64,
    ) -> PyResult<f64> {
        Ok(black_scholes::theta(
            s,
            k,
            t,
            r,
            sigma,
            parse_opt(option_type)?,
            q,
        )?)
    }

    #[pyfunction]
    #[pyo3(signature = (s, k, t, r, sigma, option_type="call", q=0.0))]
    fn bs_rho(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        option_type: &str,
        q: f64,
    ) -> PyResult<f64> {
        Ok(black_scholes::rho(
            s,
            k,
            t,
            r,
            sigma,
            parse_opt(option_type)?,
            q,
        )?)
    }

    /// Price plus all five first-order Greeks as a dict.
    #[pyfunction]
    #[pyo3(signature = (s, k, t, r, sigma, option_type="call", q=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn bs_greeks<'py>(
        py: Python<'py>,
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        option_type: &str,
        q: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let opt = parse_opt(option_type)?;
        let g = black_scholes::greeks(s, k, t, r, sigma, opt, q)?;
        let price = black_scholes::price(s, k, t, r, sigma, opt, q)?;
        let d = PyDict::new(py);
        d.set_item("price", price)?;
        d.set_item("delta", g.delta)?;
        d.set_item("gamma", g.gamma)?;
        d.set_item("vega", g.vega)?;
        d.set_item("theta", g.theta)?;
        d.set_item("rho", g.rho)?;
        Ok(d)
    }

    /// Cox-Ross-Rubinstein binomial price (European or American).
    #[pyfunction]
    #[pyo3(signature = (s, k, t, r, sigma, option_type="call", exercise="european", steps=512, q=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn binomial_price(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        option_type: &str,
        exercise: &str,
        steps: usize,
        q: f64,
    ) -> PyResult<f64> {
        let opt = parse_opt(option_type)?;
        let ex = Exercise::parse(exercise)?;
        Ok(binomial::price(s, k, t, r, sigma, opt, ex, steps, q)?)
    }

    /// Monte Carlo price of a European option. Returns `(price, std_error)`.
    #[pyfunction]
    #[pyo3(signature = (s, k, t, r, sigma, option_type="call", n_paths=100_000, seed=0, q=0.0, antithetic=true, control_variate=true))]
    #[allow(clippy::too_many_arguments)]
    fn mc_price(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        option_type: &str,
        n_paths: usize,
        seed: u64,
        q: f64,
        antithetic: bool,
        control_variate: bool,
    ) -> PyResult<(f64, f64)> {
        let opt = parse_opt(option_type)?;
        let res = monte_carlo::price(
            s,
            k,
            t,
            r,
            sigma,
            opt,
            n_paths,
            seed,
            q,
            antithetic,
            control_variate,
        )?;
        Ok((res.price, res.std_error))
    }

    /// Black-Scholes implied volatility from a market price.
    #[pyfunction]
    #[pyo3(signature = (market_price, s, k, t, r, option_type="call", q=0.0))]
    fn implied_volatility(
        market_price: f64,
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        option_type: &str,
        q: f64,
    ) -> PyResult<f64> {
        let opt = parse_opt(option_type)?;
        Ok(implied_vol::implied_vol(market_price, s, k, t, r, opt, q)?)
    }

    /// The compiled module, importable as `pyoptx._core`.
    #[pymodule]
    fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add(
            "__doc__",
            "Rust core for pyoptx: pricing, Greeks, MC and implied vol.",
        )?;
        m.add_function(wrap_pyfunction!(bs_price, m)?)?;
        m.add_function(wrap_pyfunction!(bs_delta, m)?)?;
        m.add_function(wrap_pyfunction!(bs_gamma, m)?)?;
        m.add_function(wrap_pyfunction!(bs_vega, m)?)?;
        m.add_function(wrap_pyfunction!(bs_theta, m)?)?;
        m.add_function(wrap_pyfunction!(bs_rho, m)?)?;
        m.add_function(wrap_pyfunction!(bs_greeks, m)?)?;
        m.add_function(wrap_pyfunction!(binomial_price, m)?)?;
        m.add_function(wrap_pyfunction!(mc_price, m)?)?;
        m.add_function(wrap_pyfunction!(implied_volatility, m)?)?;
        Ok(())
    }
}

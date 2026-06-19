//! `pyoptx_core` — the Rust numerical core behind the `pyoptx` Python library.
//!
//! The numerical modules are pure Rust (no Python dependency) so they can be
//! unit-tested directly with `cargo test`. PyO3 bindings are added later behind
//! the `extension-module` feature and exposed to Python as `pyoptx._core`.

pub mod binomial;
pub mod black_scholes;
pub mod distributions;
pub mod error;
pub mod implied_vol;
pub mod monte_carlo;

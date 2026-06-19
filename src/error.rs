//! Error type shared by the numerical core.
//!
//! The math modules are pure Rust and never panic on bad input: they return
//! `Result<_, PyoptxError>`. The PyO3 bindings (in `lib.rs`) convert these into
//! the appropriate Python exceptions.

use std::fmt;

/// All recoverable error conditions produced by the pricing/risk routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyoptxError {
    /// An input was outside its valid domain (e.g. negative spot, T < 0).
    InvalidInput(String),
    /// An iterative solver failed to converge within its budget.
    Convergence(String),
}

impl fmt::Display for PyoptxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PyoptxError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            PyoptxError::Convergence(msg) => write!(f, "convergence error: {msg}"),
        }
    }
}

impl std::error::Error for PyoptxError {}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, PyoptxError>;

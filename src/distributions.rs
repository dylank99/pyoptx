//! Standard normal distribution helpers used by the pricing models.
//!
//! Everything here is pure `f64` arithmetic with no external dependencies so it
//! can be unit-tested without Python in the loop.

use std::f64::consts::PI;

/// Standard normal probability density function.
///
/// `φ(x) = exp(-x² / 2) / √(2π)`
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal cumulative distribution function.
///
/// Uses the Hart algorithm as given by Graeme West, "Better Approximations to
/// Cumulative Normal Functions" (2009). It is accurate to roughly double
/// precision (|error| ~ 1e-15) across the whole real line, which matters for
/// stable Greeks and implied-vol round-trips.
#[inline]
// Constants are quoted verbatim from West (2009); keep their published digits.
#[allow(clippy::excessive_precision)]
pub fn norm_cdf(x: f64) -> f64 {
    let xabs = x.abs();
    if xabs > 37.0 {
        // The tail is below f64's smallest normal; clamp to the exact limit.
        return if x > 0.0 { 1.0 } else { 0.0 };
    }

    let exponential = (-0.5 * xabs * xabs).exp();
    // `tail` is P(X < -xabs), the small lower-tail probability.
    let tail = if xabs < 7.071_067_811_865_475 {
        // Rational (Hart) approximation for the central region.
        let mut num = 3.526_249_659_989_109e-2 * xabs + 0.700_383_064_443_688;
        num = num * xabs + 6.373_962_203_531_650;
        num = num * xabs + 33.912_866_078_383_0;
        num = num * xabs + 112.079_291_497_871;
        num = num * xabs + 221.213_596_169_931;
        num = num * xabs + 220.206_867_912_376;
        num *= exponential;

        let mut den = 8.838_834_764_831_84e-2 * xabs + 1.755_667_163_182_64;
        den = den * xabs + 16.064_177_579_207_0;
        den = den * xabs + 86.780_732_202_946_1;
        den = den * xabs + 296.564_248_779_674;
        den = den * xabs + 637.333_633_378_831;
        den = den * xabs + 793.826_512_519_948;
        den = den * xabs + 440.413_735_824_752;

        num / den
    } else {
        // Continued-fraction tail for |x| in [7.07, 37].
        let mut build = xabs + 0.65;
        build = xabs + 4.0 / build;
        build = xabs + 3.0 / build;
        build = xabs + 2.0 / build;
        build = xabs + 1.0 / build;
        exponential / build / 2.506_628_274_631_000_5
    };

    if x > 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_known_values() {
        // φ(0) = 1/√(2π) ≈ 0.3989422804
        assert!((norm_pdf(0.0) - 0.3989422804014327).abs() < 1e-12);
        // symmetry
        assert!((norm_pdf(1.3) - norm_pdf(-1.3)).abs() < 1e-15);
    }

    #[test]
    fn cdf_known_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-12);
        // Φ(1.96) ≈ 0.9750 (the classic two-sided 95% point)
        assert!((norm_cdf(1.96) - 0.9750021048).abs() < 1e-6);
        assert!((norm_cdf(-1.96) - 0.0249978952).abs() < 1e-6);
        // tails
        assert!((norm_cdf(3.0) - 0.9986501020).abs() < 1e-6);
    }

    #[test]
    fn cdf_symmetry_and_monotonic() {
        // Φ(x) + Φ(-x) = 1
        for &x in &[0.1, 0.5, 1.0, 2.5, 4.0] {
            assert!((norm_cdf(x) + norm_cdf(-x) - 1.0).abs() < 1e-6);
        }
        // monotonic increasing
        assert!(norm_cdf(-1.0) < norm_cdf(0.0));
        assert!(norm_cdf(0.0) < norm_cdf(1.0));
    }
}

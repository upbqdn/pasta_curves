//! Variable-time MSM with the curve-specific planner and a generic Booth fallback.
//! Ported from zakura-pasta-curves 5.4.0, src/arithmetic/multiexp.rs.

use alloc::{vec, vec::Vec};
use ff::PrimeField;
use group::Group as _;
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;

use super::{CurveAffine, CurveExt};

/// Computes a public-scalar MSM; variable-time in scalars and bases.
/// Panics on mismatched lengths.
pub fn best_multiexp_vartime<C: CurveAffine>(coeffs: &[C::ScalarExt], bases: &[C]) -> C::CurveExt {
    assert_eq!(coeffs.len(), bases.len());
    if let Some(result) = C::CurveExt::try_multiexp_vartime(coeffs, bases) {
        return result;
    }
    signed_booth_multiexp(coeffs, bases, current_num_threads())
}

/// Floors of `e^k` for `k = 4..=22`.
///
/// These preserve the established `ceil(ln(terms))` window schedule without
/// requiring floating-point transcendental functions, which are unavailable
/// in `no_std` builds.
const DEFAULT_MULTIEXP_WINDOW_THRESHOLDS: [u32; 19] = [
    54,
    148,
    403,
    1_096,
    2_980,
    8_103,
    22_026,
    59_874,
    162_754,
    442_413,
    1_202_604,
    3_269_017,
    8_886_110,
    24_154_952,
    65_659_969,
    178_482_300,
    485_165_195,
    1_318_815_734,
    3_584_912_846,
];

pub(crate) fn default_multiexp_window_bits(terms: usize) -> Option<usize> {
    if terms < 4 {
        Some(1)
    } else if terms < 32 {
        Some(3)
    } else {
        let terms = u32::try_from(terms).ok()?;
        let threshold = DEFAULT_MULTIEXP_WINDOW_THRESHOLDS
            .iter()
            .position(|upper| terms <= *upper)
            .unwrap_or(DEFAULT_MULTIEXP_WINDOW_THRESHOLDS.len());
        Some(4 + threshold)
    }
}

fn current_num_threads() -> usize {
    #[cfg(feature = "multicore")]
    {
        maybe_rayon::current_num_threads()
    }
    #[cfg(not(feature = "multicore"))]
    {
        1
    }
}

/// The generic Signed-Booth bucket method (formerly the body of
/// `halo2_proofs::arithmetic::best_multiexp`).
pub(crate) fn signed_booth_multiexp<C: CurveAffine>(
    coeffs: &[C::ScalarExt],
    bases: &[C],
    num_threads: usize,
) -> C::CurveExt {
    // `ceil(ln(terms))` above 32 terms, without floating point.
    let c = default_multiexp_window_bits(bases.len())
        .unwrap_or(4 + DEFAULT_MULTIEXP_WINDOW_THRESHOLDS.len());

    // Convert to canonical representations once instead of once per window.
    let coeffs = coeffs.iter().map(PrimeField::to_repr).collect::<Vec<_>>();

    let mut multi_buckets: Vec<BoothBuckets<C>> =
        vec![BoothBuckets::new(c); booth_window_count(&coeffs, c)];

    // Each window's bucket sum is independent, so workers fill windows in
    // parallel. The windows are then combined on this thread by Horner's
    // rule, which costs about one scalar's worth of doublings in total;
    // shifting every window by its own offset in parallel instead would
    // spend about `c * windows^2 / 2` doublings.
    #[cfg(feature = "multicore")]
    let window_sums = if should_parallelize_multiexp(coeffs.len(), num_threads) {
        multi_buckets
            .par_iter_mut()
            .enumerate()
            .map(|(i, buckets)| buckets.sum(&coeffs, bases, i))
            .collect::<Vec<_>>()
    } else {
        serial_window_sums(&mut multi_buckets, &coeffs, bases)
    };
    #[cfg(not(feature = "multicore"))]
    let window_sums = {
        let _ = num_threads;
        serial_window_sums(&mut multi_buckets, &coeffs, bases)
    };

    window_sums
        .into_iter()
        .rev()
        .fold(C::CurveExt::identity(), |mut sum, window| {
            // restore original evaluation point
            (0..c).for_each(|_| sum = sum.double());
            sum + window
        })
}

fn serial_window_sums<C: CurveAffine>(
    multi_buckets: &mut [BoothBuckets<C>],
    coeffs: &[<C::ScalarExt as PrimeField>::Repr],
    bases: &[C],
) -> Vec<C::CurveExt> {
    multi_buckets
        .iter_mut()
        .enumerate()
        .map(|(i, buckets)| buckets.sum(coeffs, bases, i))
        .collect()
}

#[cfg(feature = "multicore")]
fn should_parallelize_multiexp(num_coeffs: usize, num_threads: usize) -> bool {
    // The parallel algorithm shifts each window result independently. With a
    // single worker, this only adds doublings compared to serial evaluation.
    num_threads > 1 && num_coeffs > num_threads
}

#[derive(Clone, Copy)]
enum Bucket<C: CurveAffine> {
    None,
    Affine(C),
    Projective(C::CurveExt),
}

impl<C: CurveAffine> Bucket<C> {
    fn add_assign(&mut self, other: C) {
        *self = match *self {
            Bucket::None => Bucket::Affine(other),
            Bucket::Affine(a) => Bucket::Projective(a + other),
            Bucket::Projective(mut a) => {
                a += other;
                Bucket::Projective(a)
            }
        }
    }

    fn add(self, mut other: C::CurveExt) -> C::CurveExt {
        match self {
            Bucket::None => other,
            Bucket::Affine(a) => {
                other += a;
                other
            }
            Bucket::Projective(a) => other + &a,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoothDigit {
    magnitude: usize,
    negative: bool,
}

fn booth_digit(bytes: &[u8], window_bits: usize, window: usize) -> BoothDigit {
    debug_assert!(window_bits > 0);
    debug_assert!(window_bits < u64::BITS as usize);

    let window_start = window * window_bits;
    let byte_start = window_start / 8;
    let mut encoded = [0; 8];
    if byte_start < bytes.len() {
        for (slot, byte) in encoded.iter_mut().zip(&bytes[byte_start..]) {
            *slot = *byte;
        }
    }

    let bit_offset = window_start - byte_start * 8;
    let radix = 1usize << window_bits;
    let value = ((u64::from_le_bytes(encoded) >> bit_offset) as usize) & (radix - 1);
    let overlap = if window_start == 0 {
        0
    } else {
        let bit = window_start - 1;
        bytes
            .get(bit / 8)
            .map_or(0, |byte| usize::from((byte >> (bit % 8)) & 1))
    };

    // The bit below each window is its carry-in, while the window's high bit
    // is its carry-out. These terms cancel between adjacent windows, leaving
    // a signed digit whose magnitude is at most half the radix.
    if value < radix / 2 {
        BoothDigit {
            magnitude: value + overlap,
            negative: false,
        }
    } else {
        let magnitude = radix - value - overlap;
        BoothDigit {
            magnitude,
            negative: magnitude != 0,
        }
    }
}

fn booth_window_count<R: AsRef<[u8]>>(reprs: &[R], window_bits: usize) -> usize {
    debug_assert!(window_bits > 0);

    let repr_bytes = reprs
        .iter()
        .map(|repr| repr.as_ref().len())
        .max()
        .unwrap_or(0);
    let repr_bits = repr_bytes
        .checked_mul(u8::BITS as usize)
        .expect("scalar representation is too large");

    // Integer division reserves an additional carry window exactly when the
    // representation ends on a window boundary. Otherwise, the last partial
    // window cannot produce a carry.
    repr_bits / window_bits + 1
}

#[derive(Clone)]
struct BoothBuckets<C: CurveAffine> {
    c: usize,
    coeffs: Vec<Bucket<C>>,
}

impl<C: CurveAffine> BoothBuckets<C> {
    fn new(c: usize) -> Self {
        Self {
            c,
            coeffs: vec![Bucket::None; 1 << (c - 1)],
        }
    }

    fn sum(
        &mut self,
        coeffs: &[<C::ScalarExt as PrimeField>::Repr],
        bases: &[C],
        i: usize,
    ) -> C::CurveExt {
        // Recode each scalar window into a signed digit and add its base to
        // the bucket for the digit's magnitude.
        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
            let digit = booth_digit(coeff.as_ref(), self.c, i);
            if digit.magnitude != 0 {
                let base = if digit.negative { -*base } else { *base };
                self.coeffs[digit.magnitude - 1].add_assign(base);
            }
        }
        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut acc = C::CurveExt::identity();
        let mut sum = C::CurveExt::identity();
        self.coeffs.iter().rev().for_each(|b| {
            sum = b.add(sum);
            acc += sum;
        });
        acc
    }
}


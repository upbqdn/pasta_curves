use super::{Ep, EpAffine, Eq, EqAffine, Fp, Fq};
use crate::arithmetic::{CurveAffine, CurveExt};
use group::{Curve, CurveAffine as _, Group};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::vec::Vec;

mod decompression;
mod normalization;

// Built in a `const` context, so that `from_xy_unchecked` staying `const` is
// pinned by the build rather than by its signature.
const PALLAS_GENERATOR: EpAffine = EpAffine::from_xy_unchecked(
    Fp::neg(&Fp::from_raw([1, 0, 0, 0])),
    Fp::from_raw([2, 0, 0, 0]),
);
const VESTA_GENERATOR: EqAffine = EqAffine::from_xy_unchecked(
    Fq::neg(&Fq::from_raw([1, 0, 0, 0])),
    Fq::from_raw([2, 0, 0, 0]),
);

/// Rebuilds `p` from its coordinates through `unchecked`, and checks that
/// the result is on the curve and equal to what `from_xy` produces.
fn assert_agrees_with_from_xy<C: CurveAffine>(p: C, unchecked: impl Fn(C::Base, C::Base) -> C) {
    let coords = p.coordinates().unwrap();
    let (x, y) = (*coords.x(), *coords.y());
    let q = unchecked(x, y);
    assert!(bool::from(q.is_on_curve()));
    assert!(q == C::from_xy(x, y).unwrap());
    assert!(q == p);
}

fn check_vartime_normalization<C: CurveExt>(points: &[C]) {
    for point in points {
        assert_eq!(point.to_affine_vartime(), point.to_affine());
    }

    let mut vartime = vec![C::Affine::generator(); points.len()];
    let mut constant_time = vec![C::Affine::generator(); points.len()];
    C::batch_normalize_vartime(points, &mut vartime);
    C::batch_normalize(points, &mut constant_time);
    assert_eq!(vartime, constant_time);
    for (point, affine) in points.iter().zip(vartime.iter()) {
        assert_eq!(*affine, point.to_affine());
    }
}

fn check_curve_normalization<C: CurveExt>() {
    let generator = C::generator();
    let double = generator.double();
    let triple = double + generator;
    let identity = C::identity();

    check_vartime_normalization::<C>(&[]);
    check_vartime_normalization(&[identity]);
    check_vartime_normalization(&[generator]);
    check_vartime_normalization(&[identity, identity, identity]);
    check_vartime_normalization(&[
        identity, generator, double, identity, triple, generator, identity,
    ]);
    check_vartime_normalization(&[generator, double, triple]);

    let mut rng = XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);
    for len in [1, 2, 3, 7, 16, 31] {
        let points: Vec<C> = (0..len)
            .map(|i| {
                if i % 3 == 0 {
                    C::identity()
                } else {
                    C::try_random(&mut rng).unwrap().double()
                }
            })
            .collect();
        check_vartime_normalization(&points);
    }
}

#[test]
fn vartime_normalization_matches_group_curve() {
    check_curve_normalization::<Ep>();
    check_curve_normalization::<Eq>();
}

#[test]
fn from_xy_unchecked_generator() {
    assert!(bool::from(PALLAS_GENERATOR.is_on_curve()));
    assert!(bool::from(VESTA_GENERATOR.is_on_curve()));
    assert!(PALLAS_GENERATOR == <EpAffine as group::CurveAffine>::generator());
    assert!(VESTA_GENERATOR == <EqAffine as group::CurveAffine>::generator());
    assert_agrees_with_from_xy(PALLAS_GENERATOR, EpAffine::from_xy_unchecked);
    assert_agrees_with_from_xy(VESTA_GENERATOR, EqAffine::from_xy_unchecked);
}

#[test]
fn from_xy_unchecked_random_points() {
    let mut rng = XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);
    for _ in 0..100 {
        let p = Ep::try_random(&mut rng).unwrap().to_affine();
        assert_agrees_with_from_xy(p, EpAffine::from_xy_unchecked);
        let q = Eq::try_random(&mut rng).unwrap().to_affine();
        assert_agrees_with_from_xy(q, EqAffine::from_xy_unchecked);
    }
}

#[test]
fn from_xy_unchecked_zero_is_identity() {
    let p = EpAffine::from_xy_unchecked(Fp::zero(), Fp::zero());
    assert!(bool::from(p.is_identity()));
    assert!(bool::from(p.is_on_curve()));
    assert!(p == EpAffine::from_xy(Fp::zero(), Fp::zero()).unwrap());
    let q = EqAffine::from_xy_unchecked(Fq::zero(), Fq::zero());
    assert!(bool::from(q.is_identity()));
    assert!(bool::from(q.is_on_curve()));
    assert!(q == EqAffine::from_xy(Fq::zero(), Fq::zero()).unwrap());
}

#[test]
fn from_xy_unchecked_does_not_check() {
    // (1, 1) is not on y^2 = x^3 + 5: the unchecked constructor accepts it,
    // and the checked one rejects it.
    let p = EpAffine::from_xy_unchecked(Fp::one(), Fp::one());
    assert!(!bool::from(p.is_on_curve()));
    assert!(bool::from(
        EpAffine::from_xy(Fp::one(), Fp::one()).is_none()
    ));
}

use group::{CurveAffine, Group};
use zeroize::Zeroize;

use super::{Ep, EpAffine, Eq, EqAffine};

#[test]
fn projective_points_zeroize_to_identity() {
    let mut p = Ep::generator();
    p.zeroize();
    assert!(bool::from(p.is_identity()));

    let mut q = Eq::generator();
    q.zeroize();
    assert!(bool::from(q.is_identity()));
}

#[test]
fn affine_points_zeroize_to_identity() {
    let mut p = EpAffine::generator();
    p.zeroize();
    assert!(bool::from(p.is_identity()));

    let mut q = EqAffine::generator();
    q.zeroize();
    assert!(bool::from(q.is_identity()));
}

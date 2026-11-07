#![cfg(feature = "alloc")]

use ff::Field;
use group::{Curve, Group};
use pasta_curves::{
    arithmetic::{Coordinates, CurveAffine},
    pallas, vesta,
};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

fn check<C: CurveAffine>() {
    let mut rng = XorShiftRng::from_seed([23; 16]);
    for _ in 0..257 {
        let point = C::CurveExt::random(&mut rng).to_affine();
        if let Some(coords) = Option::<Coordinates<C>>::from(point.coordinates()) {
            assert_eq!(
                <C as CurveAffine>::from_xy_unchecked(*coords.x(), *coords.y()),
                point
            );
        }
    }
    assert_eq!(
        <C as CurveAffine>::from_xy_unchecked(C::Base::ZERO, C::Base::ZERO),
        C::identity()
    );
}

#[test]
fn trusted_coordinates_reconstruct_both_curves() {
    check::<pallas::Affine>();
    check::<vesta::Affine>();
}

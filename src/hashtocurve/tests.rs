use super::{iso_map, map_to_curve_simple_swu};
use crate::{
    Ep, Eq, Fp, Fq,
    arithmetic::CurveExt,
    curves::{IsoEp, IsoEq},
};
use ff::Field;
use group::Group;

/// The simplified SWU map's `gx2 = Z^3 u^6 gx1` shortcut is invalid on
/// the exceptional inputs `Z^2 u^4 + Z u^2 = 0` (see the comment in
/// `map_to_curve_simple_swu`), so with the curve-equation check gone
/// from release builds the map must never take the nonsquare branch
/// there. Pin the two facts that guarantee it for each Pasta curve:
/// `g(B / (Z A))` is a square (so `u = 0` takes the square branch), and
/// `-1 / Z` is not a square (so no other exceptional input exists);
/// then check `u = 0` itself lands on `x = B / (Z A)`, on the curve.
macro_rules! exceptional_swu_inputs_are_harmless {
    ($F:ty, $C:ty, $I:ty) => {{
        let a = <$I as CurveExt>::a();
        let b = <$I as CurveExt>::b();
        let z = <$C>::Z;

        let x1 = b * (z * a).invert().unwrap();
        let gx1 = (x1.square() + a) * x1 + b;
        assert!(
            bool::from(gx1.sqrt().is_some()),
            "g(B / (Z A)) must be a square so that u = 0 takes the square branch"
        );
        assert!(
            bool::from((-z.invert().unwrap()).sqrt().is_none()),
            "-1 / Z must be a nonsquare so that no u with Z u^2 = -1 exists"
        );

        let q = map_to_curve_simple_swu::<$F, $C, $I>(&<$F>::ZERO, <$C>::THETA, z);
        assert!(bool::from(q.is_on_curve()));
        assert!(!bool::from(q.is_identity()));
        let (x, _, zz) = q.jacobian_coordinates();
        assert_eq!(
            x * zz.square().invert().unwrap(),
            x1,
            "u = 0 must map to x = B / (Z A)"
        );
        assert!(bool::from(
            iso_map::<_, $C, _>(&q, &<$C>::ISOGENY_CONSTANTS).is_on_curve()
        ));
    }};
}

#[test]
fn exceptional_swu_inputs() {
    exceptional_swu_inputs_are_harmless!(Fp, Ep, IsoEp);
    exceptional_swu_inputs_are_harmless!(Fq, Eq, IsoEq);
}

#[test]
fn unchecked_map_outputs_are_on_curve() {
    for i in 0..256 {
        let q0 = map_to_curve_simple_swu::<Fp, Ep, IsoEp>(&Fp::from(i), Ep::THETA, Ep::Z);
        let q1 = map_to_curve_simple_swu::<Fp, Ep, IsoEp>(&Fp::from(i + 256), Ep::THETA, Ep::Z);
        assert!(bool::from(q0.is_on_curve()));
        assert!(bool::from(q1.is_on_curve()));
        assert!(bool::from(
            iso_map::<_, Ep, _>(&(q0 + q1), &Ep::ISOGENY_CONSTANTS).is_on_curve()
        ));

        let q0 = map_to_curve_simple_swu::<Fq, Eq, IsoEq>(&Fq::from(i), Eq::THETA, Eq::Z);
        let q1 = map_to_curve_simple_swu::<Fq, Eq, IsoEq>(&Fq::from(i + 256), Eq::THETA, Eq::Z);
        assert!(bool::from(q0.is_on_curve()));
        assert!(bool::from(q1.is_on_curve()));
        assert!(bool::from(
            iso_map::<_, Eq, _>(&(q0 + q1), &Eq::ISOGENY_CONSTANTS).is_on_curve()
        ));
    }
}

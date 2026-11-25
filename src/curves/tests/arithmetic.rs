use ff::Field;
use group::{Curve, CurveAffine as _, Group};
use proptest::prelude::*;
use subtle::ConditionallySelectable;

use super::*;

// The previous addition and doubling formulas, including their Choice-based identity checks.
macro_rules! old_arithmetic {
    ($module:ident, $curve:ident, $affine:ident, $base:ident, $scalar:ident) => {
        mod $module {
            use super::*;

            fn double(p: &$curve) -> $curve {
                let a = Field::square(&p.x);
                let b = Field::square(&p.y);
                let c = Field::square(&b);
                let d = p.x + b;
                let d = Field::square(&d);
                let d = d - a - c;
                let d = d + d;
                let e = a + a + a;
                let f = Field::square(&e);
                let z3 = p.z * p.y;
                let z3 = z3 + z3;
                let x3 = f - (d + d);
                let c = c + c;
                let c = c + c;
                let c = c + c;
                let y3 = e * (d - x3) - c;
                let tmp = $curve {
                    x: x3,
                    y: y3,
                    z: z3,
                };
                $curve::conditional_select(&tmp, &$curve::identity(), p.is_identity())
            }

            fn projective_plus_projective(p: &$curve, q: &$curve) -> $curve {
                if bool::from(p.is_identity()) {
                    *q
                } else if bool::from(q.is_identity()) {
                    *p
                } else {
                    let z1z1 = Field::square(&p.z);
                    let z2z2 = Field::square(&q.z);
                    let u1 = p.x * z2z2;
                    let u2 = q.x * z1z1;
                    let s1 = p.y * z2z2 * q.z;
                    let s2 = q.y * z1z1 * p.z;
                    if bool::from(u1.ct_eq(&u2)) {
                        if bool::from(s1.ct_eq(&s2)) {
                            double(p)
                        } else {
                            $curve::identity()
                        }
                    } else {
                        let h = u2 - u1;
                        let i = Field::square(&(h + h));
                        let j = h * i;
                        let r = s2 - s1;
                        let r = r + r;
                        let v = u1 * i;
                        let x3 = Field::square(&r) - j - v - v;
                        let s1 = s1 * j;
                        let s1 = s1 + s1;
                        let y3 = r * (v - x3) - s1;
                        let z3 = Field::square(&(p.z + q.z)) - z1z1 - z2z2;
                        let z3 = z3 * h;
                        $curve {
                            x: x3,
                            y: y3,
                            z: z3,
                        }
                    }
                }
            }

            fn projective_plus_affine(p: &$curve, q: &$affine) -> $curve {
                if bool::from(p.is_identity()) {
                    q.to_curve()
                } else if bool::from(q.is_identity()) {
                    *p
                } else {
                    let z1z1 = Field::square(&p.z);
                    let u2 = q.x * z1z1;
                    let s2 = q.y * z1z1 * p.z;
                    if bool::from(p.x.ct_eq(&u2)) {
                        if bool::from(p.y.ct_eq(&s2)) {
                            double(p)
                        } else {
                            $curve::identity()
                        }
                    } else {
                        let h = u2 - p.x;
                        let hh = Field::square(&h);
                        let i = hh + hh;
                        let i = i + i;
                        let j = h * i;
                        let r = s2 - p.y;
                        let r = r + r;
                        let v = p.x * i;
                        let x3 = Field::square(&r) - j - v - v;
                        let j = p.y * j;
                        let j = j + j;
                        let y3 = r * (v - x3) - j;
                        let z3 = Field::square(&(p.z + h)) - z1z1 - hh;
                        $curve {
                            x: x3,
                            y: y3,
                            z: z3,
                        }
                    }
                }
            }

            fn affine_plus_affine(p: &$affine, q: &$affine) -> $curve {
                if bool::from(p.is_identity()) {
                    q.to_curve()
                } else if bool::from(q.is_identity()) {
                    p.to_curve()
                } else if bool::from(p.x.ct_eq(&q.x)) {
                    if bool::from(p.y.ct_eq(&q.y)) {
                        double(&p.to_curve())
                    } else {
                        $curve::identity()
                    }
                } else {
                    let h = q.x - p.x;
                    let hh = Field::square(&h);
                    let i = hh + hh;
                    let i = i + i;
                    let j = h * i;
                    let r = q.y - p.y;
                    let r = r + r;
                    let v = p.x * i;
                    let x3 = Field::square(&r) - j - v - v;
                    let j = p.y * j;
                    let j = j + j;
                    let y3 = r * (v - x3) - j;
                    let z3 = h + h;
                    $curve {
                        x: x3,
                        y: y3,
                        z: z3,
                    }
                }
            }

            fn same(got: $curve, old: $curve) {
                assert_eq!((got.x, got.y, got.z), (old.x, old.y, old.z));
            }

            fn compare(p: $curve, q: $curve, a: $affine, b: $affine) {
                same(p + q, projective_plus_projective(&p, &q));
                same(p + b, projective_plus_affine(&p, &b));
                same(a + b, affine_plus_affine(&a, &b));
                same(a + q, projective_plus_affine(&q, &a));
                same(p.double(), double(&p));
                same(p - q, projective_plus_projective(&p, &-q));
                same(p - b, projective_plus_affine(&p, &-b));
                same(a - b, affine_plus_affine(&a, &-b));
                same(a - q, projective_plus_affine(&-q, &a));
                let mut assigned = p;
                assigned += q;
                same(assigned, projective_plus_projective(&p, &q));
                let mut assigned = p;
                assigned += b;
                same(assigned, projective_plus_affine(&p, &b));
                same(a.to_curve_vartime(), a.to_curve());
                same(b.to_curve_vartime(), b.to_curve());
            }

            proptest! {
                #[test]
                fn exact_coordinates(a in any::<u64>(), b in any::<u64>()) {
                    let generator = $curve::generator();
                    let p = generator * $scalar::from(a);
                    let q = generator * $scalar::from(b);
                    let pa = p.to_affine();
                    let qa = q.to_affine();
                    let identity = $curve::identity();
                    let affine_identity = $affine::identity();
                    compare(p, q, pa, qa);
                    compare(p, p, pa, pa);
                    compare(p, -p, pa, -pa);
                    compare(identity, q, affine_identity, qa);
                    compare(p, identity, pa, affine_identity);
                    compare(identity, identity, affine_identity, affine_identity);

                    // The identity is any projective z=0, even when x and y are nonzero.
                    let noncanonical_identity = $curve {
                        x: $base::from(a),
                        y: $base::from(b),
                        z: $base::ZERO,
                    };
                    compare(noncanonical_identity, q, affine_identity, qa);
                    // Unchecked affine points can be off-curve; retain their exact coordinates.
                    let unchecked = $affine::from_xy_unchecked($base::from(a), $base::from(b));
                    compare(p, q, unchecked, qa);
                    // Half-zero points catch an identity test that checks only one coordinate.
                    for half_zero in [
                        $affine::from_xy_unchecked($base::ZERO, $base::from(b) + $base::ONE),
                        $affine::from_xy_unchecked($base::from(a) + $base::ONE, $base::ZERO),
                    ] {
                        compare(p, q, half_zero, qa);
                        compare(p, q, pa, half_zero);
                    }
                    let unchecked_projective = $curve {
                        x: $base::from(a),
                        y: $base::from(b),
                        z: $base::ONE,
                    };
                    compare(unchecked_projective, q, unchecked, qa);
                }
            }
        }
    };
}

use subtle::ConstantTimeEq;
old_arithmetic!(pallas, Ep, EpAffine, Fp, Fq);
old_arithmetic!(vesta, Eq, EqAffine, Fq, Fp);

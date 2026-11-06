use super::*;

/// The known answers above that are not the field types' own constants, recomputed on the same
/// limbs by the field types' inherent `const fn`s, which never use the backend.
#[test]
fn known_answers_match_the_portable_arithmetic() {
    macro_rules! check {
        ($field:ident, $f:expr) => {{
            use crate::fields::$field;
            let f = $f;
            let portable_add = |x: Limbs, y: Limbs| $field::add(&$field(x), &$field(y)).0;
            let portable_mul = |x: Limbs, y: Limbs| $field::mul(&$field(x), &$field(y)).0;
            let portable_square = |x: Limbs| $field::square(&$field(x)).0;
            let pm1 = p_minus_1(f);
            assert_eq!(portable_add(f.r, f.r), f.two_r);
            assert_eq!(portable_add(f.two_r, f.r), f.three_r);
            assert_eq!(portable_add(pm1, pm1), f.pm2);
            assert_eq!(portable_mul(f.r2, f.r3), f.r4);
            assert_eq!(
                portable_mul(portable_square(portable_square(f.r2)), f.r3),
                f.r7
            );
            assert_eq!(portable_mul(pm1, pm1), f.pm1_sq);
            // `from_mont` of the all-ones input: its Montgomery product with the residue `1`.
            assert_eq!(portable_mul([u64::MAX; 4], ONE), f.from_mont_ones);
            // `e0` is the integer `2^562 mod p`, the canonical integer of `2R` to the 562nd power.
            let pow = |base: Limbs, exponent: &Limbs| {
                portable_pow(base, exponent, f.r, portable_square, portable_mul)
            };
            assert_eq!(portable_mul(pow(f.two_r, &[562, 0, 0, 0]), ONE), f.e0);
            // Each inversion's output is the Montgomery form of the inverse of the element whose
            // Montgomery form is its input: the input to the power `p - 2`.
            for (x, z) in &f.inversions {
                assert_eq!((f.portable_inverse)(x), *z, "{x:x?}");
            }
        }};
    }
    check!(Fp, &FP);
    check!(Fq, &FQ);
}

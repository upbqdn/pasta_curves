use super::*;

#[test]
fn low_half_square_matches_classical_and_multiplication() {
    use rand::SeedableRng;

    const ONE_SQUARED: Fp = Fp::one().square();
    assert_eq!(ONE_SQUARED, Fp::one());
    let classical =
        |t: [u64; 8]| Fp::montgomery_reduce(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    let low_half = |t: [u64; 8]| {
        Fp(portable::canonicalize(
            &portable::montgomery_reduce_low_lazy(&t, &MODULUS.0, INV),
            &MODULUS.0,
        ))
    };
    let check_square = |a: Fp| {
        let wide = a.square_unreduced();
        assert_eq!(wide, a.mul_unreduced(&a));
        assert_eq!(a.square(), classical(wide));
        assert_eq!(a.square(), low_half(wide));
        assert_eq!(a.square(), a * a);
        assert_eq!(a.square(), Field::square(&a));
    };
    // Field boundaries and canonical Montgomery limbs just below the modulus.
    for a in [
        Fp::zero(),
        Fp::one(),
        -Fp::one(),
        Fp::from(2),
        Fp::from(3),
        Fp::from_raw([u64::MAX; 4]),
        Fp([MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]]),
        Fp([u64::MAX, u64::MAX, u64::MAX, MODULUS.0[3] - 1]),
    ] {
        check_square(a);
    }
    for bit in 0..255 {
        let mut limbs = [0; 4];
        limbs[bit / 64] = 1 << (bit % 64);
        check_square(Fp(limbs));
    }

    // Exercise the reducer's full domain, not just square products:
    // arbitrary low limbs and a high half below the modulus.
    for lo in [[0; 4], [u64::MAX; 4]] {
        for hi in [[0; 4], [MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]]] {
            let t = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
            assert_eq!(low_half(t), classical(t));
        }
    }
    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x3c; 16]);
    for _ in 0..20_000 {
        let a = Fp::random(&mut rng);
        check_square(a);
        let t = [
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            a.0[0],
            a.0[1],
            a.0[2],
            a.0[3],
        ];
        assert_eq!(low_half(t), classical(t));
    }
}

impl Fp {
    // Preserve the pre-port portable operations as independent eager oracles.
    fn sqr_n_eager(&self, n: u32) -> Self {
        (0..n).fold(*self, |acc, _| acc.square_portable())
    }

    fn sqr_n_mul_eager(&self, n: u32, by: &Self) -> Self {
        self.sqr_n_eager(n).mul(by)
    }

    #[cfg(feature = "sqrt-table")]
    fn pow_by_t_minus1_over2_eager(&self) -> Self {
        let r10 = self.square_runtime();
        let r11 = r10 * self;
        let r110 = r11.square_runtime();
        let r111 = r110 * self;
        let r1001 = r111 * r10;
        let r1101 = r111 * r110;
        let ra = self.sqr_n_mul_eager(129, self);
        let rb = ra.sqr_n_mul_eager(7, &r1001);
        let rc = rb.sqr_n_mul_eager(7, &r1101);
        let rd = rc.sqr_n_mul_eager(4, &r11);
        let re = rd.sqr_n_mul_eager(6, &r111);
        let rf = re.sqr_n_mul_eager(3, &r111);
        let rg = rf.sqr_n_mul_eager(10, &r1001);
        let rh = rg.sqr_n_mul_eager(5, &r1001);
        let ri = rh.sqr_n_mul_eager(4, &r1001);
        let rj = ri.sqr_n_mul_eager(3, &r111);
        let rk = rj.sqr_n_mul_eager(4, &r1001);
        let rl = rk.sqr_n_mul_eager(5, &r11);
        let rm = rl.sqr_n_mul_eager(4, &r111);
        let rn = rm.sqr_n_mul_eager(4, &r11);
        let ro = rn.sqr_n_mul_eager(6, &r1001);
        let rp = ro.sqr_n_mul_eager(5, &r1101);
        let rq = rp.sqr_n_mul_eager(4, &r11);
        let rr = rq.sqr_n_mul_eager(7, &r111);
        let rs = rr.sqr_n_mul_eager(3, &r11);
        rs.square_runtime()
    }

    fn sqrt_eager(&self) -> CtOption<Self> {
        #[cfg(feature = "sqrt-table")]
        {
            let (is_square, res) =
                FP_TABLES.sqrt_alt_eager(self, &self.pow_by_t_minus1_over2_eager());
            CtOption::new(res, is_square)
        }
        #[cfg(not(feature = "sqrt-table"))]
        ff::helpers::sqrt_tonelli_shanks(self, T_MINUS1_OVER2)
    }
}

#[test]
fn lazy_square_and_sqrt_match_eager() {
    use rand::SeedableRng;

    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x46; 16]);
    let max = [MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]];
    let boundaries = [
        Fp::zero(),
        Fp::one(),
        -Fp::one(),
        Fp::from_raw(max), // Integer p - 1, independently of negation.
        Fp(max),           // Largest canonical Montgomery residue.
        Fp::ROOT_OF_UNITY,
        Fp::from(5),
    ];

    for a in boundaries
        .into_iter()
        .chain((0..4_096).map(|_| Fp::random(&mut rng)))
    {
        assert_eq!(a.sqr_n_runtime(1).0, a.square_portable().0);
        assert_eq!(a.sqr_n_runtime(1).0, Field::square(&a).0);
        for n in [0, 1, 2, 3, 4, 7, 8, 11, 32, 129, 256, 512] {
            let eager = a.sqr_n_eager(n);
            assert_eq!(a.sqr_n_runtime(n).0, eager.0, "sqr_n, n = {n}");
            assert_eq!(crate::arithmetic::square_fp_n(&a, n).0, eager.0);
            for by in [Fp::zero(), Fp::one(), -Fp::one(), a] {
                let expected = a.sqr_n_mul_eager(n, &by);
                assert_eq!(a.sqr_n_mul_runtime(n, &by).0, expected.0);
                // Also exercise the new portable path when the assembly feature is enabled.
                let lazy = Fp(portable::sqr_n_lazy(&a.0, n, &MODULUS.0, INV));
                assert_eq!(lazy.mul_portable(&by).0, expected.0);
            }
        }

        #[cfg(feature = "sqrt-table")]
        {
            assert_eq!(
                a.pow_by_t_minus1_over2().0,
                a.pow_by_t_minus1_over2_eager().0
            );
            let (choice, root) = FP_TABLES.sqrt_alt(&a);
            let (old_choice, old_root) =
                FP_TABLES.sqrt_alt_eager(&a, &a.pow_by_t_minus1_over2_eager());
            assert_eq!(choice.unwrap_u8(), old_choice.unwrap_u8());
            assert_eq!(root.0, old_root.0);
        }
        let root = a.sqrt();
        let old_root = a.sqrt_eager();
        assert_eq!(root.is_some().unwrap_u8(), old_root.is_some().unwrap_u8());
        assert_eq!(
            root.unwrap_or(Fp::zero()).0,
            old_root.unwrap_or(Fp::zero()).0
        );
    }

    for nonsquare in [Fp::ROOT_OF_UNITY, Fp::from(5)] {
        assert!(bool::from(nonsquare.sqrt().is_none()));
        assert!(bool::from(nonsquare.sqrt_eager().is_none()));
    }
}

#[test]
fn lazy_square_accumulator_stays_below_two_moduli() {
    use rand::SeedableRng;

    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x9d; 16]);
    let (m0, carry) = adc(MODULUS.0[0], MODULUS.0[0], 0);
    let (m1, carry) = adc(MODULUS.0[1], MODULUS.0[1], carry);
    let (m2, carry) = adc(MODULUS.0[2], MODULUS.0[2], carry);
    let (m3, carry) = adc(MODULUS.0[3], MODULUS.0[3], carry);
    assert_eq!(carry, 0);
    let two_moduli = [m0, m1, m2, m3];
    let max = Fp([MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]]);
    for a in [Fp::zero(), Fp::one(), -Fp::one(), max]
        .into_iter()
        .chain((0..64).map(|_| Fp::random(&mut rng)))
    {
        let mut lazy = a.0;
        let mut eager = a;
        for _ in 0..2_048 {
            lazy = portable::sqr_n_lazy(&lazy, 1, &MODULUS.0, INV);
            eager = eager.square_portable();
            assert!(lazy.iter().rev().cmp(two_moduli.iter().rev()).is_lt());
            assert_eq!(portable::canonicalize(&lazy, &MODULUS.0), eager.0);
        }
    }
}

// The tests in these blocks compare the field types' runtime path (the operators, `Field::square`,
// `sqr_n_mul_runtime`, and `to_repr`) with their portable `const fn`s. They are compiled only where
// that path is the assembly backend: elsewhere it is the portable code, and each comparison would
// be of the code with itself.
#[cfg(all(test, feature = "asm"))]
if_asm_supported! {
fn asm_portable_repr(value: Fp) -> [u8; 32] {
    let value = Fp::montgomery_reduce(value.0[0], value.0[1], value.0[2], value.0[3], 0, 0, 0, 0);
    let mut repr = [0; 32];
    for (bytes, limb) in repr.chunks_exact_mut(8).zip(value.0) {
        bytes.copy_from_slice(&limb.to_le_bytes());
    }
    repr
}

fn asm_check_repr(value: Fp) {
    let portable = asm_portable_repr(value);
    assert_eq!(value.to_repr(), portable);
    assert_eq!(Fp::from_repr(portable).unwrap(), value);
    assert_eq!(value.is_odd().unwrap_u8(), portable[0] & 1);
}

fn asm_portable_cmp(lhs: Fp, rhs: Fp) -> core::cmp::Ordering {
    asm_portable_repr(lhs)
        .iter()
        .zip(asm_portable_repr(rhs).iter())
        .rev()
        .find_map(|(lhs, rhs)| match lhs.cmp(rhs) {
            core::cmp::Ordering::Equal => None,
            ordering => Some(ordering),
        })
        .unwrap_or(core::cmp::Ordering::Equal)
}

#[test]
fn asm_matches_portable_arithmetic() {
    use rand::SeedableRng;

    let max_montgomery_residue = Fp([MODULUS.0[0] - 1, MODULUS.0[1], MODULUS.0[2], MODULUS.0[3]]);
    let boundaries = [
        Fp::zero(),
        Fp::one(),
        -Fp::one(),
        Fp::from_raw([1, 0, 0, 0]),
        max_montgomery_residue,
        Fp::from_raw([u64::MAX; 4]),
    ];

    fn portable_sqr_n_mul(value: Fp, n: u32, by: Fp) -> Fp {
        (0..n).fold(value, |acc, _| Fp::square(&acc)).mul(&by)
    }

    for lhs in boundaries {
        asm_check_repr(lhs);
        assert_eq!(<Fp as Field>::square(&lhs), Fp::square(&lhs));
        for rhs in boundaries {
            assert_eq!(lhs.cmp(&rhs), asm_portable_cmp(lhs, rhs));
            assert_eq!(&lhs * &rhs, Fp::mul(&lhs, &rhs));
            for n in [1, 2, 7] {
                assert_eq!(
                    lhs.sqr_n_mul_runtime(n, &rhs),
                    portable_sqr_n_mul(lhs, n, rhs)
                );
            }
        }
    }

    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x5a; 16]);
    for _ in 0..1024 {
        let lhs = Fp::from_raw([
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
        ]);
        let rhs = Fp::from_raw([
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
        ]);

        asm_check_repr(lhs);
        assert_eq!(lhs.cmp(&rhs), asm_portable_cmp(lhs, rhs));
        assert_eq!(&lhs * &rhs, Fp::mul(&lhs, &rhs));
        assert_eq!(<Fp as Field>::square(&lhs), Fp::square(&lhs));
        for n in [1, 129] {
            assert_eq!(
                lhs.sqr_n_mul_runtime(n, &rhs),
                portable_sqr_n_mul(lhs, n, rhs)
            );
        }
    }
}
}

#[cfg(feature = "sqrt-table")]
#[test]
fn fp_sqrt_table_hash_is_perfect() {
    FP_TABLES.assert_hash_table_is_consistent();
}

#[cfg(feature = "sqrt-table")]
#[test]
fn fp_sqrt_table_matches_tonelli_shanks() {
    use rand::SeedableRng;

    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x46; 16]);
    for _ in 0..1_000 {
        let input = Fp::random(&mut rng);
        let table = input.sqrt();
        let reference = ff::helpers::sqrt_tonelli_shanks(&input, T_MINUS1_OVER2);

        assert_eq!(table.is_some().unwrap_u8(), reference.is_some().unwrap_u8());
        if bool::from(table.is_some()) {
            let table = table.unwrap();
            let reference = reference.unwrap();
            assert!(table == reference || table == -reference);
        }
    }
}

#[test]
fn test_pow_vartime() {
    use rand::SeedableRng;

    // The classic square-and-multiply loop, as a reference for the fused
    // implementation.
    fn pow_vartime_reference(base: &Fp, exp: &[u64]) -> Fp {
        let mut res = Fp::one();
        let mut found_one = false;
        for e in exp.iter().rev() {
            for i in (0..64).rev() {
                if found_one {
                    res = Fp::square(&res);
                }

                if ((*e >> i) & 1) == 1 {
                    found_one = true;
                    res = Fp::mul(&res, base);
                }
            }
        }
        res
    }

    let mut rng = rand_xorshift::XorShiftRng::from_seed([0xa5; 16]);

    let mut exponents = vec![
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        // A single high bit exercises the trailing-squarings flush.
        [0, 0, 0, 1 << 63],
        [1 << 63, 0, 0, 0],
        [u64::MAX; 4],
        // The p - 2 exponent used by `invert`.
        [
            0x992d30ecffffffff,
            0x224698fc094cf91b,
            0x0,
            0x4000000000000000,
        ],
    ];
    for _ in 0..10 {
        exponents.push([
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
        ]);
    }

    for base in [
        Fp::zero(),
        Fp::one(),
        -Fp::one(),
        Fp::random(&mut rng),
        Fp::random(&mut rng),
    ] {
        for exp in &exponents {
            assert_eq!(base.pow_vartime(exp), pow_vartime_reference(&base, exp));
        }
        // Short and empty exponent slices behave like zero-padded ones.
        assert_eq!(base.pow_vartime([7]), pow_vartime_reference(&base, &[7]));
        assert_eq!(
            base.pow_vartime([0u64; 0]),
            pow_vartime_reference(&base, &[])
        );
    }
}

/// Corner cases of `from_u512`. Each input is the 512-bit integer `d0 + 2^256 d1`, where `d0` and
/// `d1` are the 256-bit digits that `from_u512` splits it into. The expected residues mod `p` were
/// computed with Python's integers. In the last three cases the digits are the left operands that
/// come nearest to overflowing the inline `mul` when the right operand is `R2` or `R3`, the
/// constants that `from_u512` multiplies its digits by.
#[test]
fn test_from_u512_corner_cases() {
    #[rustfmt::skip]
    let cases: [([u64; 8], [u64; 4]); 12] = [
        // 0
        (
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
             0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ),
        // p - 1
        (
            [0x992d30ed00000000, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000,
             0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x992d30ed00000000, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000],
        ),
        // p
        (
            [0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000,
             0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ),
        // 2^256 - 1
        (
            [0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
             0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x34786d38fffffffc, 0x992c350be41914ad, 0xffffffffffffffff, 0x3fffffffffffffff],
        ),
        // 2^256
        (
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
             0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x34786d38fffffffd, 0x992c350be41914ad, 0xffffffffffffffff, 0x3fffffffffffffff],
        ),
        // p 2^256
        (
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
             0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ),
        // (p - 1) 2^256 + p - 1
        (
            [0x992d30ed00000000, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000,
             0x992d30ed00000000, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000],
            [0x64b4c3b400000003, 0x891a63f02533e46e, 0x0000000000000000, 0x0000000000000000],
        ),
        // 2^511
        (
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
             0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000],
            [0x92d30ed000000008, 0x7d0cd35cca2d6d01, 0x3bcbd4cde1e4ae8c, 0x24b6a0d7bdce5b8a],
        ),
        // 2^512 - 1
        (
            [0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
             0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff],
            [0x8c78ecb30000000e, 0xd7d30dbd8b0de0e7, 0x7797a99bc3c95d18, 0x096d41af7b9cb714],
        ),
        // forced_q_r2, the left operand nearest to overflowing `mul` with `R2`
        (
            [0x3cc9961eeeeeeeef, 0x907f42c685cc8a31, 0xffffffffffffffff, 0xffffffffffffffff,
             0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x71420357eeeeeeec, 0x29ab77d269e59ede, 0xffffffffffffffff, 0x3fffffffffffffff],
        ),
        // forced_q_r3 2^256, the same for `R3`
        (
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
             0x032c286da5f9b149, 0x3f747fab2d936552, 0xffffffffffffffff, 0xffffffffffffffff],
            [0xd5abd7bb68193aea, 0x4b22a76b19f2f1ce, 0xa87dc2a20b1a34e5, 0x308bc2b27a3a82ee],
        ),
        // forced_q_r2 + forced_q_r3 2^256
        (
            [0x3cc9961eeeeeeeef, 0x907f42c685cc8a31, 0xffffffffffffffff, 0xffffffffffffffff,
             0x032c286da5f9b149, 0x3f747fab2d936552, 0xffffffffffffffff, 0xffffffffffffffff],
            [0xadc0aa26570829d5, 0x528786417a8b9791, 0xa87dc2a20b1a34e4, 0x308bc2b27a3a82ee],
        ),
    ];
    for (input, residue) in cases {
        let repr = ff::PrimeField::to_repr(&Fp::from_u512(input));
        let limbs: [u64; 4] = core::array::from_fn(|i| {
            u64::from_le_bytes(repr[8 * i..8 * (i + 1)].try_into().unwrap())
        });
        assert_eq!(limbs, residue, "{input:x?}");
    }
}

#[test]
fn test_from_u512_against_horner() {
    use rand::SeedableRng;

    // `from_u512` splits the 512-bit integer into two 256-bit digits and recombines them with
    // `R2` and `R3`. Horner's rule computes the same value one limb at a time instead.
    let horner = |l: [u64; 8]| {
        let two_64 = Fp::from(u64::MAX) + Fp::one();
        l.iter()
            .rev()
            .fold(Fp::zero(), |acc, &limb| acc * two_64 + Fp::from(limb))
    };
    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x3c; 16]);
    let mut inputs = vec![[0u64; 8], [u64::MAX; 8]];
    for _ in 0..1000 {
        inputs.push(core::array::from_fn(|_| rng.try_next_u64().unwrap()));
    }
    for l in inputs {
        assert_eq!(Fp::from_u512(l), horner(l), "limbs {l:x?}");
    }
}

#[cfg(feature = "zeroize")]
#[test]
fn test_zeroize() {
    use zeroize::Zeroize;

    let mut a = <Fp as ff::Field>::ONE;
    a.zeroize();
    assert_eq!(a, Fp::zero());
}

/// Whether `x` holds a reduced residue (a value below the modulus).
#[cfg(test)]
fn is_canonical(x: &Fp) -> bool {
    for i in (0..4).rev() {
        if x.0[i] != MODULUS.0[i] {
            return x.0[i] < MODULUS.0[i];
        }
    }
    false
}

#[test]
fn constants_are_canonical() {
    // Every named constant must be a reduced residue: the `asm`
    // multiplication requires a canonical rhs, and constants are the one
    // class of values that bypass the reducing constructors.
    assert!(
        !is_canonical(&MODULUS),
        "the modulus itself is not canonical"
    );
    let constants: [(&str, Fp); 11] = [
        ("R", R),
        ("R2", R2),
        ("R3", R3),
        ("GENERATOR", GENERATOR),
        ("ROOT_OF_UNITY", ROOT_OF_UNITY),
        ("DELTA", DELTA),
        ("TWO_INV", <Fp as ff::PrimeField>::TWO_INV),
        (
            "MULTIPLICATIVE_GENERATOR",
            <Fp as ff::PrimeField>::MULTIPLICATIVE_GENERATOR,
        ),
        (
            "ROOT_OF_UNITY_INV",
            <Fp as ff::PrimeField>::ROOT_OF_UNITY_INV,
        ),
        ("ZETA", <Fp as ff::WithSmallOrderMulGroup<3>>::ZETA),
        ("zero", Fp::zero()),
    ];
    for (name, value) in constants {
        assert!(
            is_canonical(&value),
            "{name} is not canonical: {:x?}",
            value.0
        );
    }
}

// Compiled only with the backend, like `asm_matches_portable_arithmetic`, for the same reason.
#[cfg(all(test, feature = "asm"))]
if_asm_supported! {
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "requires a canonical lhs")]
fn asm_mul_rejects_non_canonical_lhs_in_debug() {
    // The modulus itself is the smallest non-canonical value. This causes the asm mul
    // contract to constrain the rhs limbs, and limb 2 of `R` (1 in Montgomery form) is
    // `2^64 - 1`, violating the constraint.
    // Call the entry directly: portable fallback has no assembly contract to assert.
    let _ = crate::asm::mul(&MODULUS.0, &R.0, &MODULUS.0, INV);
}
}

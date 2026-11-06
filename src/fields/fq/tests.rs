use super::*;

#[test]
fn low_half_square_matches_classical_and_multiplication() {
    use rand::SeedableRng;

    const ONE_SQUARED: Fq = Fq::one().square();
    assert_eq!(ONE_SQUARED, Fq::one());
    let classical =
        |t: [u64; 8]| Fq::montgomery_reduce(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    let low_half = |t: [u64; 8]| {
        Fq(portable::canonicalize(
            &portable::montgomery_reduce_low_lazy(&t, &MODULUS.0, INV),
            &MODULUS.0,
        ))
    };
    let check_square = |a: Fq| {
        let wide = a.square_unreduced();
        assert_eq!(wide, a.mul_unreduced(&a));
        assert_eq!(a.square(), classical(wide));
        assert_eq!(a.square(), low_half(wide));
        assert_eq!(a.square(), a * a);
        assert_eq!(a.square(), Field::square(&a));
    };
    // Field boundaries and canonical Montgomery limbs just below the modulus.
    for a in [
        Fq::zero(),
        Fq::one(),
        -Fq::one(),
        Fq::from(2),
        Fq::from(3),
        Fq::from_raw([u64::MAX; 4]),
        Fq([MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]]),
        Fq([u64::MAX, u64::MAX, u64::MAX, MODULUS.0[3] - 1]),
    ] {
        check_square(a);
    }
    for bit in 0..255 {
        let mut limbs = [0; 4];
        limbs[bit / 64] = 1 << (bit % 64);
        check_square(Fq(limbs));
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
        let a = Fq::random(&mut rng);
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

impl Fq {
    // Preserve the pre-port portable operations as independent eager oracles.
    fn sqr_n_eager(&self, n: u32) -> Self {
        (0..n).fold(*self, |acc, _| acc.square_portable())
    }

    fn sqr_n_mul_eager(&self, n: u32, by: &Self) -> Self {
        self.sqr_n_eager(n).mul(by)
    }

    #[cfg(feature = "sqrt-table")]
    fn pow_by_t_minus1_over2_eager(&self) -> Self {
        let s10 = self.square_runtime();
        let s11 = s10 * self;
        let s111 = s11.sqr_n_mul_eager(1, self);
        let s1001 = s111 * s10;
        let s1011 = s1001 * s10;
        let s1101 = s1011 * s10;
        let sa = self.sqr_n_mul_eager(129, self);
        let sb = sa.sqr_n_mul_eager(7, &s1001);
        let sc = sb.sqr_n_mul_eager(7, &s1101);
        let sd = sc.sqr_n_mul_eager(4, &s11);
        let se = sd.sqr_n_mul_eager(6, &s111);
        let sf = se.sqr_n_mul_eager(3, &s111);
        let sg = sf.sqr_n_mul_eager(10, &s1001);
        let sh = sg.sqr_n_mul_eager(4, &s1001);
        let si = sh.sqr_n_mul_eager(5, &s1001);
        let sj = si.sqr_n_mul_eager(5, &s1001);
        let sk = sj.sqr_n_mul_eager(3, &s1001);
        let sl = sk.sqr_n_mul_eager(4, &s1011);
        let sm = sl.sqr_n_mul_eager(4, &s1011);
        let sn = sm.sqr_n_mul_eager(5, &s11);
        let so = sn.sqr_n_mul_eager(4, self);
        let sp = so.sqr_n_mul_eager(5, &s11);
        let sq = sp.sqr_n_mul_eager(4, &s111);
        let sr = sq.sqr_n_mul_eager(5, &s1011);
        let ss = sr.sqr_n_mul_eager(3, self);
        (0..4).fold(ss, |x, _| x.square_runtime())
    }

    fn sqrt_eager(&self) -> CtOption<Self> {
        #[cfg(feature = "sqrt-table")]
        {
            let (is_square, res) =
                FQ_TABLES.sqrt_alt_eager(self, &self.pow_by_t_minus1_over2_eager());
            CtOption::new(res, is_square)
        }
        #[cfg(not(feature = "sqrt-table"))]
        ff::helpers::sqrt_tonelli_shanks(self, T_MINUS1_OVER2)
    }
}

#[test]
fn lazy_square_and_sqrt_match_eager() {
    use rand::SeedableRng;

    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x71; 16]);
    let max = [MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]];
    let boundaries = [
        Fq::zero(),
        Fq::one(),
        -Fq::one(),
        Fq::from_raw(max), // Integer q - 1, independently of negation.
        Fq(max),           // Largest canonical Montgomery residue.
        Fq::ROOT_OF_UNITY,
        Fq::from(5),
    ];

    for a in boundaries
        .into_iter()
        .chain((0..4_096).map(|_| Fq::random(&mut rng)))
    {
        assert_eq!(a.sqr_n_runtime(1).0, a.square_portable().0);
        assert_eq!(a.sqr_n_runtime(1).0, Field::square(&a).0);
        for n in [0, 1, 2, 3, 4, 7, 8, 11, 32, 129, 256, 512] {
            let eager = a.sqr_n_eager(n);
            assert_eq!(a.sqr_n_runtime(n).0, eager.0, "sqr_n, n = {n}");
            assert_eq!(crate::arithmetic::square_fq_n(&a, n).0, eager.0);
            for by in [Fq::zero(), Fq::one(), -Fq::one(), a] {
                let expected = a.sqr_n_mul_eager(n, &by);
                assert_eq!(a.sqr_n_mul_runtime(n, &by).0, expected.0);
                // Also exercise the new portable path when the assembly feature is enabled.
                let lazy = Fq(portable::sqr_n_lazy(&a.0, n, &MODULUS.0, INV));
                assert_eq!(lazy.mul_portable(&by).0, expected.0);
            }
        }

        #[cfg(feature = "sqrt-table")]
        {
            assert_eq!(
                a.pow_by_t_minus1_over2().0,
                a.pow_by_t_minus1_over2_eager().0
            );
            let (choice, root) = FQ_TABLES.sqrt_alt(&a);
            let (old_choice, old_root) =
                FQ_TABLES.sqrt_alt_eager(&a, &a.pow_by_t_minus1_over2_eager());
            assert_eq!(choice.unwrap_u8(), old_choice.unwrap_u8());
            assert_eq!(root.0, old_root.0);
        }
        let root = a.sqrt();
        let old_root = a.sqrt_eager();
        assert_eq!(root.is_some().unwrap_u8(), old_root.is_some().unwrap_u8());
        assert_eq!(
            root.unwrap_or(Fq::zero()).0,
            old_root.unwrap_or(Fq::zero()).0
        );
    }

    for nonsquare in [Fq::ROOT_OF_UNITY, Fq::from(5)] {
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
    let max = Fq([MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]]);
    for a in [Fq::zero(), Fq::one(), -Fq::one(), max]
        .into_iter()
        .chain((0..64).map(|_| Fq::random(&mut rng)))
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
fn asm_portable_repr(value: Fq) -> [u8; 32] {
    let value = Fq::montgomery_reduce(value.0[0], value.0[1], value.0[2], value.0[3], 0, 0, 0, 0);
    let mut repr = [0; 32];
    for (bytes, limb) in repr.chunks_exact_mut(8).zip(value.0) {
        bytes.copy_from_slice(&limb.to_le_bytes());
    }
    repr
}

fn asm_check_repr(value: Fq) {
    let portable = asm_portable_repr(value);
    assert_eq!(value.to_repr(), portable);
    assert_eq!(Fq::from_repr(portable).unwrap(), value);
    assert_eq!(value.is_odd().unwrap_u8(), portable[0] & 1);
}

fn asm_portable_cmp(lhs: Fq, rhs: Fq) -> core::cmp::Ordering {
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

    let max_montgomery_residue = Fq([MODULUS.0[0] - 1, MODULUS.0[1], MODULUS.0[2], MODULUS.0[3]]);
    let boundaries = [
        Fq::zero(),
        Fq::one(),
        -Fq::one(),
        Fq::from_raw([1, 0, 0, 0]),
        max_montgomery_residue,
        Fq::from_raw([u64::MAX; 4]),
    ];

    fn portable_sqr_n_mul(value: Fq, n: u32, by: Fq) -> Fq {
        (0..n).fold(value, |acc, _| Fq::square(&acc)).mul(&by)
    }

    for lhs in boundaries {
        asm_check_repr(lhs);
        assert_eq!(<Fq as Field>::square(&lhs), Fq::square(&lhs));
        for rhs in boundaries {
            assert_eq!(lhs.cmp(&rhs), asm_portable_cmp(lhs, rhs));
            assert_eq!(&lhs * &rhs, Fq::mul(&lhs, &rhs));
            for n in [1, 2, 7] {
                assert_eq!(
                    lhs.sqr_n_mul_runtime(n, &rhs),
                    portable_sqr_n_mul(lhs, n, rhs)
                );
            }
        }
    }

    let mut rng = rand_xorshift::XorShiftRng::from_seed([0xa5; 16]);
    for _ in 0..1024 {
        let lhs = Fq::from_raw([
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
        ]);
        let rhs = Fq::from_raw([
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
        ]);

        asm_check_repr(lhs);
        assert_eq!(lhs.cmp(&rhs), asm_portable_cmp(lhs, rhs));
        assert_eq!(&lhs * &rhs, Fq::mul(&lhs, &rhs));
        assert_eq!(<Fq as Field>::square(&lhs), Fq::square(&lhs));
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
fn fq_sqrt_table_hash_is_perfect() {
    FQ_TABLES.assert_hash_table_is_consistent();
}

#[cfg(feature = "sqrt-table")]
#[test]
fn fq_sqrt_table_matches_tonelli_shanks() {
    use rand::SeedableRng;

    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x71; 16]);
    for _ in 0..1_000 {
        let input = Fq::random(&mut rng);
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
    fn pow_vartime_reference(base: &Fq, exp: &[u64]) -> Fq {
        let mut res = Fq::one();
        let mut found_one = false;
        for e in exp.iter().rev() {
            for i in (0..64).rev() {
                if found_one {
                    res = Fq::square(&res);
                }

                if ((*e >> i) & 1) == 1 {
                    found_one = true;
                    res = Fq::mul(&res, base);
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
        // The q - 2 exponent used by `invert`.
        [
            0x8c46eb20ffffffff,
            0x224698fc0994a8dd,
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
        Fq::zero(),
        Fq::one(),
        -Fq::one(),
        Fq::random(&mut rng),
        Fq::random(&mut rng),
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
            [0x8c46eb2100000000, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000,
             0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x8c46eb2100000000, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000],
        ),
        // p
        (
            [0x8c46eb2100000001, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000,
             0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ),
        // 2^256 - 1
        (
            [0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
             0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x5b2b3e9cfffffffc, 0x992c350be3420567, 0xffffffffffffffff, 0x3fffffffffffffff],
        ),
        // 2^256
        (
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
             0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x5b2b3e9cfffffffd, 0x992c350be3420567, 0xffffffffffffffff, 0x3fffffffffffffff],
        ),
        // p 2^256
        (
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
             0x8c46eb2100000001, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ),
        // (p - 1) 2^256 + p - 1
        (
            [0x8c46eb2100000000, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000,
             0x8c46eb2100000000, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000],
            [0x311bac8400000003, 0x891a63f02652a376, 0x0000000000000000, 0x0000000000000000],
        ),
        // 2^511
        (
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
             0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000],
            [0xc46eb21000000008, 0x4500ee1cc9575fe0, 0xbfd7118802667ac8, 0x24b6a0d7be67ed54],
        ),
        // 2^512 - 1
        (
            [0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
             0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff],
            [0xfc9678ff0000000e, 0x67bb433d891a16e3, 0x7fae231004ccf590, 0x096d41af7ccfdaa9],
        ),
        // forced_q_r2, the left operand nearest to overflowing `mul` with `R2`
        (
            [0xf3bfcadeeeeeeeef, 0x27fa6352b2545d71, 0xffffffffffffffff, 0xffffffffffffffff,
             0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4eeb097beeeeeeec, 0xc126985e959662d9, 0xfffffffffffffffe, 0x3fffffffffffffff],
        ),
        // forced_q_r3 2^256, the same for `R3`
        (
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
             0x0000000000000d24, 0x00000000000007c2, 0xffffffffffffffff, 0xffffffffffffffff],
            [0x9461ac2cffffcb7d, 0x8bb47985e004a903, 0x120e828ab7b4d3a1, 0x1287a59fa32279f8],
        ),
        // forced_q_r2 + forced_q_r3 2^256
        (
            [0xf3bfcadeeeeeeeef, 0x27fa6352b2545d71, 0xffffffffffffffff, 0xffffffffffffffff,
             0x0000000000000d24, 0x00000000000007c2, 0xffffffffffffffff, 0xffffffffffffffff],
            [0x5705ca87eeeeba68, 0x2a9478e86c0662ff, 0x120e828ab7b4d3a0, 0x1287a59fa32279f8],
        ),
    ];
    for (input, residue) in cases {
        let repr = ff::PrimeField::to_repr(&Fq::from_u512(input));
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
        let two_64 = Fq::from(u64::MAX) + Fq::one();
        l.iter()
            .rev()
            .fold(Fq::zero(), |acc, &limb| acc * two_64 + Fq::from(limb))
    };
    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x3c; 16]);
    let mut inputs = vec![[0u64; 8], [u64::MAX; 8]];
    for _ in 0..1000 {
        inputs.push(core::array::from_fn(|_| rng.try_next_u64().unwrap()));
    }
    for l in inputs {
        assert_eq!(Fq::from_u512(l), horner(l), "limbs {l:x?}");
    }
}

#[cfg(feature = "zeroize")]
#[test]
fn test_zeroize() {
    use zeroize::Zeroize;

    let mut a = <Fq as ff::Field>::ONE;
    a.zeroize();
    assert_eq!(a, Fq::zero());
}

/// Whether `x` holds a reduced residue (a value below the modulus).
#[cfg(test)]
fn is_canonical(x: &Fq) -> bool {
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
    let constants: [(&str, Fq); 11] = [
        ("R", R),
        ("R2", R2),
        ("R3", R3),
        ("GENERATOR", GENERATOR),
        ("ROOT_OF_UNITY", ROOT_OF_UNITY),
        ("DELTA", DELTA),
        ("TWO_INV", <Fq as ff::PrimeField>::TWO_INV),
        (
            "MULTIPLICATIVE_GENERATOR",
            <Fq as ff::PrimeField>::MULTIPLICATIVE_GENERATOR,
        ),
        (
            "ROOT_OF_UNITY_INV",
            <Fq as ff::PrimeField>::ROOT_OF_UNITY_INV,
        ),
        ("ZETA", <Fq as ff::WithSmallOrderMulGroup<3>>::ZETA),
        ("zero", Fq::zero()),
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

use super::*;
use core::cmp::Ordering;
use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

const SEED: [u8; 16] = [
    0x62, 0x64, 0x69, 0x76, 0x73, 0x74, 0x65, 0x70, 0x73, 0x36, 0x32, 0x76, 0x61, 0x72, 0x21, 0x21,
];

/// Fixed regression vectors: internal little-endian representations with
/// their exact expected 62-divstep batch counts, produced by an
/// independently validated simulation of this exact algorithm. If a count
/// drifts, the port has deviated from the validated control flow:
/// investigate the divergence, do not re-pin the count.
const PALLAS_VECTORS: &[([u64; 4], u32)] = &[
    (
        [
            0x34786d38fffffffd,
            0x992c350be41914ad,
            0xffffffffffffffff,
            0x3fffffffffffffff,
        ],
        7,
    ),
    ([0x00000000000005d1, 0, 0, 0], 8),
    (
        [
            0xe096c0a18679d7ae,
            0x2cc4e34bc6b6f06a,
            0x20eddc12b5a7661d,
            0x0c3c5e66537210ad,
        ],
        8,
    ),
    ([1, 0, 0, 0], 9),
    (
        [
            0xa162a7d34ad63d62,
            0xba71beb748b1fa25,
            0x68dc1330fab3847b,
            0x2dc205e082f2c197,
        ],
        10,
    ),
    (
        [
            0x7ceb4d8e9534361d,
            0x8d7879adc97a8e79,
            0x19ed0538ef7b15eb,
            0x35fb58b0ee06c5b4,
        ],
        10,
    ),
];

const VESTA_VECTORS: &[([u64; 4], u32)] = &[
    (
        [
            0x5b2b3e9cfffffffd,
            0x992c350be3420567,
            0xffffffffffffffff,
            0x3fffffffffffffff,
        ],
        7,
    ),
    (
        [
            0x3ea12df55a259593,
            0xd9c58668e724391e,
            0xfcf9d370dd78552a,
            0x1f7fd517b4f7efeb,
        ],
        8,
    ),
    (
        [
            0x4973c2bd762bc27a,
            0x9085d079ecab3a12,
            0x9f66128174a6731a,
            0x3e7853d1fb18fcca,
        ],
        8,
    ),
    ([1, 0, 0, 0], 9),
    ([0x00000ba5f061296c, 0, 0, 0], 10),
    (
        [
            0xc00e79606e554fa8,
            0x13f9947f444f41d3,
            0xd5a780db63e83468,
            0x337e275425a385f3,
        ],
        10,
    ),
];

// ---- signed-62 value helpers (test-side big arithmetic) ----

/// `a + m`, preserving canonical form.
fn s62_add_m(a: &Signed62, m: &[i64; 5]) -> Signed62 {
    let mut r = [0i64; 5];
    let mut carry: i64 = 0;
    for i in 0..4 {
        let s = a.0[i] + m[i] + carry;
        r[i] = s & MASK62 as i64;
        carry = s >> 62;
    }
    r[4] = a.0[4] + m[4] + carry;
    Signed62(r)
}

/// `a - m`, preserving canonical form.
fn s62_sub_m(a: &Signed62, m: &[i64; 5]) -> Signed62 {
    let mut r = [0i64; 5];
    let mut borrow: i64 = 0;
    for i in 0..4 {
        let s = a.0[i] - m[i] + borrow;
        r[i] = s & MASK62 as i64;
        borrow = s >> 62;
    }
    r[4] = a.0[4] - m[4] + borrow;
    Signed62(r)
}

/// `-a`, preserving canonical form.
fn s62_neg(a: &Signed62) -> Signed62 {
    let mut r = [0i64; 5];
    let mut borrow: i64 = 0;
    for i in 0..4 {
        let s = -a.0[i] + borrow;
        r[i] = s & MASK62 as i64;
        borrow = s >> 62;
    }
    r[4] = -a.0[4] + borrow;
    Signed62(r)
}

/// Reduces a value in `(-2m, 2m)` to `[0, m)`.
fn s62_mod_m(a: &Signed62, m: &[i64; 5]) -> Signed62 {
    let zero = Signed62([0; 5]);
    let ms = Signed62(*m);
    let mut x = *a;
    while checks::mul_cmp_62(&x, 5, &zero, 0) == Ordering::Less {
        x = s62_add_m(&x, m);
    }
    while checks::mul_cmp_62(&x, 5, &ms, 1) != Ordering::Less {
        x = s62_sub_m(&x, m);
    }
    x
}

/// Widens a `len`-limb value to five canonical limbs (sign-preserving).
fn s62_widen(s: &Signed62, len: usize) -> Signed62 {
    let mut r = [0i64; 5];
    r[..len].copy_from_slice(&s.0[..len]);
    for i in (len - 1)..4 {
        let carry = r[i] >> 62;
        r[i] &= MASK62 as i64;
        r[i + 1] += carry;
    }
    Signed62(r)
}

// ---- upstream-ordered reference driver with generic kernels ----

/// Generic-modulus port of upstream `secp256k1_modinv64_update_de_62`:
/// multiplies by every modulus limb (including the zero one), applies no
/// first-batch or terminal specialization.
#[allow(clippy::many_single_char_names)]
fn ref_update_de(d: &mut Signed62, e: &mut Signed62, t: &Trans2x2, m: &[i64; 5], mu: u64) {
    let (u, v, q, r) = (t.u, t.v, t.q, t.r);
    let sd = d.0[4] >> 63;
    let se = e.0[4] >> 63;
    let mut md = (u & sd) + (v & se);
    let mut me = (q & sd) + (r & se);
    let mut cd = i128::from(u) * i128::from(d.0[0]) + i128::from(v) * i128::from(e.0[0]);
    let mut ce = i128::from(q) * i128::from(d.0[0]) + i128::from(r) * i128::from(e.0[0]);
    md = md.wrapping_sub((mu.wrapping_mul(cd as u64).wrapping_add(md as u64) & MASK62) as i64);
    me = me.wrapping_sub((mu.wrapping_mul(ce as u64).wrapping_add(me as u64) & MASK62) as i64);
    cd += i128::from(m[0]) * i128::from(md);
    ce += i128::from(m[0]) * i128::from(me);
    assert!(cd as u64 & MASK62 == 0 && ce as u64 & MASK62 == 0);
    cd >>= 62;
    ce >>= 62;
    let d_in = *d;
    let e_in = *e;
    for i in 1..5 {
        cd += i128::from(u) * i128::from(d_in.0[i]) + i128::from(v) * i128::from(e_in.0[i]);
        ce += i128::from(q) * i128::from(d_in.0[i]) + i128::from(r) * i128::from(e_in.0[i]);
        cd += i128::from(m[i]) * i128::from(md);
        ce += i128::from(m[i]) * i128::from(me);
        d.0[i - 1] = (cd as u64 & MASK62) as i64;
        cd >>= 62;
        e.0[i - 1] = (ce as u64 & MASK62) as i64;
        ce >>= 62;
    }
    assert!(cd == i128::from(cd as i64) && ce == i128::from(ce as i64));
    d.0[4] = cd as i64;
    e.0[4] = ce as i64;
}

/// Generic-modulus copy of [`normalize_62`].
fn ref_normalize(r: &mut Signed62, sign: i64, m: &[i64; 5]) {
    let mut l = r.0;
    let mut cond_add = l[4] >> 63;
    for i in 0..5 {
        l[i] += m[i] & cond_add;
    }
    let cond_negate = sign >> 63;
    for i in 0..5 {
        l[i] = (l[i] ^ cond_negate) - cond_negate;
    }
    for i in 0..4 {
        l[i + 1] += l[i] >> 62;
        l[i] &= MASK62 as i64;
    }
    cond_add = l[4] >> 63;
    for i in 0..5 {
        l[i] += m[i] & cond_add;
    }
    for i in 0..4 {
        l[i + 1] += l[i] >> 62;
        l[i] &= MASK62 as i64;
    }
    r.0 = l;
}

/// One recorded batch of a reference run: the transition matrix and the
/// full state *before* the batch's updates.
#[derive(Clone, Copy)]
struct BatchRec {
    t: Trans2x2,
    d_in: Signed62,
    e_in: Signed62,
    f_in: Signed62,
    g_in: Signed62,
    len: usize,
    terminal: bool,
}

struct RefRun {
    out: [u64; 4],
    batches: u32,
    trace: [BatchRec; 12],
}

/// Upstream-ordered reference driver: d,e updated with the generic kernel
/// *every* batch (including the terminal one) before the g == 0 check —
/// the ordering `secp256k1_modinv64_var` uses. Output must be limb-exact
/// equal to the production driver's.
fn ref_invert(x: &[u64; 4], m: &[i64; 5], mu: u64, e_seed: &[i64; 5]) -> Option<RefRun> {
    if x == &[0u64; 4] {
        return None;
    }
    let dummy = BatchRec {
        t: Trans2x2 {
            u: 1,
            v: 0,
            q: 0,
            r: 1,
        },
        d_in: Signed62([0; 5]),
        e_in: Signed62([0; 5]),
        f_in: Signed62([0; 5]),
        g_in: Signed62([0; 5]),
        len: 5,
        terminal: false,
    };
    let mut trace = [dummy; 12];
    let mut f = Signed62(*m);
    let mut g = pack62(x);
    let mut d = Signed62([0; 5]);
    let mut e = Signed62(*e_seed);
    let mut eta: i64 = -1;
    let mut len: usize = 5;
    for batch in 1..=12u32 {
        let (t, new_eta) = divsteps_62_var(eta, f.0[0] as u64, g.0[0] as u64);
        eta = new_eta;
        let rec = &mut trace[(batch - 1) as usize];
        *rec = BatchRec {
            t,
            d_in: d,
            e_in: e,
            f_in: f,
            g_in: g,
            len,
            terminal: false,
        };
        ref_update_de(&mut d, &mut e, &t, m, mu);
        update_fg_62_var(len, &mut f, &mut g, &t);
        if is_zero(&g, len) {
            trace[(batch - 1) as usize].terminal = true;
            ref_normalize(&mut d, f.0[len - 1], m);
            return Some(RefRun {
                out: unpack62(&d),
                batches: batch,
                trace,
            });
        }
        shrink_len(&mut f, &mut g, &mut len);
    }
    panic!("reference driver exceeded 12 batches");
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn checked_mode_is_active() {
    // T24: the verify! strategy relies on cfg(test) being live even in
    // --release test runs; a refactor to debug_assert! would disarm it.
    assert!(cfg!(any(test, debug_assertions)));
}

macro_rules! modinv_field_tests {
    ($F:path, $mod:ident, $P:ty, $fermat_exp:expr_2021, $vectors:expr_2021) => {
        mod $mod {
            use super::*;

            fn modulus62() -> [i64; 5] {
                <$P as InvParams>::MODULUS
            }

            fn m64() -> [u64; 4] {
                unpack62(&Signed62(modulus62()))
            }

            fn fermat_invert(x: &$F) -> $F {
                x.pow_vartime(&$fermat_exp)
            }

            fn rnd_elt(rng: &mut XorShiftRng) -> $F {
                <$F>::from_raw([
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ])
            }

            /// Lifts a signed-62 value in (-2m, 2m) into the field by the
            /// identity representation map (its canonical residue becomes
            /// the internal representation).
            fn lift(s: &Signed62) -> $F {
                let m = modulus62();
                $F(unpack62(&s62_mod_m(s, &m)))
            }

            fn lift_fg(s: &Signed62, len: usize) -> $F {
                lift(&s62_widen(s, len))
            }

            /// Asserts the scaled invariants Xd = R^2 f and Xe = R^2 g
            /// (mod m) at every recorded batch boundary of a reference
            /// run. Field multiplication divides both sides by the same
            /// R, so representation-level products compare exactly.
            fn assert_trace_congruences(x_repr: &[u64; 4], run: &RefRun) {
                let xf = $F(*x_repr);
                let r2f = $F(unpack62(&Signed62(<$P as InvParams>::R2)));
                for rec in run.trace[..run.batches as usize].iter() {
                    assert_eq!(
                        xf * lift(&rec.d_in),
                        r2f * lift_fg(&rec.f_in, rec.len),
                        "Xd = R^2 f violated"
                    );
                    assert_eq!(
                        xf * lift(&rec.e_in),
                        r2f * lift_fg(&rec.g_in, rec.len),
                        "Xe = R^2 g violated"
                    );
                }
            }

            fn ref_run(x: &[u64; 4]) -> RefRun {
                ref_invert(
                    x,
                    &modulus62(),
                    <$P as InvParams>::MU,
                    &<$P as InvParams>::R2,
                )
                .expect("nonzero input")
            }

            #[test]
            fn constants_cross_check() {
                // T1: the radix-62 constants are computed, not trusted.
                // Rebuild m from field arithmetic: (-1).to_repr() + 1.
                let m_minus_1 = (-<$F>::ONE).to_repr();
                let mut limbs = [0u64; 4];
                for (i, chunk) in m_minus_1.as_ref().chunks(8).enumerate() {
                    limbs[i] = u64::from_le_bytes(chunk.try_into().unwrap());
                }
                let mut carry = 1u64;
                for l in limbs.iter_mut() {
                    let (s, c) = l.overflowing_add(carry);
                    *l = s;
                    carry = u64::from(c);
                }
                assert_eq!(limbs, m64(), "MODULUS mismatch vs field constants");

                // The representation of the field value 2^256 is
                // 2^256 * R = R^2 (mod m): exactly the R2 constant.
                let r_val = <$F>::from_u128(1u128 << 64).square().square();
                assert_eq!(
                    unpack62(&Signed62(<$P as InvParams>::R2)),
                    r_val.0,
                    "R2 mismatch vs field constants"
                );

                // mu * m = 1 (mod 2^62), and the sparse shape the kernels
                // hardcode.
                let m62 = modulus62();
                assert_eq!(
                    <$P as InvParams>::MU.wrapping_mul(m62[0] as u64) & MASK62,
                    1
                );
                assert_eq!(&m62[2..], &[2, 0, 64]);
                // R2 is canonical and reduced.
                let r2 = Signed62(<$P as InvParams>::R2);
                for l in r2.0 {
                    assert!(l >= 0 && (l as u64) <= MASK62);
                }
                assert!(
                    checks::mul_cmp_62(&r2, 5, &Signed62(m62), 1) == Ordering::Less,
                    "R2 must be below m"
                );
            }

            #[test]
            fn pack_unpack_roundtrip() {
                // T2
                let m = m64();
                let mut cases = [[0u64; 4]; 16];
                cases[1] = [1, 0, 0, 0];
                // m - 1:
                cases[2] = {
                    let mut t = m;
                    t[0] -= 1;
                    t
                };
                // single bits (all below both moduli, which exceed 2^254):
                let bits = [1u32, 61, 62, 63, 124, 186, 248, 254];
                for (i, b) in bits.iter().enumerate() {
                    let mut t = [0u64; 4];
                    t[(b / 64) as usize] = 1u64 << (b % 64);
                    cases[3 + i] = t;
                }
                cases[11] = [MASK62, 0, 0, 0];
                cases[12] = [u64::MAX, 0, 0, 0];
                cases[13] = [0, u64::MAX, 0, 0];
                cases[14] = [0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555, 0, 0];
                cases[15] = [u64::MAX, u64::MAX, u64::MAX, 0x3fff_ffff_ffff_ffff];
                for c in cases {
                    assert_eq!(unpack62(&pack62(&c)), c);
                }
                let mut rng = XorShiftRng::from_seed(SEED);
                for _ in 0..1024 {
                    let c = rnd_elt(&mut rng).0;
                    let p = pack62(&c);
                    for l in p.0 {
                        assert!(l >= 0 && (l as u64) <= MASK62);
                    }
                    assert_eq!(unpack62(&p), c);
                }
            }

            #[test]
            fn divsteps_matrix_properties() {
                // T3
                let mut rng = XorShiftRng::from_seed(SEED);
                for _ in 0..10_000 {
                    let f0 = rng.next_u64() | 1;
                    let g0 = rng.next_u64();
                    let eta_in = (rng.next_u64() % 61) as i64 - 30;
                    let (t, eta_out) = divsteps_62_var(eta_in, f0, g0);
                    assert_eq!(
                        i128::from(t.u) * i128::from(t.r) - i128::from(t.v) * i128::from(t.q),
                        1 << 62
                    );
                    assert!(t.u.unsigned_abs() + t.v.unsigned_abs() <= 1 << 62);
                    assert!(t.q.unsigned_abs() + t.r.unsigned_abs() <= 1 << 62);
                    // After 62 divsteps both rows are divisible by 2^62.
                    assert_eq!(
                        (t.u as u64)
                            .wrapping_mul(f0)
                            .wrapping_add((t.v as u64).wrapping_mul(g0))
                            & MASK62,
                        0
                    );
                    assert_eq!(
                        (t.q as u64)
                            .wrapping_mul(f0)
                            .wrapping_add((t.r as u64).wrapping_mul(g0))
                            & MASK62,
                        0
                    );
                    assert!(eta_out >= eta_in - 62 && eta_out <= eta_in.abs() + 62);
                }
            }

            #[test]
            fn divsteps_zero_g_identity() {
                // T10: with g0 = 0 the whole batch is halvings, giving
                // the matrix diag(2^62, 1); update_fg then keeps f and
                // shifts g down one limb.
                let mut rng = XorShiftRng::from_seed(SEED);
                for _ in 0..64 {
                    let f0 = rng.next_u64() | 1;
                    let (t, _) = divsteps_62_var(-1, f0, 0);
                    assert_eq!(t.u, 1 << 62);
                    assert_eq!(t.v, 0);
                    assert_eq!(t.q, 0);
                    assert_eq!(t.r, 1);
                    let mut f = pack62(&rnd_elt(&mut rng).0);
                    let f_before = f;
                    let mut g = pack62(&rnd_elt(&mut rng).0);
                    g.0[0] = 0;
                    let g_before = g;
                    update_fg_62_var(5, &mut f, &mut g, &t);
                    assert_eq!(f, f_before);
                    assert_eq!(
                        g.0,
                        [
                            g_before.0[1],
                            g_before.0[2],
                            g_before.0[3],
                            g_before.0[4],
                            0
                        ]
                    );
                }
            }

            #[test]
            fn update_de_sparse_matches_generic() {
                // T4 + T8: the production sparse-modulus kernel is
                // limb-exact against the generic reference over real
                // trajectory states and synthetic boundary states, and
                // its outputs respect the (-2m, m) range invariant.
                let m = modulus62();
                let mu = <$P as InvParams>::MU;
                let mut rng = XorShiftRng::from_seed(SEED);
                let check_state = |d: &Signed62, e: &Signed62, t: &Trans2x2| {
                    let mut d_prod = *d;
                    let mut e_prod = *e;
                    update_de_62::<$P>(&mut d_prod, &mut e_prod, t);
                    let mut d_ref = *d;
                    let mut e_ref = *e;
                    ref_update_de(&mut d_ref, &mut e_ref, t, &m, mu);
                    assert_eq!(d_prod, d_ref, "d limbs diverge from generic kernel");
                    assert_eq!(e_prod, e_ref, "e limbs diverge from generic kernel");
                    checks::assert_coeff_range::<$P>(&d_prod, "d output");
                    checks::assert_coeff_range::<$P>(&e_prod, "e output");
                };
                // Real trajectory states.
                for _ in 0..50 {
                    let x = rnd_elt(&mut rng).0;
                    let run = ref_run(&x);
                    for rec in run.trace[..run.batches as usize].iter() {
                        check_state(&rec.d_in, &rec.e_in, &rec.t);
                    }
                }
                // Synthetic boundary states under real matrices.
                let zero = Signed62([0; 5]);
                let one = Signed62([1, 0, 0, 0, 0]);
                let m_s = Signed62(m);
                let m_minus_1 = s62_neg(&s62_sub_m(&one, &m)); // -(1 - m)
                let neg_one = s62_neg(&one);
                let neg_m = s62_neg(&m_s);
                let neg_2m_plus_1 = s62_add_m(&s62_neg(&s62_add_m(&m_s, &m)), &one.0);
                let boundary = [zero, one, m_minus_1, neg_one, neg_m, neg_2m_plus_1];
                let sample = ref_run(&rnd_elt(&mut rng).0);
                for rec in sample.trace[..sample.batches as usize].iter() {
                    for d in &boundary {
                        for e in &boundary {
                            check_state(d, e, &rec.t);
                        }
                    }
                }
            }

            #[test]
            fn update_fg_first_matches_generic() {
                // T5
                let m = modulus62();
                let mut rng = XorShiftRng::from_seed(SEED);
                for _ in 0..1000 {
                    let x = rnd_elt(&mut rng).0;
                    let g0 = pack62(&x);
                    let (t, _) = divsteps_62_var(-1, m[0] as u64, g0.0[0] as u64);
                    let mut f_prod = Signed62(m);
                    let mut g_prod = g0;
                    update_fg_62_first::<$P>(&mut f_prod, &mut g_prod, &t);
                    let mut f_ref = Signed62(m);
                    let mut g_ref = g0;
                    update_fg_62_var(5, &mut f_ref, &mut g_ref, &t);
                    assert_eq!(f_prod, f_ref);
                    assert_eq!(g_prod, g_ref);
                }
            }

            #[test]
            fn update_de_first_matches_generic() {
                // T6
                let m = modulus62();
                let mu = <$P as InvParams>::MU;
                let mut rng = XorShiftRng::from_seed(SEED);
                for i in 0..1000 {
                    // The real first-batch e is R2; also cover random
                    // e in [0, m).
                    let e = if i == 0 {
                        Signed62(<$P as InvParams>::R2)
                    } else {
                        pack62(&rnd_elt(&mut rng).0)
                    };
                    let g0 = pack62(&rnd_elt(&mut rng).0);
                    let (t, _) = divsteps_62_var(-1, m[0] as u64, g0.0[0] as u64);
                    let (d_prod, e_prod) = update_de_62_first::<$P>(&e, &t);
                    let mut d_ref = Signed62([0; 5]);
                    let mut e_ref = e;
                    ref_update_de(&mut d_ref, &mut e_ref, &t, &m, mu);
                    assert_eq!(d_prod, d_ref);
                    assert_eq!(e_prod, e_ref);
                }
            }

            #[test]
            fn update_d_only_matches_generic_row0() {
                // T7: the terminal kernel equals the d-row of the generic
                // update on real terminal states.
                let m = modulus62();
                let mu = <$P as InvParams>::MU;
                let mut rng = XorShiftRng::from_seed(SEED);
                let mut seen_terminal = 0u32;
                for _ in 0..200 {
                    let x = rnd_elt(&mut rng).0;
                    let run = ref_run(&x);
                    let rec = &run.trace[(run.batches - 1) as usize];
                    assert!(rec.terminal);
                    let mut d_prod = rec.d_in;
                    update_d_only_62::<$P>(&mut d_prod, &rec.e_in, rec.t.u, rec.t.v);
                    let mut d_ref = rec.d_in;
                    let mut e_ref = rec.e_in;
                    ref_update_de(&mut d_ref, &mut e_ref, &rec.t, &m, mu);
                    assert_eq!(d_prod, d_ref);
                    seen_terminal += 1;
                }
                assert_eq!(seen_terminal, 200);
            }

            #[test]
            fn normalize_quadrants() {
                // T9: all sign/range quadrants, against reference
                // reduction, catching in particular any wrong-sign-limb
                // regression.
                let m = modulus62();
                let m_s = Signed62(m);
                let one = Signed62([1, 0, 0, 0, 0]);
                let mut rng = XorShiftRng::from_seed(SEED);
                let mut cases: [Signed62; 24] = [Signed62([0; 5]); 24];
                cases[0] = s62_add_m(&s62_neg(&s62_add_m(&m_s, &m)), &one.0); // -2m + 1
                cases[1] = s62_neg(&s62_add_m(&m_s, &one.0)); // -m - 1
                cases[2] = s62_neg(&m_s); // -m
                cases[3] = s62_add_m(&s62_neg(&m_s), &one.0); // -m + 1
                cases[4] = s62_neg(&one); // -1
                cases[5] = Signed62([0; 5]); // 0
                cases[6] = one; // 1
                cases[7] = s62_sub_m(&s62_add_m(&Signed62([0; 5]), &m), &one.0); // m - 1
                for i in 8..24 {
                    // random values across (-2m, m): canonical in [0, m)
                    // minus 0, 1, or 2 copies of m.
                    let mut x = pack62(&rnd_elt(&mut rng).0);
                    for _ in 0..(i % 3) {
                        x = s62_sub_m(&x, &m);
                    }
                    cases[i] = x;
                }
                for c in &cases {
                    for sign in [1i64, -1] {
                        let mut got = *c;
                        normalize_62::<$P>(&mut got, sign);
                        let expected = if sign < 0 {
                            s62_mod_m(&s62_neg(c), &m)
                        } else {
                            s62_mod_m(c, &m)
                        };
                        assert_eq!(got, expected, "normalize mismatch");
                    }
                }
            }

            #[test]
            fn invert_matches_fermat_50k() {
                // T11
                let mut rng = XorShiftRng::from_seed(SEED);
                for _ in 0..50_000 {
                    let x = rnd_elt(&mut rng);
                    if bool::from(x.is_zero()) {
                        continue;
                    }
                    let (out, batches) = invert_counted::<$P>(&x.0).expect("nonzero");
                    assert!(batches <= 12);
                    assert_eq!(out, fermat_invert(&x).0);
                }
            }

            #[test]
            fn invert_matches_reference_driver() {
                // T12: limb-exact against the upstream-ordered generic
                // driver, whose per-batch states also satisfy the scaled
                // congruence invariants.
                let mut rng = XorShiftRng::from_seed(SEED);
                for i in 0..2000 {
                    let x = rnd_elt(&mut rng).0;
                    let (out, batches) = invert_counted::<$P>(&x).expect("nonzero");
                    let run = ref_run(&x);
                    assert_eq!(out, run.out, "production diverges from reference");
                    assert_eq!(batches, run.batches, "batch count diverges");
                    if i < 200 {
                        assert_trace_congruences(&x, &run);
                    }
                }
                for (repr, _) in $vectors {
                    let (out, batches) = invert_counted::<$P>(repr).expect("nonzero");
                    let run = ref_run(repr);
                    assert_eq!(out, run.out);
                    assert_eq!(batches, run.batches);
                    assert_trace_congruences(repr, &run);
                }
            }

            #[test]
            fn invert_fixed_edge_values() {
                // T13: field-value edges. `invert()` here is already the
                // divstep implementation under test.
                let r_val = <$F>::from_u128(1u128 << 64).square().square();
                let mut edge = [
                    <$F>::ONE,
                    -<$F>::ONE,
                    <$F>::from(2u64),
                    <$F>::TWO_INV,
                    <$F>::MULTIPLICATIVE_GENERATOR,
                    <$F>::ROOT_OF_UNITY,
                    <$F>::ZETA,
                    r_val,
                    r_val.square(),
                ]
                .to_vec();
                for k in 2u64..=64 {
                    edge.push(<$F>::from(k));
                }
                for x in edge {
                    let inv = x.invert().unwrap();
                    assert_eq!(x * inv, <$F>::ONE);
                    assert_eq!(inv, fermat_invert(&x));
                }
            }

            #[test]
            fn invert_fixed_edge_reprs() {
                // T14: raw representation edges (trailing-zero, eta, and
                // length-shrink stress). All values below m.
                let m = m64();
                let mut reprs = std::vec::Vec::new();
                reprs.push([1u64, 0, 0, 0]);
                reprs.push([2, 0, 0, 0]);
                reprs.push([3, 0, 0, 0]);
                let mut m_minus = m;
                m_minus[0] -= 1;
                reprs.push(m_minus); // m - 1
                m_minus[0] -= 1;
                reprs.push(m_minus); // m - 2
                for k in [1u32, 61, 62, 63, 123, 124, 186, 247, 254] {
                    let mut t = [0u64; 4];
                    t[(k / 64) as usize] = 1u64 << (k % 64);
                    reprs.push(t);
                }
                reprs.push([MASK62, 0, 0, 0]); // 2^62 - 1
                for k in [1u64, 3, 0x5d1] {
                    // k << 200
                    reprs.push([0, 0, 0, k << 8]);
                }
                let mut rng = XorShiftRng::from_seed(SEED);
                for _ in 0..16 {
                    reprs.push([
                        0,
                        rng.next_u64(),
                        rng.next_u64(),
                        rng.next_u64() & 0x3fff_ffff_ffff_ffff,
                    ]);
                }
                for repr in reprs {
                    let x = $F(repr);
                    let inv = x.invert().unwrap();
                    assert_eq!(x * inv, <$F>::ONE, "repr {:x?}", repr);
                    assert_eq!(inv, fermat_invert(&x), "repr {:x?}", repr);
                }
            }

            #[test]
            fn invert_zero_is_none() {
                // T15
                assert!(invert::<$P>(&[0u64; 4]).is_none());
                assert!(bool::from(<$F>::ZERO.invert().is_none()));
                assert!(bool::from(<$F>::ONE.invert().is_some()));
            }

            #[test]
            fn batch_count_distribution_pinned() {
                // T16: exact batch counts for the searched vectors. See
                // the vector table's comment: on mismatch, investigate;
                // do not re-pin.
                for (repr, expected) in $vectors {
                    let (out, batches) = invert_counted::<$P>(repr).expect("nonzero");
                    assert_eq!(batches, *expected, "batch count drifted for {:x?}", repr);
                    assert_eq!($F(*repr) * $F(out), <$F>::ONE);
                }
            }

            #[test]
            fn invert_self_consistency_200k() {
                // T17: x * x^-1 == 1 is a complete correctness check for
                // inversion; also bounds the batch-count distribution.
                let mut rng = XorShiftRng::from_seed(SEED);
                let mut histogram = [0u32; 13];
                for _ in 0..200_000 {
                    let x = rnd_elt(&mut rng);
                    if bool::from(x.is_zero()) {
                        continue;
                    }
                    let (out, batches) = invert_counted::<$P>(&x.0).expect("nonzero");
                    histogram[batches as usize] += 1;
                    assert_eq!($F(out) * x, <$F>::ONE);
                }
                for (batches, count) in histogram.iter().enumerate() {
                    assert!(
                        *count == 0 || (5..=12).contains(&batches),
                        "unexpected batch count {} ({} times)",
                        batches,
                        count
                    );
                }
                // Structured families exercise the long tails.
                for j in 0..252u32 {
                    let mut repr = [0u64; 4];
                    repr[(j / 64) as usize] = 1u64 << (j % 64);
                    repr[0] |= 5;
                    let x = $F(repr);
                    let (out, batches) = invert_counted::<$P>(&x.0).expect("nonzero");
                    assert!(batches <= 12);
                    assert_eq!($F(out) * x, <$F>::ONE);
                }
            }

            #[test]
            fn termination_bound_constants() {
                // T18: Bernstein-Yang iteration bound for 255-bit inputs,
                // as used by the driver's 12-batch cap.
                assert_eq!((49 * 255 + 57) / 17, 738);
                assert!(738 <= 12 * 62);
            }

            #[test]
            fn r2_seeding_equivalence() {
                // T19: seeding e with R^2 is exactly a classic (e = 1)
                // inversion followed by two R2 Montgomery multiplications.
                let m = modulus62();
                let one_seed = [1i64, 0, 0, 0, 0];
                let r2f = $F(unpack62(&Signed62(<$P as InvParams>::R2)));
                let mut rng = XorShiftRng::from_seed(SEED);
                for _ in 0..1000 {
                    let x = rnd_elt(&mut rng);
                    if bool::from(x.is_zero()) {
                        continue;
                    }
                    let classic = ref_invert(&x.0, &m, <$P as InvParams>::MU, &one_seed)
                        .expect("nonzero")
                        .out;
                    let production = invert::<$P>(&x.0).expect("nonzero");
                    assert_eq!(($F(classic) * r2f * r2f).0, production);
                }
            }

            proptest::proptest! {
                #[test]
                fn pbt_inverse(a in proptest::prelude::any::<[u64; 4]>()) {
                    // T25
                    let x = <$F>::from_raw(a);
                    if !bool::from(x.is_zero()) {
                        let inv = x.invert().unwrap();
                        proptest::prop_assert_eq!(inv, fermat_invert(&x));
                        proptest::prop_assert_eq!(x * inv, <$F>::ONE);
                        proptest::prop_assert_eq!(inv.invert().unwrap(), x);
                    }
                }
            }
        }
    };
}

modinv_field_tests!(
    crate::fields::Fp,
    fp,
    FpParams,
    [
        0x992d_30ec_ffff_ffff,
        0x2246_98fc_094c_f91b,
        0x0000_0000_0000_0000,
        0x4000_0000_0000_0000,
    ],
    PALLAS_VECTORS
);

modinv_field_tests!(
    crate::fields::Fq,
    fq,
    FqParams,
    [
        0x8c46_eb20_ffff_ffff,
        0x2246_98fc_0994_a8dd,
        0x0000_0000_0000_0000,
        0x4000_0000_0000_0000,
    ],
    VESTA_VECTORS
);

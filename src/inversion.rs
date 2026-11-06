//! The constant-time inversion: one Rust driver over the six blocks of a backend, which implements
//! [`InvertBlocks`]. `lean/PastaCurves/Compositions.lean` mirrors the driver over a record of the
//! blocks, and `lean/PastaCurves/Inversion/Composition.lean` proves it equal to the algorithm's
//! model for any blocks that meet the record of their specifications.
//!
//! [`invert`] runs the AArch64 assembly blocks where the AArch64 backend is compiled, and the
//! same blocks in portable Rust (`portable`) everywhere else.

// The inversion is not yet reached from the field types.
#![allow(dead_code)]

use crate::limbs::Limbs;

if_asm_supported! {
    // The portable blocks serve wherever the assembly blocks do not; beside the AArch64 backend
    // they are compiled for their tests and the documentation only.
    #[cfg(any(not(target_arch = "aarch64"), test, doc))]
    mod portable;
}
if_asm_unsupported! {
    mod portable;
}

/// The six blocks of the inversion, as a backend provides them. The trait mirrors the Lean
/// record `InvertBlocks`, and each method's contract restates in words the block specification
/// that `InvertBlocks.Spec` requires of it. A backend that meets the contracts gets the driver's
/// correctness from the shared proof.
pub(crate) trait InvertBlocks {
    /// Fifty-nine half-delta divsteps on the low words of `f` and `g`. `two_delta = 2δ`, as a
    /// two's-complement word, and `f0` and `g0` are the low 64 bits of `f` and `g`, `f` odd.
    /// Returns `[two_delta', u, v, q, r]`: the new `two_delta`, and the 59-step transition matrix
    /// `M` with `2^59 (f', g') = M (f, g)`, as two's-complement words.
    fn divstep59(two_delta: u64, f0: u64, g0: u64) -> [u64; 5];

    /// The sign-magnitude form of a transition matrix whose entries are below `2^63` in magnitude:
    /// `[u, v, q, r, su, sv, sq, sr]`, each entry's magnitude, then its sign as a mask,
    /// all ones for a negative entry and zero otherwise.
    fn sign_mag(u: u64, v: u64, q: u64, r: u64) -> [u64; 8];

    /// The row `(a f + b g) / 2^59` of the `f`, `g` update, rounded down, in five two's-complement
    /// words. `f` and `g` are five-word values below `2^256` in magnitude, and `a` and `b` are
    /// given as the magnitudes `m0`, `m1` with the sign masks `s0`, `s1`, with `|a| + |b| ≤ 2^63`.
    fn fg_row(f: &[u64; 5], g: &[u64; 5], m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5];

    /// The row `a d + b e` of the `d`, `e` combination, exact, in five two's-complement words. `d`
    /// and `e` are four-word values, and `a` and `b` are given as for [`fg_row`](Self::fg_row).
    fn de_row(d: &Limbs, e: &Limbs, m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5];

    /// The almost-Montgomery reduction of a five-word value `t` with `|t| < 2^315`: adds `2^61 p`,
    /// then performs one word of Montgomery reduction. The result is below `2p` and below `2^256`,
    /// and congruent to `t / 2^64` modulo `p`.
    fn amontred(t: &[u64; 5], modulus: &Limbs, inv: u64) -> Limbs;

    /// Subtracts the modulus from `value` unless the subtraction borrows: `value mod p` for
    /// `value < 2p`.
    fn cond_sub(value: &Limbs, modulus: &Limbs) -> Limbs;
}

/// The constant-time inversion in Montgomery form over the blocks `B`: for a canonical `x`, the
/// canonical `z` with `x · z ≡ 2^512 (mod p)`, and `0` for `x = 0`.
///
/// The composition of the blocks that `lean/PastaCurves/Inversion/Model.lean`'s `montInvModel`
/// specifies. From `(two_delta, f, g, d, e) = (1, p, x, 0, e0)`, with `e0 = 2^562 mod p`, nine rounds each
/// run `divstep59` on the low words of `f` and `g`, take the matrix's sign-magnitude form, update
/// `f` and `g` by its two rows, and combine `d` and `e` by the two rows, each combination reduced
/// by `amontred`. The invariant after round `i` is `(f, g) ≡ x · 2^(5i - 562) · (d, e) (mod p)`.
/// The tenth round computes only `d`, with the sign of the new `f` (which is `±1`, `g` being `0`)
/// folded into the row's masks, and reduces strictly. That sign is the top bit of the low word of
/// `u · f + v · g`, since that sum is `2^59 · f'`.
#[inline]
pub(crate) fn invert_with<B: InvertBlocks>(
    x: &Limbs,
    modulus: &Limbs,
    inv: u64,
    e0: &Limbs,
) -> Limbs {
    let mut two_delta: u64 = 1;
    let mut f = [modulus[0], modulus[1], modulus[2], modulus[3], 0];
    let mut g = [x[0], x[1], x[2], x[3], 0];
    let mut d: Limbs = [0; 4];
    let mut e: Limbs = *e0;
    for _ in 0..9 {
        let [two_delta_new, u, v, q, r] = B::divstep59(two_delta, f[0], g[0]);
        two_delta = two_delta_new;
        let [u, v, q, r, su, sv, sq, sr] = B::sign_mag(u, v, q, r);
        let f_new = B::fg_row(&f, &g, u, v, su, sv);
        g = B::fg_row(&f, &g, q, r, sq, sr);
        f = f_new;
        let td = B::de_row(&d, &e, u, v, su, sv);
        let te = B::de_row(&d, &e, q, r, sq, sr);
        d = B::amontred(&td, modulus, inv);
        e = B::amontred(&te, modulus, inv);
    }
    let [_, u, v, q, r] = B::divstep59(two_delta, f[0], g[0]);
    let sign = ((f[0].wrapping_mul(u).wrapping_add(g[0].wrapping_mul(v))) as i64 >> 63) as u64;
    let [u, v, _, _, su, sv, _, _] = B::sign_mag(u, v, q, r);
    let t = B::de_row(&d, &e, u, v, su ^ sign, sv ^ sign);
    B::cond_sub(&B::amontred(&t, modulus, inv), modulus)
}

if_asm_supported! {
    /// The blocks that [`invert`] runs: the AArch64 assembly blocks where the AArch64 backend is
    /// compiled, and the portable blocks elsewhere.
    #[cfg(target_arch = "aarch64")]
    type Selected = crate::asm::aarch64::Backend;
    /// The blocks that [`invert`] runs: the AArch64 assembly blocks where the AArch64 backend is
    /// compiled, and the portable blocks elsewhere.
    #[cfg(not(target_arch = "aarch64"))]
    type Selected = portable::Backend;
}
if_asm_unsupported! {
    /// The blocks that [`invert`] runs: the AArch64 assembly blocks where the AArch64 backend is
    /// compiled, and the portable blocks elsewhere.
    type Selected = portable::Backend;
}

/// Inverts a canonical Montgomery residue for a Pasta modulus, in constant time.
///
/// Returns the canonical `z` with `x * z ≡ 2^512 (mod p)`: for `x` the Montgomery form of a nonzero
/// residue `X`, `z` is the Montgomery form of `X^-1`; for `x = 0` it is `0`, so a caller that needs
/// an optional inverse checks for zero separately. The algorithm is the serial variant of
/// Bernstein, Chen, Harrison, Huang, Maxwell, Wang, Wuille, and Yang, "Accelerating and verifying
/// constant-time modular inversion" (EUROCRYPT 2026), as in s2n-bignum's `bignum_montinv_p256`: 590
/// half-delta divsteps in ten rounds of 59, computed on packed words, with the coefficients reduced
/// by one Montgomery word per round. It runs a fixed sequence of six blocks, in register-only
/// assembly on AArch64 and in portable Rust on every other target (and on AArch64 with the assembly
/// disabled), with no data-dependent branch or memory access in either, so its timing does not
/// depend on `x`. The design and the correctness argument are in `book/src/design/inversion.md`.
///
/// Outputs are canonical.
///
/// # Safety
///
/// `x` must be canonical. This is debug-asserted. Under that precondition the machine-checked
/// proofs in `lean/` establish the result on AArch64 (`invert_entry_spec`, from `montInv_spec` on
/// words and the six block proofs). Of the portable blocks, only `cond_sub` is proved, from its
/// Aeneas translation; all six are tested against the same known answers as the assembly blocks.
///
/// `modulus` must be either the Pallas or Vesta field modulus, `inv` must be correctly derived
/// from it, and `e0` must be `2^562 mod p`, the starting value of the coefficient `e`, which
/// compensates the ten one-word Montgomery reductions (`2^562 = 2^(512 + 5 * 10)`). Any other
/// values will cause undefined results.
#[inline]
pub(crate) fn invert(x: &Limbs, modulus: &Limbs, inv: u64, e0: &Limbs) -> Limbs {
    debug_assert!(
        crate::limbs::is_canonical(x, modulus),
        "pasta_curves::inversion::invert requires a canonical input"
    );
    invert_with::<Selected>(x, modulus, inv, e0)
}

/// Known answers for every block and for the inversion, and the checks that run them over any
/// backend's blocks; a backend's own test module calls the checks with its blocks.
#[cfg(test)]
pub(crate) mod tests;

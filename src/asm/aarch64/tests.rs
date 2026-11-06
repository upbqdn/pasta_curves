use super::Backend;
use crate::inversion::tests::DIVSTEP59_VECTORS;
use core::arch::asm;

/// One step of the `divstep!` macro on a packed state, for tracing the block step by step:
/// the parity test that precedes a batch's first step, then the step without the test that
/// would feed a next one.
fn divstep_once(mut two_delta: u64, mut pf: u64, mut pg: u64) -> [u64; 3] {
    // SAFETY: register-only arithmetic with declared inputs and outputs.
    unsafe {
        asm!(
            "tst {pg}, #1",
            divstep!(last),
            two_delta = inout(reg) two_delta,
            pf = inout(reg) pf,
            pg = inout(reg) pg,
            t = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [two_delta, pf, pg]
}

/// The packed state `(two_delta, pf, pg)` before each of the first batch's twenty steps on the
/// first vector of `DIVSTEP59_VECTORS`, and after the last, from the integer recurrence on
/// the packed state (`Inversion/Packed.lean`'s `packedDivsteps`).
const DIVSTEP_TRACE: [[u64; 3]; 21] = [
    [0x0000000000000001, 0xfffffe0000000001, 0xc0000000000d9046],
    [0x0000000000000003, 0xfffffe0000000001, 0xe00000000006c823],
    [0xffffffffffffffff, 0xe00000000006c823, 0xf000010000036411],
    [0x0000000000000001, 0xe00000000006c823, 0xe80000800005161a],
    [0x0000000000000003, 0xe00000000006c823, 0xf400004000028b0d],
    [0xffffffffffffffff, 0xf400004000028b0d, 0x0a00001ffffde175],
    [0x0000000000000001, 0xf400004000028b0d, 0xff00003000003641],
    [0x0000000000000001, 0xff00003000003641, 0x057ffff7fffed59a],
    [0x0000000000000003, 0xff00003000003641, 0x02bffffbffff6acd],
    [0xffffffffffffffff, 0x02bffffbffff6acd, 0x01dfffe5ffff9a46],
    [0x0000000000000001, 0x02bffffbffff6acd, 0x00effff2ffffcd23],
    [0x0000000000000001, 0x00effff2ffffcd23, 0xff17fffb8000312b],
    [0x0000000000000001, 0xff17fffb8000312b, 0xff14000440003204],
    [0x0000000000000003, 0xff17fffb8000312b, 0xff8a000220001902],
    [0x0000000000000005, 0xff17fffb8000312b, 0xffc5000110000c81],
    [0xfffffffffffffffd, 0xffc5000110000c81, 0x00568002c7ffedab],
    [0xffffffffffffffff, 0xffc5000110000c81, 0x000dc001ebfffd16],
    [0x0000000000000001, 0xffc5000110000c81, 0x0006e000f5fffe8b],
    [0x0000000000000001, 0x0006e000f5fffe8b, 0x0020effff2fff905],
    [0x0000000000000001, 0x0020effff2fff905, 0x000d07ff7e7ffd3d],
    [0x0000000000000001, 0x000d07ff7e7ffd3d, 0xfff60bffc5c0021c],
];

#[test]
fn divstep_trace() {
    let [two_delta, f0, g0, ..] = DIVSTEP59_VECTORS[0];
    let packed = [two_delta, (f0 & 0xfffff) | 0xfffffe0000000000, (g0 & 0xfffff) | 0xc000000000000000];
    assert_eq!(packed, DIVSTEP_TRACE[0]);
    for (i, pair) in DIVSTEP_TRACE.windows(2).enumerate() {
        let [two_delta, pf, pg] = pair[0];
        assert_eq!(divstep_once(two_delta, pf, pg), pair[1], "step {}", i + 1);
    }
}

// The inversion's checks, over this backend's blocks.

#[test]
fn sign_mag_known_answers() {
    crate::inversion::tests::sign_mag_known_answers::<Backend>();
}

#[test]
fn divstep59_known_answers() {
    crate::inversion::tests::divstep59_known_answers::<Backend>();
}

#[test]
fn fg_row_known_answers() {
    crate::inversion::tests::fg_row_known_answers::<Backend>();
}

#[test]
fn de_row_known_answers() {
    crate::inversion::tests::de_row_known_answers::<Backend>();
}

#[test]
fn amontred_known_answers() {
    crate::inversion::tests::amontred_known_answers::<Backend>();
}

#[test]
fn cond_sub_known_answers() {
    crate::inversion::tests::cond_sub_known_answers::<Backend>();
}

#[test]
fn invert_known_answers() {
    crate::inversion::tests::invert_known_answers::<Backend>();
}

#[test]
fn invert_small_and_near_modulus() {
    crate::inversion::tests::invert_small_and_near_modulus::<Backend>();
}

#[test]
fn invert_random() {
    crate::inversion::tests::invert_random::<Backend>();
}


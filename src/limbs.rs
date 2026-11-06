//! The limb type of the assembly backends and of the inversion, and the operand checks of their
//! debug assertions.

use core::hint::black_box;

/// Runs `$body` once for each index in the list, with `$i` bound to it: a loop of fixed length,
/// unrolled, so that its translation to Lean is straight-line code.
macro_rules! unroll_limbs {
    ($i:ident in [$($n:literal),* $(,)?] $body:block) => {
        $({
            let $i: usize = $n;
            $body
        })*
    };
}
pub(crate) use unroll_limbs;

/// Four little-endian 64-bit limbs, least significant first: a field element
/// (in Montgomery form, or canonical after `from_mont`) or a modulus.
pub type Limbs = [u64; 4];

/// `1` when `value < modulus` as little-endian 256-bit integers, else `0`: the borrow out of
/// the four-limb subtraction `value - modulus`, computed limb by limb. The debug assertions
/// must leave the routines' timing as it is, so the check has no data-dependent branch, and
/// each borrow passes through [`black_box`], the barrier that `subtle` uses, which keeps the
/// optimizer from turning the chain or its callers' combinations back into branches: an
/// opaque word can only be combined arithmetically.
#[inline(always)]
pub(crate) fn is_canonical_word(value: &Limbs, modulus: &Limbs) -> u64 {
    let mut borrow = 0;
    unroll_limbs!(i in [0, 1, 2, 3] {
        let (difference, underflow) = value[i].overflowing_sub(modulus[i]);
        let (_, borrow_underflow) = difference.overflowing_sub(borrow);
        borrow = black_box(u64::from(underflow | borrow_underflow));
    });
    borrow
}

/// Whether `value < modulus` as little-endian 256-bit integers; see [`is_canonical_word`].
#[inline(always)]
pub(crate) fn is_canonical(value: &Limbs, modulus: &Limbs) -> bool {
    is_canonical_word(value, modulus) == 1
}

#[cfg(test)]
mod tests;

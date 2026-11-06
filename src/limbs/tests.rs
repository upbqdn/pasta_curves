use super::*;

/// The borrow chain of `is_canonical` decides `value < modulus` at the limb boundaries.
#[test]
fn is_canonical_borrow_chain() {
    let m = [5, 0, 0, 7];
    assert!(is_canonical(&[4, 0, 0, 7], &m));
    assert!(!is_canonical(&m, &m));
    assert!(!is_canonical(&[6, 0, 0, 7], &m));
    // A borrow out of the low limbs is absorbed by a larger top limb, and forced by a smaller one.
    assert!(is_canonical(&[u64::MAX, u64::MAX, u64::MAX, 6], &m));
    assert!(!is_canonical(&[0, 0, 0, 8], &m));
    // A middle limb decides when the top limbs agree.
    assert!(!is_canonical(&[0, 1, 0, 7], &m));
    assert!(is_canonical(&[u64::MAX, 0, 0, 6], &m));
    assert!(is_canonical(&[0, 0, 0, 0], &m));
}

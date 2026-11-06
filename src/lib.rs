//! Implementation of the Pallas / Vesta curve cycle.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(unknown_lints)]
#![allow(clippy::op_ref, clippy::same_item_push, clippy::upper_case_acronyms)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;

// The assembly backends, and the `if_asm_supported!` and `if_asm_unsupported!` macros that gate
// code on them; declared first so that the macros are in scope in the modules below.
#[macro_use]
mod asm;
pub use asm::BACKEND;
pub use asm::selected_backend;

// The limb type of the backends and the inversion, the checks of their debug assertions, and the
// `unroll_limbs!` macro, which its users import by path.
mod limbs;

// The constant-time inversion, over the AArch64 assembly blocks or the portable ones.
mod inversion;

// The fields' constants and known answers for the tests of the backends and the inversion.
#[cfg(test)]
mod test_fields;

#[macro_use]
mod macros;
mod curves;
mod fields;

pub mod arithmetic;
#[cfg(feature = "deferred")]
#[cfg_attr(docsrs, doc(cfg(feature = "deferred")))]
pub mod deferred;
pub mod pallas;
pub mod vesta;

#[cfg(feature = "glv")]
#[cfg_attr(docsrs, doc(cfg(feature = "glv")))]
pub mod glv;

#[cfg(feature = "alloc")]
mod hashtocurve;

#[cfg(feature = "serde")]
mod serde_impl;

pub use curves::*;
pub use fields::*;

pub extern crate group;

#[cfg(feature = "alloc")]
#[test]
fn test_endo_consistency() {
    use crate::arithmetic::CurveExt;
    use group::{Group, ff::WithSmallOrderMulGroup};

    let a = pallas::Point::generator();
    assert_eq!(a * pallas::Scalar::ZETA, a.endo());
    let a = vesta::Point::generator();
    assert_eq!(a * vesta::Scalar::ZETA, a.endo());
}

#[cfg(test)]
mod backend_tests;

// The point of these tests is that they compile: each macro is used in every position the backend's
// users need (an item, a `let` binding, and a block inside a function), so the assertions are
// trivially true by design.
#[cfg(feature = "asm")]
#[cfg(test)]
mod asm_gate_tests;

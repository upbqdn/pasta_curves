// The crate denies unsafe code by default; the assembly backend is the one place that allows
// it, and its safety argument is the operand contracts stated on each routine.
#![allow(unsafe_code)]

//! Assembly backends for the Pasta fields: AArch64, x86-64, and ARMv7E-M.
//!
//! # Availability
//!
//! With the `asm` feature enabled, the ARMv7E-M backend is intended for
//! `thumbv7em-none-eabihf`, where it uses rolling UMAAL multiplication and
//! squaring and dedicated Montgomery conversion; field addition and subtraction
//! remain portable. The AArch64 backend provides all inversion blocks in
//! assembly; the ARM backend uses its `divstep59`, `amontred`, and `cond_sub`
//! blocks alongside portable Rust for the other three.
//!
//! Some ARM targets match the gate but lack the required Thumb-2 or DSP
//! instructions. Pass `--cfg pasta_curves_noasm` through `RUSTFLAGS` or
//! `rustflags` in `.cargo/config.toml` to select portable arithmetic on those
//! targets. On x86-64, the x32 ABI is excluded because some routines read limbs
//! through pointers, and Apple targets reserve `rbp`, leaving too few registers
//! for the squaring blocks. Outside SGX, x86-64 field operations detect BMI2
//! and ADX at run time and fall back to portable arithmetic when either is
//! missing. Enabling both target features at compile time removes detection;
//! SGX requires both because it cannot use CPUID.
//!
//! Code that uses the backend declares its items under `if_asm_supported!`
//! and selects operations through `if_asm!`, which includes runtime dispatch.
//! The complementary `if_asm_unsupported!` selects items where no backend is
//! compiled in. [`BACKEND`] names the compiled-in backend; [`selected_backend`]
//! reports the runtime selection.
//!
//! Nothing is assembled at build time: the blocks are compiled by the Rust
//! toolchain, so no C toolchain is needed. On x86-64, `asm` also enables
//! `cpufeatures` for AVX2 helper dispatch. Hosted targets detect AVX2 at runtime;
//! SGX, freestanding and UEFI targets require compile-time AVX2 guarantees.
//!
//! # Timing
//!
//! The blocks have no data-dependent branch or memory access. Runtime dispatch
//! depends only on the CPU, not field operands, so their timing should not
//! depend on operands unless the Rust toolchain or platform introduces an
//! unexpected obstacle. A debug build also runs the assertions' checks,
//! and debug mode carries no constant-time guarantee. The checks are written
//! without data-dependent branches, and pass their words through
//! `core::hint::black_box`, as `subtle` does. An inspection of the output of
//! one toolchain (AArch64, Rust 1.96.1) found only the assertions' own branches
//! left, but that is best effort, which the compiler owes nothing to.
//!
//! # Provenance
//!
//! The AArch64 and x86-64 Montgomery routines are transcriptions of
//! Supranational's [Semolina] v0.1.4. The ARMv7E-M implementation is original.
//! See `src/asm/README.md` for the history of the transcriptions.
//!
//! [Semolina]: https://github.com/supranational/semolina

/// Declares items, a local binding, or a block only where this module has a backend.
///
/// The expansion carries the target, presence of the `asm` feature, and absence of
/// `--cfg pasta_curves_noasm`, so code that uses the backend does not repeat the backend
/// condition. Write an extra pair of braces to cfg-gate arbitrary statements in a block:
/// `if_asm_supported! {{ ... }}`. A binding used after the macro must instead be
/// written without the extra braces: `if_asm_supported! { let value = expression; }`.
macro_rules! if_asm_supported {
    (@cfg $($tokens:tt)+) => {
        // The x86-64 multiplication family addresses limbs through pointers, so the backend needs
        // 64-bit pointers; and Apple x86-64 targets reserve `rbp`, and so have fewer available
        // registers than the squaring blocks need.
        #[cfg(all(
            feature = "asm",
            not(pasta_curves_noasm),
            any(
                target_arch = "aarch64",
                all(
                    target_arch = "x86_64",
                    target_pointer_width = "64",
                    not(target_vendor = "apple"),
                ),
                all(
                    target_arch = "arm",
                    any(
                        target_os = "none",
                        all(test, pasta_curves_armv7em_test, target_os = "linux"),
                    ),
                    target_abi = "eabihf",
                    target_pointer_width = "32",
                    target_endian = "little",
                )
            )
        ))]
        $($tokens)+
    };
    ({ $($body:tt)* }) => {
        if_asm_supported! { @cfg { $($body)* } }
    };
    (let $name:ident $(: $ty:ty)? = $value:expr;) => {
        if_asm_supported! { @cfg let $name $(: $ty)? = $value; }
    };
    ($($item:item)*) => { $(
        if_asm_supported! { @cfg $item }
    )* };
}

/// Declares items, a local binding, or a block where this module has no backend.
/// This is the complement of `if_asm_supported!` for portable fallbacks.
macro_rules! if_asm_unsupported {
    (@cfg $($tokens:tt)+) => {
        #[cfg(not(all(
            feature = "asm",
            not(pasta_curves_noasm),
            any(
                target_arch = "aarch64",
                all(
                    target_arch = "x86_64",
                    target_pointer_width = "64",
                    not(target_vendor = "apple"),
                ),
                all(
                    target_arch = "arm",
                    any(
                        target_os = "none",
                        all(test, pasta_curves_armv7em_test, target_os = "linux"),
                    ),
                    target_abi = "eabihf",
                    target_pointer_width = "32",
                    target_endian = "little",
                )
            )
        )))]
        $($tokens)+
    };
    ({ $($body:tt)* }) => {
        if_asm_unsupported! { @cfg { $($body)* } }
    };
    (let $name:ident $(: $ty:ty)? = $value:expr;) => {
        if_asm_unsupported! { @cfg let $name $(: $ty)? = $value; }
    };
    ($($item:item)*) => { $(
        if_asm_unsupported! { @cfg $item }
    )* };
}

if_asm_supported! {
    /// The assembly backend compiled into this build: `"aarch64"`, `"x86-64"`, or `"armv7em"`
    /// where the crate has one, and `"portable"` where it does not. Intended for diagnostics only,
    /// such as logs and benchmark labels; it is not a stable interface.
    pub const BACKEND: &str = if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "arm") {
        "armv7em"
    } else {
        "x86-64"
    };

    /// Whether field operations can use the compiled-in backend on this CPU.
    #[inline(always)]
    pub(crate) fn available() -> bool {
        #[cfg(test)]
        if tests::FORCE_PORTABLE.get() {
            return false;
        }
        #[cfg(all(
            target_arch = "x86_64",
            not(target_env = "sgx"),
            not(all(target_feature = "bmi2", target_feature = "adx"))
        ))]
        { dispatch::available() }
        #[cfg(any(
            target_arch = "aarch64",
            target_arch = "arm",
            all(target_feature = "bmi2", target_feature = "adx")
        ))]
        { true }
        #[cfg(all(
            target_arch = "x86_64",
            target_env = "sgx",
            not(all(target_feature = "bmi2", target_feature = "adx"))
        ))]
        { false }
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_env = "sgx"),
        not(all(target_feature = "bmi2", target_feature = "adx"))
    ))]
    mod dispatch;

    #[cfg(any(target_arch = "aarch64", doc))]
    pub(crate) mod aarch64;

    #[cfg(any(target_arch = "x86_64", doc))]
    mod x86_64;

    #[cfg(any(target_arch = "arm", doc))]
    pub(crate) mod armv7em;

    // Tests use std for thread-local dispatch overrides and to catch debug assertions.
    #[cfg(test)]
    extern crate std;

    #[cfg(test)]
    mod tests;

    mod entry;
    pub use entry::*;
}

if_asm_unsupported! {
    /// The assembly backend compiled into this build: `"aarch64"`, `"x86-64"`, or `"armv7em"`
    /// where the crate has one, and `"portable"` where it does not. Intended for diagnostics only,
    /// such as logs and benchmark labels; it is not a stable interface.
    pub const BACKEND: &str = "portable";
}

/// Returns the arithmetic backend selected for field operations on this CPU.
///
/// The result is `"aarch64"`, `"armv7em"`, `"x86-64"`, or `"portable"`. Builds
/// without assembly return `"portable"`. Selection uses cached CPU detection
/// on x86-64, except when BMI2 and ADX are enabled at compile time. In SGX,
/// assembly requires both target features. Intended for diagnostics only;
/// not a stable interface.
pub fn selected_backend() -> &'static str {
    if_asm_supported! {{ if available() { BACKEND } else { "portable" } }}
    if_asm_unsupported! {{ "portable" }}
}

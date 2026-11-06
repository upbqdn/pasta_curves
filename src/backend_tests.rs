use super::*;

#[test]
fn backend_name() {
    // The backend's condition, restated: the feature, the target, and the absence of the opt-out
    // flag.
    let supported = cfg!(all(
        feature = "asm",
        not(pasta_curves_noasm),
        any(
            target_arch = "aarch64",
            all(
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(target_vendor = "apple")
            )
        )
    ));
    let expected = if !supported {
        "portable"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "x86-64"
    };
    assert_eq!(BACKEND, expected);
    if !supported {
        assert_eq!(selected_backend(), "portable");
    }
}

if_asm_supported! {
    fn backend_name() -> &'static str { crate::BACKEND }
}
if_asm_unsupported! {
    fn backend_name() -> &'static str { "portable" }
}

fn selected_backend() -> bool {
    if_asm_supported! {{ crate::BACKEND != "portable" }}
    if_asm_unsupported! {{ false }}
}

#[test]
fn cfg_gated_local_bindings_and_blocks() {
    if_asm_supported! { let backend: &str = backend_name(); }
    if_asm_unsupported! { let backend: &str = backend_name(); }
    if_asm_supported! {{ assert_ne!(backend, "portable"); }}
    if_asm_unsupported! {{ assert_eq!(backend, "portable"); }}
    assert_eq!(backend == "portable", !selected_backend());
}

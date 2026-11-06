use super::Backend;
use crate::inversion::tests as checks;

#[test]
fn sign_mag_known_answers() {
    checks::sign_mag_known_answers::<Backend>();
}

#[test]
fn divstep59_known_answers() {
    checks::divstep59_known_answers::<Backend>();
}

#[test]
fn fg_row_known_answers() {
    checks::fg_row_known_answers::<Backend>();
}

#[test]
fn de_row_known_answers() {
    checks::de_row_known_answers::<Backend>();
}

#[test]
fn amontred_known_answers() {
    checks::amontred_known_answers::<Backend>();
}

#[test]
fn cond_sub_known_answers() {
    checks::cond_sub_known_answers::<Backend>();
}

#[test]
fn invert_known_answers() {
    checks::invert_known_answers::<Backend>();
}

#[test]
fn invert_small_and_near_modulus() {
    checks::invert_small_and_near_modulus::<Backend>();
}

#[test]
fn invert_random() {
    checks::invert_random::<Backend>();
}

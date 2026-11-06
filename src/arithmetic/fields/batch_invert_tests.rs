use std::vec::Vec;

use group::ff::{BatchInvert, Field, FromUniformBytes};
use proptest::prelude::*;

use super::{VartimeBatchInvert, VartimeField};

fn check_batch<F: VartimeField + core::fmt::Debug + Eq>(inputs: &[F]) {
    let expected: Vec<F> = inputs
        .iter()
        .map(|x| Option::<F>::from(x.invert()).unwrap_or(F::ZERO))
        .collect();
    let product = inputs
        .iter()
        .filter(|x| !x.is_zero_vartime())
        .fold(F::ONE, |acc, x| acc * x);
    let expected_product_inverse = product.invert().unwrap();

    let mut vartime = inputs.to_vec();
    let mut constant_time = inputs.to_vec();
    let vartime_product_inverse = vartime.iter_mut().batch_invert_vartime();
    let constant_time_product_inverse = constant_time.iter_mut().batch_invert();

    assert_eq!(vartime, expected);
    assert_eq!(vartime, constant_time);
    assert_eq!(vartime_product_inverse, expected_product_inverse);
    assert_eq!(vartime_product_inverse, constant_time_product_inverse);
}

fn check_both_fields(values: &[u64]) {
    let fp: Vec<_> = values.iter().copied().map(crate::Fp::from).collect();
    let fq: Vec<_> = values.iter().copied().map(crate::Fq::from).collect();
    check_batch(&fp);
    check_batch(&fq);
}

#[test]
fn batch_invert_edge_cases() {
    for values in [
        &[][..],
        &[0][..],
        &[1][..],
        &[7][..],
        &[0, 0, 0][..],
        &[0, 1, 0, 2, 0, 3, 0][..],
        &[2, 3, 5, 7][..],
        &[9, 9, 0, 9][..],
    ] {
        check_both_fields(values);
    }
}

fn from_limbs<F: FromUniformBytes<64>>(limbs: [u64; 4]) -> F {
    let mut bytes = [0u8; 64];
    for (chunk, limb) in bytes.chunks_exact_mut(8).zip(limbs) {
        chunk.copy_from_slice(&limb.to_le_bytes());
    }
    F::from_uniform_bytes(&bytes)
}

proptest! {
    #[test]
    fn batch_invert_matches_individual_inversion(
        values in proptest::collection::vec(
            proptest::option::of(proptest::array::uniform4(any::<u64>())), 0..32
        )
    ) {
        let fp: Vec<_> = values.iter().map(|value| match value {
            Some(limbs) => from_limbs::<crate::Fp>(*limbs),
            None => crate::Fp::ZERO,
        }).collect();
        let fq: Vec<_> = values.iter().map(|value| match value {
            Some(limbs) => from_limbs::<crate::Fq>(*limbs),
            None => crate::Fq::ZERO,
        }).collect();
        check_batch(&fp);
        check_batch(&fq);
    }
}

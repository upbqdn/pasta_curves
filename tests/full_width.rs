#![cfg(feature = "glv")]

use ff::Field;
use group::CurveAffine;
use pasta_curves::{arithmetic::CurveExt, pallas, vesta};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

fn check<C: CurveExt>() {
    let mut rng = XorShiftRng::from_seed([47; 16]);
    let bases = (0..1025)
        .map(|_| C::random(&mut rng).to_affine())
        .collect::<Vec<_>>();
    let scalars = (0..1025)
        .map(|_| C::ScalarExt::random(&mut rng))
        .collect::<Vec<_>>();
    let run = || {
        let mut accepted = 0;
        for count in [0, 1, 16, 24, 25, 65, 66, 67, 255, 256, 257, 1024] {
            for mode in 0..4 {
                let mut bases = bases[..count].to_vec();
                let mut scalars = scalars[..count].to_vec();
                match mode {
                    1 => scalars.fill(C::ScalarExt::ZERO),
                    2 => bases.fill(C::AffineExt::identity()),
                    3 => {
                        for (i, scalar) in scalars.iter_mut().enumerate() {
                            *scalar = if i % 2 == 0 {
                                -C::ScalarExt::ONE
                            } else {
                                C::ScalarExt::ONE
                            };
                            if i % 3 == 0 {
                                bases[i] = -bases[0];
                            }
                        }
                    }
                    _ => {}
                }
                let expected = C::try_multiexp_vartime(&scalars, &bases).unwrap_or_else(|| {
                    scalars
                        .iter()
                        .zip(&bases)
                        .fold(C::identity(), |acc, (scalar, base)| acc + *base * scalar)
                });
                if let Some(actual) =
                    C::try_multiexp_full_width_is_identity_vartime(&scalars, &bases)
                {
                    accepted += 1;
                    assert_eq!(
                        actual,
                        bool::from(expected.is_identity()),
                        "size={count} mode={mode}"
                    );
                }
                if count > 0 && mode == 0 {
                    bases.push((-expected).to_affine());
                    scalars.push(C::ScalarExt::ONE);
                    if let Some(actual) =
                        C::try_multiexp_full_width_is_identity_vartime(&scalars, &bases)
                    {
                        assert!(actual, "size={count} cancellation");
                    }
                }
            }
        }
        let mut cancelling = vec![C::ScalarExt::ZERO; 1024];
        let a = C::ScalarExt::from(73u64);
        cancelling[0] = a;
        cancelling[1] = -a;
        assert_eq!(
            C::try_multiexp_full_width_is_identity_vartime(&cancelling, &[bases[0]; 1024]),
            Some(true),
            "same-base a·P + (r−a)·P must cancel",
        );
        assert!(accepted > 0, "the full-width backend must be exercised");
        assert!(C::try_multiexp_full_width_is_identity_vartime(&scalars[..1], &[]).is_none());
    };
    #[cfg(feature = "multicore")]
    for workers in [1, 2, 32] {
        maybe_rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test environment supports worker pools")
            .install(run);
    }
    #[cfg(not(feature = "multicore"))]
    run();
}

#[test]
fn pallas_full_width_matches_ordinary_msm() {
    check::<pallas::Point>();
}

#[test]
fn vesta_full_width_matches_ordinary_msm() {
    check::<vesta::Point>();
}

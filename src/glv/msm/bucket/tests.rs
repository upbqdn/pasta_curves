use ff::Field;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

use super::*;
use crate::glv::booth::checked_signed_magnitudes;
use crate::glv::decompose;
use crate::{pallas, vesta};

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "reconstructed point is on curve")]
fn reconstructed_pallas_points_must_be_on_curve() {
    affine_from_xy_unchecked::<pallas::Point>(crate::Fp::ONE, crate::Fp::ONE);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "reconstructed point is on curve")]
fn reconstructed_vesta_points_must_be_on_curve() {
    affine_from_xy_unchecked::<vesta::Point>(crate::Fq::ONE, crate::Fq::ONE);
}

/// Inputs for the window tests: zero and unit scalars among varied ones, and
/// bases that are the identity, equal or opposite.
fn exceptional_inputs<C>(terms: usize) -> (Vec<C::ScalarExt>, Vec<C::AffineExt>)
where
    C: GlvParams,
{
    let scalars = (0..terms)
        .map(|i| {
            let i = u64::try_from(i).unwrap();
            match i % 8 {
                0 => C::ScalarExt::ZERO,
                1 | 2 => C::ScalarExt::ONE,
                3 => -C::ScalarExt::ONE,
                _ => (C::ScalarExt::from(i + 0x9e3779b97f4a7c15).square() + C::ScalarExt::from(i))
                    .square(),
            }
        })
        .collect::<Vec<_>>();
    // Equal and opposite bases, as well as identities, force exceptional
    // reductions independently of which backend the cost planner would pick.
    let generator = C::generator();
    let bases = (0..terms)
        .map(|i| {
            match i % 4 {
                0 => C::identity(),
                1 => generator,
                2 => -generator,
                _ => generator.double(),
            }
            .to_affine_vartime()
        })
        .collect::<Vec<_>>();
    (scalars, bases)
}

fn forced_window<C>(terms: usize, width: usize)
where
    C: GlvParams,
    AffineBase<C>: CachedField,
{
    let (scalars, bases) = exceptional_inputs::<C>(terms);
    let expected = scalars
        .iter()
        .zip(&bases)
        .fold(C::identity(), |sum, (scalar, base)| sum + *base * scalar);
    let components = scalars
        .iter()
        .map(decompose::<C>)
        .map(checked_signed_magnitudes)
        .collect::<Option<Vec<_>>>()
        .unwrap();
    let bases = multiexp_bases::<C>(&bases);
    assert_eq!(
        multiexp_serial::<C>(&components, &bases, width),
        Some(expected),
        "{terms} terms at width {width}"
    );
    #[cfg(feature = "multicore")]
    assert_eq!(
        multiexp_parallel::<C>(&components, &bases, width),
        Some(expected),
        "{terms} terms at width {width}, parallel"
    );
}

#[test]
fn all_window_widths() {
    for width in [1, 3, 6, 8, 9, 10] {
        forced_window::<pallas::Point>(64, width);
        forced_window::<vesta::Point>(64, width);
    }
}

/// Large clean levels take the direct compaction path; repeated/opposite bases
/// force the complete-formula fallback at different levels.
fn randomized_bucket_msm_matches_naive<C>()
where
    C: GlvParams,
    AffineBase<C>: CachedField,
{
    let mut rng = XorShiftRng::seed_from_u64(0x6d736d);
    for (terms, width) in [
        (0, 3),
        (1, 3),
        (31, 6),
        (32, 6),
        (33, 6),
        (96, 3),
        (96, 7),
        (320, 9),
    ] {
        let mut scalars = Vec::with_capacity(terms);
        let mut bases = Vec::with_capacity(terms);
        for i in 0..terms {
            scalars.push(match i % 11 {
                0 => C::ScalarExt::ZERO,
                1..=3 => C::ScalarExt::ONE,
                _ => C::ScalarExt::random(&mut rng),
            });
            let base = match i % 11 {
                0 => C::identity(),
                1 | 2 => C::generator(),
                3 => -C::generator(),
                _ => C::random(&mut rng),
            };
            bases.push(base.to_affine_vartime());
        }
        let expected = scalars
            .iter()
            .zip(&bases)
            .fold(C::identity(), |sum, (scalar, base)| sum + *base * scalar);
        let components = scalars
            .iter()
            .map(decompose::<C>)
            .map(checked_signed_magnitudes)
            .collect::<Option<Vec<_>>>()
            .unwrap();
        let prepared = multiexp_bases::<C>(&bases);
        assert_eq!(
            multiexp_serial::<C>(&components, &prepared, width),
            Some(expected),
            "{terms} terms at width {width}"
        );
    }
}

#[test]
fn randomized_bucket_msm_matches_naive_on_both_curves() {
    randomized_bucket_msm_matches_naive::<pallas::Point>();
    randomized_bucket_msm_matches_naive::<vesta::Point>();
}

#[cfg(feature = "multicore")]
#[test]
fn window_jobs_match_serial_exact_coordinates() {
    fn check<C>()
    where
        C: GlvParams,
        AffineBase<C>: CachedField,
    {
        let pools = [1, 2, 4, 32].map(|workers| {
            maybe_rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .expect("a custom worker pool is available in the test environment")
        });
        for terms in [0, 1, 17, 65, 257] {
            let (scalars, bases) = exceptional_inputs::<C>(terms);
            let components = scalars
                .iter()
                .map(decompose::<C>)
                .map(checked_signed_magnitudes)
                .collect::<Option<Vec<_>>>()
                .unwrap();
            let bases = multiexp_bases::<C>(&bases);
            for width in [1, 3, 6, 9, 10] {
                let serial = multiexp_serial::<C>(&components, &bases, width)
                    .map(|sum| sum.jacobian_coordinates());
                for (workers, pool) in [1, 2, 4, 32].into_iter().zip(&pools) {
                    let parallel = pool.install(|| {
                        multiexp_parallel::<C>(&components, &bases, width)
                            .map(|sum| sum.jacobian_coordinates())
                    });
                    assert_eq!(
                        parallel, serial,
                        "{terms} terms, width {width}, {workers} workers"
                    );
                }
            }
        }
    }
    check::<pallas::Point>();
    check::<vesta::Point>();
}

/// Reducing a window in blocks, with each block's buckets carried into the
/// next, gives the same buckets as one block. Equal and opposite bases make
/// carried points double and cancel.
fn blocks_match_single_block<C>()
where
    C: GlvParams,
{
    let (scalars, bases) = exceptional_inputs::<C>(64);
    let components = scalars
        .iter()
        .map(decompose::<C>)
        .map(checked_signed_magnitudes)
        .collect::<Option<Vec<_>>>()
        .unwrap();
    let bases = multiexp_bases::<C>(&bases);
    let as_points = |buckets: Vec<Option<AffinePoint<AffineBase<C>>>>| {
        buckets
            .into_iter()
            .map(bucket_point::<C>)
            .collect::<Vec<_>>()
    };

    for width in [1, 3, 6] {
        for window in 0..window_count(width) {
            let whole = as_points(
                WindowReducer::default()
                    .fill_window_in_blocks(&components, &bases, width, window, components.len())
                    .unwrap()
                    .to_vec(),
            );
            for block_terms in [1, 2, 3, 5, 16, 63] {
                let blocked = as_points(
                    WindowReducer::default()
                        .fill_window_in_blocks(&components, &bases, width, window, block_terms)
                        .unwrap()
                        .to_vec(),
                );
                assert_eq!(
                    blocked, whole,
                    "width {width}, window {window}, blocks of {block_terms} terms"
                );
            }
        }
    }
}

#[test]
fn window_blocks() {
    blocks_match_single_block::<pallas::Point>();
    blocks_match_single_block::<vesta::Point>();
}

#[test]
fn whole_window_above_previous_block_limit_matches_blocks() {
    let (scalars, bases) = exceptional_inputs::<vesta::Point>((1 << 14) + 1);
    let components = scalars
        .iter()
        .map(decompose::<vesta::Point>)
        .map(checked_signed_magnitudes)
        .collect::<Option<Vec<_>>>()
        .unwrap();
    let bases = multiexp_bases::<vesta::Point>(&bases);
    let as_points = |buckets: &[Option<AffinePoint<crate::Fq>>]| {
        buckets
            .iter()
            .copied()
            .map(bucket_point::<vesta::Point>)
            .collect::<Vec<_>>()
    };
    let width = 9;
    let whole = as_points(
        WindowReducer::default()
            .fill_window(&components, &bases, width, 0)
            .unwrap(),
    );
    let blocked = as_points(
        WindowReducer::default()
            .fill_window_in_blocks(&components, &bases, width, 0, 1 << 14)
            .unwrap(),
    );
    assert_eq!(whole, blocked);
}

#[test]
fn cached_scratch_alternating_curves_sizes_and_widths() {
    for (terms, width) in [
        (300, 7),
        (0, 1),
        (17, 3),
        (300, 4),
        (64, 9),
        (1, 2),
        (5, 10),
        (257, 6),
    ] {
        forced_window::<pallas::Point>(terms, width);
        forced_window::<vesta::Point>(terms, width);
    }
}

#[cfg(feature = "multicore")]
#[test]
fn nested_parallel_scratch_is_independent() {
    // One worker guarantees nested Rayon jobs re-enter the checkout thread.
    for workers in [1, 2] {
        let pool = maybe_rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            let mut outer = Checkout::<crate::Fp>::take();
            outer.counts.clear();
            outer.counts.extend([123, 456]);
            let outer_pointer = outer.counts.as_ptr().addr();
            maybe_rayon::join(
                || {
                    let mut nested = Checkout::<crate::Fp>::take();
                    assert_ne!(nested.counts.as_ptr().addr(), outer_pointer);
                    nested.counts.clear();
                    nested.counts.push(789);
                    forced_window::<pallas::Point>(65, 4);
                    assert_eq!(nested.counts, [789]);
                },
                || {
                    forced_window::<vesta::Point>(17, 3);
                    forced_window::<pallas::Point>(33, 6);
                },
            );
            assert_eq!(outer.counts, [123, 456]);
            drop(outer);
            forced_window::<pallas::Point>(5, 2);
            forced_window::<vesta::Point>(0, 8);
        });
    }
}

#[cfg(feature = "multicore")]
#[test]
fn tls_scratch_reuses_capacity_and_drops_oversized_buffers() {
    // Isolate this cache from other tests and the test runner's worker threads.
    std::thread::spawn(|| {
        let capacity = {
            let mut scratch = Checkout::<crate::Fp>::take();
            scratch.assignments.reserve(128);
            scratch.assignments.capacity()
        };
        {
            let mut scratch = Checkout::<crate::Fp>::take();
            assert_eq!(scratch.assignments.capacity(), capacity);
            scratch
                .assignments
                .reserve(cache::MAX_RETAINED_BYTES / core::mem::size_of::<WindowAssignment>() + 1);
            assert!(scratch.capacity_bytes() > cache::MAX_RETAINED_BYTES);
        }
        let scratch = Checkout::<crate::Fp>::take();
        assert_eq!(scratch.capacity_bytes(), 0);
    })
    .join()
    .unwrap();
}

#[cfg(feature = "multicore")]
#[test]
fn pool_preparation_matches_serial() {
    fn check<C>()
    where
        C: GlvParams,
        AffineBase<C>: CachedField,
    {
        let (scalars, bases) = exceptional_inputs::<C>(2_100);
        for workers in [1, 2, 8] {
            let pool = maybe_rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .expect("a custom worker pool is available in the test environment");
            pool.install(|| {
                let (components, prepared) =
                    super::super::multiexp_inputs::<C>(&scalars, &bases, workers)
                        .expect("GLV components fit in 127 bits");
                let serial = scalars
                    .iter()
                    .map(decompose::<C>)
                    .map(checked_signed_magnitudes)
                    .collect::<Option<Vec<_>>>()
                    .expect("GLV components fit in 127 bits");
                assert_eq!(components, serial);
                assert_eq!(
                    multiexp::<C>(&components, &prepared, 7, workers),
                    multiexp::<C>(&serial, &multiexp_bases::<C>(&bases), 7, workers)
                );
            });
        }
    }
    check::<pallas::Point>();
    check::<vesta::Point>();
}

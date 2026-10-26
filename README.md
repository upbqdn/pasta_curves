# `pasta_curves`

This crate provides an implementation of the Pasta elliptic curve constructions,
Pallas and Vesta. More details about the Pasta curves can be found
[in this blog post](https://electriccoin.co/blog/the-pasta-curves-for-halo-2-and-beyond/).

## RFC process

This crate follows the [zkcrypto RFC process](https://zkcrypto.github.io/rfcs/).
If you want to propose "substantial" changes to this crate, please
[create an RFC](https://github.com/zkcrypto/rfcs) for wider discussion.

## [Documentation](https://docs.rs/pasta_curves)

With `glv`, `CurveExt::try_multiexp_vartime` offers a variable-time MSM backend
for public scalars and bases. It uses a shared-doubling GLV-Strauss ladder for
small inputs and signed-Booth GLV buckets with batched affine additions for
large inputs. It returns `None` when its cost model prefers the caller's
generic MSM; callers must keep that fallback. Enable `multicore` to evaluate
independent windows through the Rayon pool; WebAssembly without atomics uses
the serial implementation. Neither feature changes ordinary constant-time
scalar multiplication.

The MSM backend reuses the existing split-wNAF tables;
the optional Eisenstein-orbit and prepared zero-check backends are not included.

## Minimum Supported Rust Version

Requires Rust **1.85** or higher.

Minimum supported Rust version can be changed in the future, but it will be done with a
minor version bump.

## Curve Descriptions

- Pallas: y<sup>2</sup> = x<sup>3</sup> + 5 over
  `GF(0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001)`.

- Vesta:  y<sup>2</sup> = x<sup>3</sup> + 5 over
  `GF(0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001)`.

The Pasta curves form a cycle with one another: the order of each curve is exactly the
base field of the other. This property is critical to the efficiency of recursive proof
systems. They are designed to be highly 2-adic, meaning that a large power-of-two
multiplicative subgroup exists in each field. This is important for the performance of
polynomial arithmetic over their scalar fields and is essential for protocols similar
to PLONK.

These curves can be reproducibly obtained
[using a curve search utility we’ve published](https://github.com/zcash/pasta).

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.

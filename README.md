# Bitshuffle

Transpose bits to improve data compression.

[![Build Status][actions-badge]][actions-url]

[actions-badge]: https://github.com/kalcutter/bitshuffle/actions/workflows/main.yml/badge.svg
[actions-url]: https://github.com/kalcutter/bitshuffle/actions/workflows/main.yml?query=branch%3Amain

## Overview

Bitshuffle is a lossless filter for improving the compression of typed data. It works by
transposing the bits of the data to reduce entropy.

This repository contains a highly optimized Bitshuffle implementation for modern processors.

For more information about Bitshuffle, refer to the classic implementation here:
<https://github.com/kiyo-masui/bitshuffle.git>.

## Scope

This library only implements the core Bitshuffle transpose operation (exposed as a C interface).

A Python interface is not provided.

## Features

* Implemented in C99.
* Optimized with SIMD instructions including SSE2, AVX2, and AVX-512.
* Runtime dispatch based on available CPU features (requires GNU IFUNC support).
* Does not allocate memory.
* Support for Clang, GCC, ICC, and MSVC.
* Tested on Linux, macOS, and Windows.

## Performance

The performance of this implementation is excellent.

Compared to [`bshuf_trans_bit_elem`][bshuf_trans_bit_elem] and
[`bshuf_untrans_bit_elem`][bshuf_untrans_bit_elem] from the classic implementation, this code
yields a typical speedup between **1.3x** and **10x** depending on the CPU architecture, function,
and arguments.

## API

The public interface is declared in the header file [bitshuffle.h](src/bitshuffle.h). Two public
functions are defined:

```c++
/* Transpose bits for compression.
 *
 * This function performs Bitshuffle transposition of a single block. The block
 * size in bytes is given by the product of `size` and `elem_size`.
 *
 * If required, the `scratch` argument must point to a buffer that the function
 * uses for scratch purposes. The size of this buffer is given by the block
 * size.
 *
 * On success, the function returns 0; otherwise, -1 is returned to indicate an
 * error. In case of error, the memory pointed to by `out` and `scratch` is left
 * unmodified.
 *
 * Errors
 * ------
 * The function returns -1 to indicate an error if:
 *
 * - The `scratch` argument is `NULL` and a scratch buffer is required for the
 *   specified element size.
 * - The `size` argument is not a multiple of 8.
 */
int bitshuf_encode_block(char* restrict out,
                         const char* restrict in,
                         char* restrict scratch,
                         size_t size,
                         size_t elem_size);
```

```c++
/* Untranspose bits after decompression.
 *
 * This function performs the inverse of `bitshuf_encode_block()`.
 *
 * If required, the `scratch` argument must point to a buffer that the function
 * uses for scratch purposes. The size of this buffer is given by the block
 * size.
 *
 * On success, the function returns 0; otherwise, -1 is returned to indicate an
 * error. In case of error, the memory pointed to by `out` and `scratch` is left
 * unmodified.
 *
 * Errors
 * ------
 * The function returns -1 to indicate an error if:
 *
 * - The `scratch` argument is `NULL` and a scratch buffer is required for the
 *   specified element size.
 * - The `size` argument is not a multiple of 8.
 */
int bitshuf_decode_block(char* restrict out,
                         const char* restrict in,
                         char* restrict scratch,
                         size_t size,
                         size_t elem_size);
```

These functions perform the same operations as [`bshuf_trans_bit_elem`][bshuf_trans_bit_elem] and
[`bshuf_untrans_bit_elem`][bshuf_untrans_bit_elem], respectively.

The header file is compatible with both C89 and C++.

[bshuf_trans_bit_elem]: https://github.com/kiyo-masui/bitshuffle/blob/b9a1546133959298c56eee686932dbb18ff80f7a/src/bitshuffle_internals.h#L50
[bshuf_untrans_bit_elem]: https://github.com/kiyo-masui/bitshuffle/blob/b9a1546133959298c56eee686932dbb18ff80f7a/src/bitshuffle_internals.h#L59

## Caveats

Only little-endian architectures are supported. Support for big-endian machines
is not planned.

## License

This repository is licensed under either of

* Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license
  ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

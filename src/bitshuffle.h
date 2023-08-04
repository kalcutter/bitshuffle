/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* Copyright (c) 2023 Kal Conley
 */
#ifndef BITSHUFFLE_H_
#define BITSHUFFLE_H_

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

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
 * Pointer arguments of this function have C99 `restrict` semantics. If the
 * `out`, `in`, or `scratch` buffers overlap, the behavior is undefined.
 *
 * Errors
 * ------
 * The function returns -1 to indicate an error if:
 *
 * - The `scratch` argument is `NULL` and a scratch buffer is required for the
 *   specified element size.
 * - The `size` argument is not a multiple of 8.
 */
int bitshuf_encode_block(char* out, const char* in, char* scratch, size_t size, size_t elem_size);

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
 * Pointer arguments of this function have C99 `restrict` semantics. If the
 * `out`, `in`, or `scratch` buffers overlap, the behavior is undefined.
 *
 * Errors
 * ------
 * The function returns -1 to indicate an error if:
 *
 * - The `scratch` argument is `NULL` and a scratch buffer is required for the
 *   specified element size.
 * - The `size` argument is not a multiple of 8.
 */
int bitshuf_decode_block(char* out, const char* in, char* scratch, size_t size, size_t elem_size);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* BITSHUFFLE_H_ */

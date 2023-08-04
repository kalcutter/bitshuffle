// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2023 Kal Conley
#include "bitshuffle.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(__INTEL_COMPILER)
#pragma warning(disable : 177) // entity-kind "entity" was declared but never referenced
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#if defined(_MSC_VER)
#pragma warning(disable : 4244) // conversion from 'type1' to 'type2', possible loss of data
#endif

#if defined(_MSC_VER)
#if defined(_M_IX86) && _M_IX86_FP == 2 || defined(_M_X64)
#ifndef __SSE2__
#define __SSE2__ 1
#endif
#endif
#endif

#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
        __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__ ||               \
    defined(__BIG_ENDIAN__)
#error big endian not supported
#endif

#ifndef BITSHUF_USE_IFUNC
#if defined(__INTEL_COMPILER) || defined(__clang__) && __clang_major__ < 8
#define BITSHUF_USE_IFUNC 0 // GFNI not supported by compiler.
#endif
#endif
#ifndef BITSHUF_USE_IFUNC
#if (__has_attribute(ifunc) && __has_attribute(target) && __has_builtin(__builtin_cpu_init) && \
     __has_builtin(__builtin_cpu_is) && __has_builtin(__builtin_cpu_supports)) ||              \
    (defined(__GNUC__) && __GNUC__ >= 8)
#define BITSHUF_USE_IFUNC 1
#else
#define BITSHUF_USE_IFUNC 0
#endif
#endif

#define STRINGIZE(x) #x

#if __has_attribute(target) && !defined(__INTEL_COMPILER)
#define ATTRIBUTE_TARGET(x) __attribute__((target(x)))
#else
#define ATTRIBUTE_TARGET(x)
#endif

#if __has_attribute(always_inline) || defined(__GNUC__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE inline
#endif

#if __has_attribute(noinline) || defined(__GNUC__)
#define NO_INLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define NO_INLINE __declspec(noinline)
#else
#define NO_INLINE
#endif

#if __has_attribute(no_sanitize_address)
#define NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
#define NO_SANITIZE_ADDRESS
#endif

#if __has_attribute(no_sanitize_memory)
#define NO_SANITIZE_MEMORY __attribute__((no_sanitize_memory))
#else
#define NO_SANITIZE_MEMORY
#endif

#if __has_attribute(no_sanitize_thread)
#define NO_SANITIZE_THREAD __attribute__((no_sanitize_thread))
#else
#define NO_SANITIZE_THREAD
#endif

#if __has_attribute(disable_sanitizer_instrumentation)
#define DISABLE_SANITIZER_INSTRUMENTATION __attribute__((disable_sanitizer_instrumentation))
#else
#define DISABLE_SANITIZER_INSTRUMENTATION
#endif

#if __has_attribute(fallthrough)
#define FALLTHROUGH __attribute__((fallthrough))
#else
#define FALLTHROUGH
#endif

#if __has_builtin(__builtin_expect) || defined(__GNUC__)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define UNLIKELY(x) (x)
#endif

// clang-format off
#define IMPLEMENT_IFUNC(NAME, PARAMS)                         \
    __attribute__((ifunc(STRINGIZE(NAME##_resolver))))        \
    static void NAME PARAMS;                                  \
                                                              \
    DISABLE_SANITIZER_INSTRUMENTATION                         \
    NO_SANITIZE_ADDRESS NO_SANITIZE_MEMORY NO_SANITIZE_THREAD \
    static void (*NAME##_resolver(void))PARAMS
// clang-format on

#define IMPLEMENT_LOAD_FUNCTION(NAME, TYPE)                \
    static ALWAYS_INLINE TYPE NAME(const void* mem_addr) { \
        TYPE a;                                            \
        memcpy(&a, mem_addr, sizeof(a));                   \
        return a;                                          \
    }

#define IMPLEMENT_STORE_FUNCTION(NAME, TYPE)                       \
    static ALWAYS_INLINE void NAME(void* mem_addr, const TYPE a) { \
        memcpy(mem_addr, &a, sizeof(a));                           \
    }

#if !defined(__SSE2__)
IMPLEMENT_LOAD_FUNCTION(LOAD_U64, uint64_t)
#endif
IMPLEMENT_STORE_FUNCTION(STORE_U8, uint8_t)
IMPLEMENT_STORE_FUNCTION(STORE_U64, uint64_t)

// Computes the transpose of an 8x8 bit matrix.
// Ref: "Hacker's Delight" 7-3 by Henry Warren.
static uint64_t transpose8(uint64_t x) {
    uint64_t t;
    t = (x ^ (x >> 7)) & 0x00aa00aa00aa00aa;
    x = (x ^ t ^ (t << 7));
    t = (x ^ (x >> 14)) & 0x0000cccc0000cccc;
    x = (x ^ t ^ (t << 14));
    t = (x ^ (x >> 28)) & 0x00000000f0f0f0f0;
    x = (x ^ t ^ (t << 28));
    return x;
}

#if !defined(__SSE2__)
NO_INLINE
static void bitshuf_trans_bit(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);
    size /= 8;

    for (size_t i = 0; i < size; i++) {
        const uint64_t a = LOAD_U64(&in[i * 8]);
        const uint64_t x = transpose8(a);
        STORE_U8(&out[0 * size + i], x);
        STORE_U8(&out[1 * size + i], x >> 8 * 1);
        STORE_U8(&out[2 * size + i], x >> 8 * 2);
        STORE_U8(&out[3 * size + i], x >> 8 * 3);
        STORE_U8(&out[4 * size + i], x >> 8 * 4);
        STORE_U8(&out[5 * size + i], x >> 8 * 5);
        STORE_U8(&out[6 * size + i], x >> 8 * 6);
        STORE_U8(&out[7 * size + i], x >> 8 * 7);
    }
}

NO_INLINE
static void bitshuf_trans_byte(char* restrict out,
                               const char* restrict in,
                               size_t size,
                               size_t elem_size) {
    assert(size % 8 == 0);

    for (size_t i = 0; i < size; i += 8) {
        for (size_t j = 0; j < elem_size; j++) {
            for (size_t k = 0; k < 8; k++) {
                out[j * size + (i + k)] = in[(i + k) * elem_size + j];
            }
        }
    }
}

NO_INLINE
static void bitshuf_trans_byte_2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);
    size = size / 8 * 8;

    for (size_t i = 0; i < size; i++) {
        out[0 * size + i] = in[i * 2 + 0];
        out[1 * size + i] = in[i * 2 + 1];
    }
}

NO_INLINE
static void bitshuf_trans_byte_4(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);
    size = size / 8 * 8;

    for (size_t i = 0; i < size; i++) {
        out[0 * size + i] = in[i * 4 + 0];
        out[1 * size + i] = in[i * 4 + 1];
        out[2 * size + i] = in[i * 4 + 2];
        out[3 * size + i] = in[i * 4 + 3];
    }
}

static void bitshuf_trans_byte_8(char* restrict out, const char* restrict in, size_t size) {
    bitshuf_trans_byte(out, in, size, 8);
}
#endif

static void bitshuf_untrans_bit_tail(char* restrict out,
                                     const char* restrict in,
                                     size_t size,
                                     size_t index) {
    assert(size % 8 == 0);
    size /= 8;

    for (size_t i = index; i < size; i++) {
        const uint64_t a = (uint64_t)(uint8_t)in[0 * size + i] |
                           (uint64_t)(uint8_t)in[1 * size + i] << 8 * 1 |
                           (uint64_t)(uint8_t)in[2 * size + i] << 8 * 2 |
                           (uint64_t)(uint8_t)in[3 * size + i] << 8 * 3 |
                           (uint64_t)(uint8_t)in[4 * size + i] << 8 * 4 |
                           (uint64_t)(uint8_t)in[5 * size + i] << 8 * 5 |
                           (uint64_t)(uint8_t)in[6 * size + i] << 8 * 6 |
                           (uint64_t)(uint8_t)in[7 * size + i] << 8 * 7;
        STORE_U64(&out[i * 8], transpose8(a));
    }
}

#if !defined(__SSE2__)
NO_INLINE
static void bitshuf_untrans_bit(char* restrict out, const char* restrict in, size_t size) {
    bitshuf_untrans_bit_tail(out, in, size, 0);
}

NO_INLINE
static void bitshuf_untrans_byte(char* restrict out,
                                 const char* restrict in,
                                 size_t size,
                                 size_t elem_size) {
    assert(size % 8 == 0);

    for (size_t i = 0; i < size; i += 8) {
        for (size_t j = 0; j < elem_size; j++) {
            for (size_t k = 0; k < 8; k++) {
                out[(i + k) * elem_size + j] = in[j * size + (i + k)];
            }
        }
    }
}

NO_INLINE
static void bitshuf_untrans_byte_2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);
    size = size / 8 * 8;

    for (size_t i = 0; i < size; i++) {
        out[i * 2 + 0] = in[0 * size + i];
        out[i * 2 + 1] = in[1 * size + i];
    }
}

NO_INLINE
static void bitshuf_untrans_byte_4(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);
    size = size / 8 * 8;

    for (size_t i = 0; i < size; i++) {
        out[i * 4 + 0] = in[0 * size + i];
        out[i * 4 + 1] = in[1 * size + i];
        out[i * 4 + 2] = in[2 * size + i];
        out[i * 4 + 3] = in[3 * size + i];
    }
}

static void bitshuf_untrans_byte_8(char* restrict out, const char* restrict in, size_t size) {
    bitshuf_untrans_byte(out, in, size, 8);
}
#endif

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)

IMPLEMENT_LOAD_FUNCTION(LOAD_I64, int64_t)
IMPLEMENT_STORE_FUNCTION(STORE_U16, uint16_t)
IMPLEMENT_STORE_FUNCTION(STORE_U32, uint32_t)

#define MM256_SETR_M128I(lo, hi) _mm256_inserti128_si256(_mm256_castsi128_si256(lo), (hi), 1)

#if defined(__clang__)
#define X(A)                   \
    ({                         \
        __asm__("" : "+x"(A)); \
        (A);                   \
    })
#else
#define X(A) (A)
#endif

#if defined(__AVX512VBMI__) && defined(__AVX512VL__) && defined(__GFNI__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("avx512vbmi,avx512vl,gfni")
static void bitshuf_trans_bit_avx512vbmi_gfni(char* restrict out,
                                              const char* restrict in,
                                              size_t size) {
    assert(size % 8 == 0);
    size /= 8;

    const __m512i BSWAP64 = _mm512_set_epi64(
        0x08090a0b0c0d0e0f, 0x0001020304050607, 0x08090a0b0c0d0e0f, 0x0001020304050607,
        0x08090a0b0c0d0e0f, 0x0001020304050607, 0x08090a0b0c0d0e0f, 0x0001020304050607);
    const __m512i C0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0x3931292119110901, 0x3830282018100800);
    const __m512i C1 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0x3b332b231b130b03, 0x3a322a221a120a02);
    const __m512i C2 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0x3d352d251d150d05, 0x3c342c241c140c04);
    const __m512i C3 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0x3f372f271f170f07, 0x3e362e261e160e06);
    const __m512i I8 = _mm512_set1_epi64(0x8040201008040201);
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m512i a = _mm512_loadu_si512(&in[i * 8]);
        const __m512i u = _mm512_gf2p8affine_epi64_epi8(I8, _mm512_shuffle_epi8(a, BSWAP64), 0x00);
        const __m128i u0 = _mm512_castsi512_si128(_mm512_permutexvar_epi8(C0, u));
        const __m128i u1 = _mm512_castsi512_si128(_mm512_permutexvar_epi8(C1, u));
        const __m128i u2 = _mm512_castsi512_si128(_mm512_permutexvar_epi8(C2, u));
        const __m128i u3 = _mm512_castsi512_si128(_mm512_permutexvar_epi8(C3, u));
        _mm_storel_epi64((__m128i*)&out[0 * size + i], u0);
        _mm_storel_epi64((__m128i*)&out[1 * size + i], _mm_srli_si128(u0, 8));
        _mm_storel_epi64((__m128i*)&out[2 * size + i], u1);
        _mm_storel_epi64((__m128i*)&out[3 * size + i], _mm_srli_si128(u1, 8));
        _mm_storel_epi64((__m128i*)&out[4 * size + i], u2);
        _mm_storel_epi64((__m128i*)&out[5 * size + i], _mm_srli_si128(u2, 8));
        _mm_storel_epi64((__m128i*)&out[6 * size + i], u3);
        _mm_storel_epi64((__m128i*)&out[7 * size + i], _mm_srli_si128(u3, 8));
    }
    if (i < size) {
        const __mmask8 k = (1U << (size - i)) - 1;
        const __m512i a = _mm512_maskz_loadu_epi64(k, &in[i * 8]);
        const __m512i u = _mm512_gf2p8affine_epi64_epi8(I8, _mm512_shuffle_epi8(a, BSWAP64), 0x00);
        const __m128i u0 = _mm512_castsi512_si128(_mm512_permutexvar_epi8(C0, u));
        const __m128i u1 = _mm512_castsi512_si128(_mm512_permutexvar_epi8(C1, u));
        const __m128i u2 = _mm512_castsi512_si128(_mm512_permutexvar_epi8(C2, u));
        const __m128i u3 = _mm512_castsi512_si128(_mm512_permutexvar_epi8(C3, u));
        _mm_mask_storeu_epi8(&out[0 * size + i], k, u0);
        _mm_mask_storeu_epi8(&out[1 * size + i], k, _mm_srli_si128(u0, 8));
        _mm_mask_storeu_epi8(&out[2 * size + i], k, u1);
        _mm_mask_storeu_epi8(&out[3 * size + i], k, _mm_srli_si128(u1, 8));
        _mm_mask_storeu_epi8(&out[4 * size + i], k, u2);
        _mm_mask_storeu_epi8(&out[5 * size + i], k, _mm_srli_si128(u2, 8));
        _mm_mask_storeu_epi8(&out[6 * size + i], k, u3);
        _mm_mask_storeu_epi8(&out[7 * size + i], k, _mm_srli_si128(u3, 8));
    }
}
#endif

#if defined(__AVX512BW__) && defined(__AVX512VL__) || BITSHUF_USE_IFUNC
IMPLEMENT_STORE_FUNCTION(STORE_MASK64, __mmask64)

NO_INLINE
ATTRIBUTE_TARGET("avx512bw,avx512vl")
static void bitshuf_trans_bit_avx512bw(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);
    size /= 8;

    const __m512i C0 = _mm512_set1_epi8(0x01);
    const __m512i C1 = _mm512_set1_epi8(0x02);
    const __m512i C2 = _mm512_set1_epi8(0x04);
    const __m512i C3 = _mm512_set1_epi8(0x08);
    const __m512i C4 = _mm512_set1_epi8(0x10);
    const __m512i C5 = _mm512_set1_epi8(0x20);
    const __m512i C6 = _mm512_set1_epi8(0x40);
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m512i a = _mm512_loadu_si512(&in[i * 8]);
        STORE_MASK64(&out[0 * size + i], _mm512_test_epi8_mask(a, C0));
        STORE_MASK64(&out[1 * size + i], _mm512_test_epi8_mask(a, C1));
        STORE_MASK64(&out[2 * size + i], _mm512_test_epi8_mask(a, C2));
        STORE_MASK64(&out[3 * size + i], _mm512_test_epi8_mask(a, C3));
        STORE_MASK64(&out[4 * size + i], _mm512_test_epi8_mask(a, C4));
        STORE_MASK64(&out[5 * size + i], _mm512_test_epi8_mask(a, C5));
        STORE_MASK64(&out[6 * size + i], _mm512_test_epi8_mask(a, C6));
        STORE_MASK64(&out[7 * size + i], _mm512_movepi8_mask(a));
    }
    if (i < size) {
        const __mmask8 k = (1U << (size - i)) - 1;
        const __m512i a = _mm512_maskz_loadu_epi64(k, &in[i * 8]);
        // clang-format off
        _mm_mask_storeu_epi8(&out[0 * size + i], k, _mm_set_epi64x(0, _mm512_test_epi8_mask(a, C0)));
        _mm_mask_storeu_epi8(&out[1 * size + i], k, _mm_set_epi64x(0, _mm512_test_epi8_mask(a, C1)));
        _mm_mask_storeu_epi8(&out[2 * size + i], k, _mm_set_epi64x(0, _mm512_test_epi8_mask(a, C2)));
        _mm_mask_storeu_epi8(&out[3 * size + i], k, _mm_set_epi64x(0, _mm512_test_epi8_mask(a, C3)));
        _mm_mask_storeu_epi8(&out[4 * size + i], k, _mm_set_epi64x(0, _mm512_test_epi8_mask(a, C4)));
        _mm_mask_storeu_epi8(&out[5 * size + i], k, _mm_set_epi64x(0, _mm512_test_epi8_mask(a, C5)));
        _mm_mask_storeu_epi8(&out[6 * size + i], k, _mm_set_epi64x(0, _mm512_test_epi8_mask(a, C6)));
        _mm_mask_storeu_epi8(&out[7 * size + i], k, _mm_set_epi64x(0, _mm512_movepi8_mask(a)));
        // clang-format on
    }
}
#endif

#if defined(__AVX2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("avx2")
static void bitshuf_trans_bit_avx2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);
    size /= 8;

    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        const __m256i a = _mm256_loadu_si256((const __m256i*)&in[i * 8]);
        __m256i u;
        STORE_U32(&out[7 * size + i], _mm256_movemask_epi8(u = a));
        STORE_U32(&out[6 * size + i], _mm256_movemask_epi8(u = _mm256_add_epi8(X(u), u)));
        STORE_U32(&out[5 * size + i], _mm256_movemask_epi8(u = _mm256_add_epi8(X(u), u)));
        STORE_U32(&out[4 * size + i], _mm256_movemask_epi8(u = _mm256_add_epi8(X(u), u)));
        STORE_U32(&out[3 * size + i], _mm256_movemask_epi8(u = _mm256_add_epi8(X(u), u)));
        STORE_U32(&out[2 * size + i], _mm256_movemask_epi8(u = _mm256_add_epi8(X(u), u)));
        STORE_U32(&out[1 * size + i], _mm256_movemask_epi8(u = _mm256_add_epi8(X(u), u)));
        STORE_U32(&out[0 * size + i], _mm256_movemask_epi8(_mm256_add_epi8(X(u), u)));
    }
    if (i + 2 <= size) {
        const __m128i a = _mm_loadu_si128((const __m128i*)&in[i * 8]);
        __m128i u;
        STORE_U16(&out[7 * size + i], _mm_movemask_epi8(u = a));
        STORE_U16(&out[6 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[5 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[4 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[3 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[2 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[1 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[0 * size + i], _mm_movemask_epi8(_mm_add_epi8(X(u), u)));
        i += 2;
    }
    if (i < size) {
        const __m128i a = _mm_loadl_epi64((const __m128i*)&in[i * 8]);
        __m128i u;
        STORE_U8(&out[7 * size + i], _mm_movemask_epi8(u = a));
        STORE_U8(&out[6 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[5 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[4 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[3 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[2 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[1 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[0 * size + i], _mm_movemask_epi8(_mm_add_epi8(X(u), u)));
    }
}
#endif

#if defined(__SSE2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("sse2")
static void bitshuf_trans_bit_sse2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);
    size /= 8;

    size_t i = 0;
    for (; i + 2 <= size; i += 2) {
        const __m128i a = _mm_loadu_si128((const __m128i*)&in[i * 8]);
        __m128i u;
        STORE_U16(&out[7 * size + i], _mm_movemask_epi8(u = a));
        STORE_U16(&out[6 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[5 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[4 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[3 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[2 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[1 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U16(&out[0 * size + i], _mm_movemask_epi8(_mm_add_epi8(X(u), u)));
    }
    if (i < size) {
        const __m128i a = _mm_loadl_epi64((const __m128i*)&in[i * 8]);
        __m128i u;
        STORE_U8(&out[7 * size + i], _mm_movemask_epi8(u = a));
        STORE_U8(&out[6 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[5 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[4 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[3 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[2 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[1 * size + i], _mm_movemask_epi8(u = _mm_add_epi8(X(u), u)));
        STORE_U8(&out[0 * size + i], _mm_movemask_epi8(_mm_add_epi8(X(u), u)));
    }
}
#endif

#if defined(__AVX512VBMI__) && defined(__AVX512VL__) && defined(__GFNI__)
#define bitshuf_trans_bit bitshuf_trans_bit_avx512vbmi_gfni
#elif BITSHUF_USE_IFUNC
IMPLEMENT_IFUNC(bitshuf_trans_bit_ifunc,
                (char* restrict out, const char* restrict in, size_t size)) {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("gfni") && !__builtin_cpu_is("intel"))
    {
        return bitshuf_trans_bit_avx512vbmi_gfni;
    }
#if defined(__AVX512BW__) && defined(__AVX512VL__)
    return bitshuf_trans_bit_avx512bw;
#else
    if (__builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl"))
        return bitshuf_trans_bit_avx512bw;
#if defined(__AVX2__)
    return bitshuf_trans_bit_avx2;
#else
    if (__builtin_cpu_supports("avx2"))
        return bitshuf_trans_bit_avx2;
#if defined(__SSE2__)
    return bitshuf_trans_bit_sse2;
#else
    if (__builtin_cpu_supports("sse2"))
        return bitshuf_trans_bit_sse2;

    return bitshuf_trans_bit;
#endif
#endif
#endif
}
#define bitshuf_trans_bit bitshuf_trans_bit_ifunc
#elif defined(__AVX512BW__) && defined(__AVX512VL__)
#define bitshuf_trans_bit bitshuf_trans_bit_avx512bw
#elif defined(__AVX2__)
#define bitshuf_trans_bit bitshuf_trans_bit_avx2
#elif defined(__SSE2__)
#define bitshuf_trans_bit bitshuf_trans_bit_sse2
#endif

#if defined(__SSE2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("sse2")
static void bitshuf_trans_byte_sse2(char* restrict out,
                                    const char* restrict in,
                                    size_t size,
                                    size_t elem_size) {
    assert(size % 8 == 0);

    size_t j = 0;
    for (; j + 8 <= elem_size; j += 8) {
        for (size_t i = 0; i < size; i += 8) {
            const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[(i + 0) * elem_size + j]);
            const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[(i + 1) * elem_size + j]);
            const __m128i a2 = _mm_loadl_epi64((const __m128i*)&in[(i + 2) * elem_size + j]);
            const __m128i a3 = _mm_loadl_epi64((const __m128i*)&in[(i + 3) * elem_size + j]);
            const __m128i a4 = _mm_loadl_epi64((const __m128i*)&in[(i + 4) * elem_size + j]);
            const __m128i a5 = _mm_loadl_epi64((const __m128i*)&in[(i + 5) * elem_size + j]);
            const __m128i a6 = _mm_loadl_epi64((const __m128i*)&in[(i + 6) * elem_size + j]);
            const __m128i a7 = _mm_loadl_epi64((const __m128i*)&in[(i + 7) * elem_size + j]);
            __m128i u0 = _mm_unpacklo_epi8(a0, a1);
            __m128i u1 = _mm_unpacklo_epi8(a2, a3);
            __m128i u2 = _mm_unpacklo_epi8(a4, a5);
            __m128i u3 = _mm_unpacklo_epi8(a6, a7);
            const __m128i v0 = _mm_unpacklo_epi16(u0, u1);
            const __m128i v1 = _mm_unpackhi_epi16(u0, u1);
            const __m128i v2 = _mm_unpacklo_epi16(u2, u3);
            const __m128i v3 = _mm_unpackhi_epi16(u2, u3);
            u0 = _mm_unpacklo_epi32(v0, v2);
            u1 = _mm_unpackhi_epi32(v0, v2);
            u2 = _mm_unpacklo_epi32(v1, v3);
            u3 = _mm_unpackhi_epi32(v1, v3);
            _mm_storel_epi64((__m128i*)&out[(j + 0) * size + i], u0);
            _mm_storel_epi64((__m128i*)&out[(j + 1) * size + i], _mm_srli_si128(u0, 8));
            _mm_storel_epi64((__m128i*)&out[(j + 2) * size + i], u1);
            _mm_storel_epi64((__m128i*)&out[(j + 3) * size + i], _mm_srli_si128(u1, 8));
            _mm_storel_epi64((__m128i*)&out[(j + 4) * size + i], u2);
            _mm_storel_epi64((__m128i*)&out[(j + 5) * size + i], _mm_srli_si128(u2, 8));
            _mm_storel_epi64((__m128i*)&out[(j + 6) * size + i], u3);
            _mm_storel_epi64((__m128i*)&out[(j + 7) * size + i], _mm_srli_si128(u3, 8));
        }
    }
    if (j < elem_size) {
        for (size_t i = 0; i + 8 < size; i += 8) {
            const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[(i + 0) * elem_size + j]);
            const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[(i + 1) * elem_size + j]);
            const __m128i a2 = _mm_loadl_epi64((const __m128i*)&in[(i + 2) * elem_size + j]);
            const __m128i a3 = _mm_loadl_epi64((const __m128i*)&in[(i + 3) * elem_size + j]);
            const __m128i a4 = _mm_loadl_epi64((const __m128i*)&in[(i + 4) * elem_size + j]);
            const __m128i a5 = _mm_loadl_epi64((const __m128i*)&in[(i + 5) * elem_size + j]);
            const __m128i a6 = _mm_loadl_epi64((const __m128i*)&in[(i + 6) * elem_size + j]);
            const __m128i a7 = _mm_loadl_epi64((const __m128i*)&in[(i + 7) * elem_size + j]);
            __m128i u0 = _mm_unpacklo_epi8(a0, a1);
            __m128i u1 = _mm_unpacklo_epi8(a2, a3);
            __m128i u2 = _mm_unpacklo_epi8(a4, a5);
            __m128i u3 = _mm_unpacklo_epi8(a6, a7);
            const __m128i v0 = _mm_unpacklo_epi16(u0, u1);
            const __m128i v1 = _mm_unpackhi_epi16(u0, u1);
            const __m128i v2 = _mm_unpacklo_epi16(u2, u3);
            const __m128i v3 = _mm_unpackhi_epi16(u2, u3);
            u0 = _mm_unpacklo_epi32(v0, v2);
            u1 = _mm_unpackhi_epi32(v0, v2);
            u2 = _mm_unpacklo_epi32(v1, v3);
            u3 = _mm_unpackhi_epi32(v1, v3);
            switch (elem_size - j) {
                case 7:
                    _mm_storel_epi64((__m128i*)&out[(j + 6) * size + i], u3);
                    FALLTHROUGH;
                case 6:
                    _mm_storel_epi64((__m128i*)&out[(j + 5) * size + i], _mm_srli_si128(u2, 8));
                    FALLTHROUGH;
                case 5:
                    _mm_storel_epi64((__m128i*)&out[(j + 4) * size + i], u2);
                    FALLTHROUGH;
                case 4:
                    _mm_storel_epi64((__m128i*)&out[(j + 3) * size + i], _mm_srli_si128(u1, 8));
                    FALLTHROUGH;
                case 3:
                    _mm_storel_epi64((__m128i*)&out[(j + 2) * size + i], u1);
                    FALLTHROUGH;
                case 2:
                    _mm_storel_epi64((__m128i*)&out[(j + 1) * size + i], _mm_srli_si128(u0, 8));
                    FALLTHROUGH;
                default:
                    _mm_storel_epi64((__m128i*)&out[(j + 0) * size + i], u0);
            }
        }
        for (; j < elem_size; j++) {
            for (size_t i = size - 8; i < size; i++)
                out[j * size + i] = in[i * elem_size + j];
        }
    }
}
#endif

#if defined(__SSE2__)
#define bitshuf_trans_byte bitshuf_trans_byte_sse2
#elif BITSHUF_USE_IFUNC
IMPLEMENT_IFUNC(bitshuf_trans_byte_ifunc,
                (char* restrict out, const char* restrict in, size_t size, size_t elem_size)) {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("sse2"))
        return bitshuf_trans_byte_sse2;

    return bitshuf_trans_byte;
}
#define bitshuf_trans_byte bitshuf_trans_byte_ifunc
#endif

#if defined(__AVX2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("avx2")
static void bitshuf_trans_byte_2_avx2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);

    const __m256i MASK = _mm256_set1_epi16(0x00ff);
    size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        const __m256i a0 = _mm256_loadu_si256((const __m256i*)&in[i * 2]);
        const __m256i a1 = _mm256_loadu_si256((const __m256i*)&in[i * 2 + 32]);
        __m256i u0 = _mm256_inserti128_si256(a0, _mm256_castsi256_si128(a1), 1);
        __m256i u1 = _mm256_permute2x128_si256(a0, a1, 0x31);
        const __m256i v0 = _mm256_and_si256(u0, MASK);
        const __m256i v1 = _mm256_and_si256(u1, MASK);
        const __m256i v2 = _mm256_srli_epi16(u0, 8);
        const __m256i v3 = _mm256_srli_epi16(u1, 8);
        u0 = _mm256_packus_epi16(v0, v1);
        u1 = _mm256_packus_epi16(v2, v3);
        _mm256_storeu_si256((__m256i*)&out[0 * size + i], u0);
        _mm256_storeu_si256((__m256i*)&out[1 * size + i], u1);
    }
    if (i + 16 <= size) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[i * 2]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[i * 2 + 16]);
        const __m128i u0 = _mm_and_si128(a0, _mm256_castsi256_si128(MASK));
        const __m128i u1 = _mm_and_si128(a1, _mm256_castsi256_si128(MASK));
        const __m128i u2 = _mm_srli_epi16(a0, 8);
        const __m128i u3 = _mm_srli_epi16(a1, 8);
        const __m128i v0 = _mm_packus_epi16(u0, u1);
        const __m128i v1 = _mm_packus_epi16(u2, u3);
        _mm_storeu_si128((__m128i*)&out[0 * size + i], v0);
        _mm_storeu_si128((__m128i*)&out[1 * size + i], v1);
        i += 16;
    }
    if (i < size) {
        const __m128i a = _mm_loadu_si128((const __m128i*)&in[i * 2]);
        const __m128i u0 = _mm_and_si128(a, _mm256_castsi256_si128(MASK));
        const __m128i u1 = _mm_srli_epi16(a, 8);
        const __m128i u = _mm_packus_epi16(u0, u1);
        _mm_storel_epi64((__m128i*)&out[0 * size + i], u);
        _mm_storel_epi64((__m128i*)&out[1 * size + i], _mm_srli_si128(u, 8));
    }
}
#endif

#if defined(__SSE2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("sse2")
static void bitshuf_trans_byte_2_sse2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);

    const __m128i MASK = _mm_set1_epi16(0x00ff);
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[i * 2]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[i * 2 + 16]);
        const __m128i u0 = _mm_packus_epi16(_mm_and_si128(a0, MASK), _mm_and_si128(a1, MASK));
        const __m128i u1 = _mm_packus_epi16(_mm_srli_epi16(a0, 8), _mm_srli_epi16(a1, 8));
        _mm_storeu_si128((__m128i*)&out[0 * size + i], u0);
        _mm_storeu_si128((__m128i*)&out[1 * size + i], u1);
    }
    if (i < size) {
        const __m128i a = _mm_loadu_si128((const __m128i*)&in[i * 2]);
        const __m128i u = _mm_packus_epi16(_mm_and_si128(a, MASK), _mm_srli_epi16(a, 8));
        _mm_storel_epi64((__m128i*)&out[0 * size + i], u);
        _mm_storel_epi64((__m128i*)&out[1 * size + i], _mm_srli_si128(u, 8));
    }
}
#endif

#if defined(__AVX2__)
#define bitshuf_trans_byte_2 bitshuf_trans_byte_2_avx2
#elif BITSHUF_USE_IFUNC
IMPLEMENT_IFUNC(bitshuf_trans_byte_2_ifunc,
                (char* restrict out, const char* restrict in, size_t size)) {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx2"))
        return bitshuf_trans_byte_2_avx2;
#if defined(__SSE2__)
    return bitshuf_trans_byte_2_sse2;
#else
    if (__builtin_cpu_supports("sse2"))
        return bitshuf_trans_byte_2_sse2;

    return bitshuf_trans_byte_2;
#endif
}
#define bitshuf_trans_byte_2 bitshuf_trans_byte_2_ifunc
#elif defined(__SSE2__)
#define bitshuf_trans_byte_2 bitshuf_trans_byte_2_sse2
#endif

#if defined(__AVX2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("avx2")
static void bitshuf_trans_byte_4_avx2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);

    const __m256i SHUF = _mm256_set_epi64x(0x0f0b07030e0a0602, 0x0d0905010c080400,
                                           0x0f0b07030e0a0602, 0x0d0905010c080400);
    const __m256i PERM = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        const __m256i a0 = _mm256_loadu_si256((const __m256i*)&in[i * 4]);
        const __m256i a1 = _mm256_loadu_si256((const __m256i*)&in[i * 4 + 32]);
        const __m256i u0 = _mm256_shuffle_epi8(a0, SHUF);
        const __m256i u1 = _mm256_shuffle_epi8(a1, SHUF);
        const __m256i v0 = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi32(u0, u1), PERM);
        const __m256i v1 = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi32(u0, u1), PERM);
        _mm_storeu_si128((__m128i*)&out[0 * size + i], _mm256_castsi256_si128(v0));
        _mm_storeu_si128((__m128i*)&out[1 * size + i], _mm256_extracti128_si256(v0, 1));
        _mm_storeu_si128((__m128i*)&out[2 * size + i], _mm256_castsi256_si128(v1));
        _mm_storeu_si128((__m128i*)&out[3 * size + i], _mm256_extracti128_si256(v1, 1));
    }
    if (i + 8 <= size) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[i * 4]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[i * 4 + 16]);
        const __m128i u0 = _mm_shuffle_epi8(a0, _mm256_castsi256_si128(SHUF));
        const __m128i u1 = _mm_shuffle_epi8(a1, _mm256_castsi256_si128(SHUF));
        const __m128i v0 = _mm_unpacklo_epi32(u0, u1);
        const __m128i v1 = _mm_unpackhi_epi32(u0, u1);
        _mm_storel_epi64((__m128i*)&out[0 * size + i], v0);
        _mm_storel_epi64((__m128i*)&out[1 * size + i], _mm_srli_si128(v0, 8));
        _mm_storel_epi64((__m128i*)&out[2 * size + i], v1);
        _mm_storel_epi64((__m128i*)&out[3 * size + i], _mm_srli_si128(v1, 8));
    }
}
#endif

#if defined(__SSE2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("sse2")
static void bitshuf_trans_byte_4_sse2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);

    const __m128i MASK = _mm_set1_epi16(0x00ff);
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[i * 4 + 16 * 0]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[i * 4 + 16 * 1]);
        const __m128i a2 = _mm_loadu_si128((const __m128i*)&in[i * 4 + 16 * 2]);
        const __m128i a3 = _mm_loadu_si128((const __m128i*)&in[i * 4 + 16 * 3]);
        const __m128i u0 = _mm_packus_epi16(_mm_and_si128(a0, MASK), _mm_and_si128(a1, MASK));
        const __m128i u1 = _mm_packus_epi16(_mm_and_si128(a2, MASK), _mm_and_si128(a3, MASK));
        const __m128i u2 = _mm_packus_epi16(_mm_srli_epi16(a0, 8), _mm_srli_epi16(a1, 8));
        const __m128i u3 = _mm_packus_epi16(_mm_srli_epi16(a2, 8), _mm_srli_epi16(a3, 8));
        const __m128i v0 = _mm_packus_epi16(_mm_and_si128(u0, MASK), _mm_and_si128(u1, MASK));
        const __m128i v1 = _mm_packus_epi16(_mm_and_si128(u2, MASK), _mm_and_si128(u3, MASK));
        const __m128i v2 = _mm_packus_epi16(_mm_srli_epi16(u0, 8), _mm_srli_epi16(u1, 8));
        const __m128i v3 = _mm_packus_epi16(_mm_srli_epi16(u2, 8), _mm_srli_epi16(u3, 8));
        _mm_storeu_si128((__m128i*)&out[0 * size + i], v0);
        _mm_storeu_si128((__m128i*)&out[1 * size + i], v1);
        _mm_storeu_si128((__m128i*)&out[2 * size + i], v2);
        _mm_storeu_si128((__m128i*)&out[3 * size + i], v3);
    }
    if (i + 8 <= size) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[i * 4]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[i * 4 + 16]);
        const __m128i u0 = _mm_packus_epi16(_mm_and_si128(a0, MASK), _mm_and_si128(a1, MASK));
        const __m128i u1 = _mm_packus_epi16(_mm_srli_epi16(a0, 8), _mm_srli_epi16(a1, 8));
        const __m128i v0 = _mm_packus_epi16(_mm_and_si128(u0, MASK), _mm_and_si128(u1, MASK));
        const __m128i v1 = _mm_packus_epi16(_mm_srli_epi16(u0, 8), _mm_srli_epi16(u1, 8));
        _mm_storel_epi64((__m128i*)&out[0 * size + i], v0);
        _mm_storel_epi64((__m128i*)&out[1 * size + i], _mm_srli_si128(v0, 8));
        _mm_storel_epi64((__m128i*)&out[2 * size + i], v1);
        _mm_storel_epi64((__m128i*)&out[3 * size + i], _mm_srli_si128(v1, 8));
    }
}
#endif

#if defined(__AVX2__)
#define bitshuf_trans_byte_4 bitshuf_trans_byte_4_avx2
#elif BITSHUF_USE_IFUNC
IMPLEMENT_IFUNC(bitshuf_trans_byte_4_ifunc,
                (char* restrict out, const char* restrict in, size_t size)) {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx2"))
        return bitshuf_trans_byte_4_avx2;
#if defined(__SSE2__)
    return bitshuf_trans_byte_4_sse2;
#else
    if (__builtin_cpu_supports("sse2"))
        return bitshuf_trans_byte_4_sse2;

    return bitshuf_trans_byte_4;
#endif
}
#define bitshuf_trans_byte_4 bitshuf_trans_byte_4_ifunc
#elif defined(__SSE2__)
#define bitshuf_trans_byte_4 bitshuf_trans_byte_4_sse2
#endif

#if defined(__SSE2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("sse2")
static void bitshuf_trans_byte_8_sse2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);

    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 0]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 1]);
        const __m128i a2 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 2]);
        const __m128i a3 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 3]);
        const __m128i a4 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 4]);
        const __m128i a5 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 5]);
        const __m128i a6 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 6]);
        const __m128i a7 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 7]);
        __m128i u0 = _mm_unpacklo_epi8(a0, a1);
        __m128i u1 = _mm_unpackhi_epi8(a0, a1);
        __m128i u2 = _mm_unpacklo_epi8(a2, a3);
        __m128i u3 = _mm_unpackhi_epi8(a2, a3);
        __m128i u4 = _mm_unpacklo_epi8(a4, a5);
        __m128i u5 = _mm_unpackhi_epi8(a4, a5);
        __m128i u6 = _mm_unpacklo_epi8(a6, a7);
        __m128i u7 = _mm_unpackhi_epi8(a6, a7);
        __m128i v0 = _mm_unpacklo_epi8(u0, u1);
        __m128i v1 = _mm_unpackhi_epi8(u0, u1);
        __m128i v2 = _mm_unpacklo_epi8(u2, u3);
        __m128i v3 = _mm_unpackhi_epi8(u2, u3);
        __m128i v4 = _mm_unpacklo_epi8(u4, u5);
        __m128i v5 = _mm_unpackhi_epi8(u4, u5);
        __m128i v6 = _mm_unpacklo_epi8(u6, u7);
        __m128i v7 = _mm_unpackhi_epi8(u6, u7);
        u0 = _mm_unpacklo_epi32(v0, v2);
        u1 = _mm_unpackhi_epi32(v0, v2);
        u2 = _mm_unpacklo_epi32(v1, v3);
        u3 = _mm_unpackhi_epi32(v1, v3);
        u4 = _mm_unpacklo_epi32(v4, v6);
        u5 = _mm_unpackhi_epi32(v4, v6);
        u6 = _mm_unpacklo_epi32(v5, v7);
        u7 = _mm_unpackhi_epi32(v5, v7);
        v0 = _mm_unpacklo_epi64(u0, u4);
        v1 = _mm_unpackhi_epi64(u0, u4);
        v2 = _mm_unpacklo_epi64(u1, u5);
        v3 = _mm_unpackhi_epi64(u1, u5);
        v4 = _mm_unpacklo_epi64(u2, u6);
        v5 = _mm_unpackhi_epi64(u2, u6);
        v6 = _mm_unpacklo_epi64(u3, u7);
        v7 = _mm_unpackhi_epi64(u3, u7);
        _mm_storeu_si128((__m128i*)&out[0 * size + i], v0);
        _mm_storeu_si128((__m128i*)&out[1 * size + i], v1);
        _mm_storeu_si128((__m128i*)&out[2 * size + i], v2);
        _mm_storeu_si128((__m128i*)&out[3 * size + i], v3);
        _mm_storeu_si128((__m128i*)&out[4 * size + i], v4);
        _mm_storeu_si128((__m128i*)&out[5 * size + i], v5);
        _mm_storeu_si128((__m128i*)&out[6 * size + i], v6);
        _mm_storeu_si128((__m128i*)&out[7 * size + i], v7);
    }
    if (i + 8 <= size) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 0]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 1]);
        const __m128i a2 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 2]);
        const __m128i a3 = _mm_loadu_si128((const __m128i*)&in[i * 8 + 16 * 3]);
        __m128i u0 = _mm_unpacklo_epi8(a0, a1);
        __m128i u1 = _mm_unpackhi_epi8(a0, a1);
        __m128i u2 = _mm_unpacklo_epi8(a2, a3);
        __m128i u3 = _mm_unpackhi_epi8(a2, a3);
        const __m128i v0 = _mm_unpacklo_epi8(u0, u1);
        const __m128i v1 = _mm_unpackhi_epi8(u0, u1);
        const __m128i v2 = _mm_unpacklo_epi8(u2, u3);
        const __m128i v3 = _mm_unpackhi_epi8(u2, u3);
        u0 = _mm_unpacklo_epi32(v0, v2);
        u1 = _mm_unpackhi_epi32(v0, v2);
        u2 = _mm_unpacklo_epi32(v1, v3);
        u3 = _mm_unpackhi_epi32(v1, v3);
        _mm_storel_epi64((__m128i*)&out[0 * size + i], u0);
        _mm_storel_epi64((__m128i*)&out[1 * size + i], _mm_srli_si128(u0, 8));
        _mm_storel_epi64((__m128i*)&out[2 * size + i], u1);
        _mm_storel_epi64((__m128i*)&out[3 * size + i], _mm_srli_si128(u1, 8));
        _mm_storel_epi64((__m128i*)&out[4 * size + i], u2);
        _mm_storel_epi64((__m128i*)&out[5 * size + i], _mm_srli_si128(u2, 8));
        _mm_storel_epi64((__m128i*)&out[6 * size + i], u3);
        _mm_storel_epi64((__m128i*)&out[7 * size + i], _mm_srli_si128(u3, 8));
    }
}
#endif

#if defined(__SSE2__)
#define bitshuf_trans_byte_8 bitshuf_trans_byte_8_sse2
#elif BITSHUF_USE_IFUNC
IMPLEMENT_IFUNC(bitshuf_trans_byte_8_ifunc,
                (char* restrict out, const char* restrict in, size_t size)) {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("sse2"))
        return bitshuf_trans_byte_8_sse2;

    return bitshuf_trans_byte_8;
}
#define bitshuf_trans_byte_8 bitshuf_trans_byte_8_ifunc
#endif

#if defined(__AVX512VBMI__) && defined(__AVX512VL__) && defined(__GFNI__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("avx512vbmi,avx512vl,gfni")
static void bitshuf_untrans_bit_avx512vbmi_gfni(char* restrict out,
                                                const char* restrict in,
                                                size_t size) {
    assert(size % 8 == 0);
    size /= 8;

    const __m512i C = _mm512_set_epi64(0x070f171f474f575f, 0x060e161e464e565e, 0x050d151d454d555d,
                                       0x040c141c444c545c, 0x030b131b434b535b, 0x020a121a424a525a,
                                       0x0109111941495159, 0x0008101840485058);
    const __m512i I8 = _mm512_set1_epi64(0x8040201008040201);
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
#if defined(__x86_64__) || defined(_M_X64)
        const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[0 * size + i]);
        const int64_t a1 = LOAD_I64(&in[1 * size + i]);
        const __m128i a2 = _mm_loadl_epi64((const __m128i*)&in[2 * size + i]);
        const int64_t a3 = LOAD_I64(&in[3 * size + i]);
        const __m128i a4 = _mm_loadl_epi64((const __m128i*)&in[4 * size + i]);
        const int64_t a5 = LOAD_I64(&in[5 * size + i]);
        const __m128i a6 = _mm_loadl_epi64((const __m128i*)&in[6 * size + i]);
        const int64_t a7 = LOAD_I64(&in[7 * size + i]);
        const __m128i u0 = _mm_insert_epi64(a0, a1, 1);
        const __m128i u1 = _mm_insert_epi64(a2, a3, 1);
        const __m128i u2 = _mm_insert_epi64(a4, a5, 1);
        const __m128i u3 = _mm_insert_epi64(a6, a7, 1);
#else
        const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[1 * size + i]);
        const __m128i a2 = _mm_loadl_epi64((const __m128i*)&in[2 * size + i]);
        const __m128i a3 = _mm_loadl_epi64((const __m128i*)&in[3 * size + i]);
        const __m128i a4 = _mm_loadl_epi64((const __m128i*)&in[4 * size + i]);
        const __m128i a5 = _mm_loadl_epi64((const __m128i*)&in[5 * size + i]);
        const __m128i a6 = _mm_loadl_epi64((const __m128i*)&in[6 * size + i]);
        const __m128i a7 = _mm_loadl_epi64((const __m128i*)&in[7 * size + i]);
        const __m128i u0 = _mm_unpacklo_epi64(a0, a1);
        const __m128i u1 = _mm_unpacklo_epi64(a2, a3);
        const __m128i u2 = _mm_unpacklo_epi64(a4, a5);
        const __m128i u3 = _mm_unpacklo_epi64(a6, a7);
#endif
        const __m256i v0 = _mm256_inserti128_si256(_mm256_castsi128_si256(u0), u1, 1);
        const __m256i v1 = _mm256_inserti128_si256(_mm256_castsi128_si256(u2), u3, 1);
        __m512i u;
        u = _mm512_permutex2var_epi8(_mm512_castsi256_si512(v0), C, _mm512_castsi256_si512(v1));
        u = _mm512_gf2p8affine_epi64_epi8(I8, u, 0x00);
        _mm512_storeu_si512(&out[i * 8], u);
    }
    if (i < size) {
        const __mmask8 k = (1U << (size - i)) - 1;
        const __m128i a0 = _mm_maskz_loadu_epi8(k, &in[0 * size + i]);
        const __m128i a1 = _mm_maskz_loadu_epi8(k, &in[1 * size + i]);
        const __m128i a2 = _mm_maskz_loadu_epi8(k, &in[2 * size + i]);
        const __m128i a3 = _mm_maskz_loadu_epi8(k, &in[3 * size + i]);
        const __m128i a4 = _mm_maskz_loadu_epi8(k, &in[4 * size + i]);
        const __m128i a5 = _mm_maskz_loadu_epi8(k, &in[5 * size + i]);
        const __m128i a6 = _mm_maskz_loadu_epi8(k, &in[6 * size + i]);
        const __m128i a7 = _mm_maskz_loadu_epi8(k, &in[7 * size + i]);
        const __m128i u0 = _mm_unpacklo_epi64(a0, a1);
        const __m128i u1 = _mm_unpacklo_epi64(a2, a3);
        const __m128i u2 = _mm_unpacklo_epi64(a4, a5);
        const __m128i u3 = _mm_unpacklo_epi64(a6, a7);
        const __m256i v0 = _mm256_inserti128_si256(_mm256_castsi128_si256(u0), u1, 1);
        const __m256i v1 = _mm256_inserti128_si256(_mm256_castsi128_si256(u2), u3, 1);
        __m512i u;
        u = _mm512_permutex2var_epi8(_mm512_castsi256_si512(v0), C, _mm512_castsi256_si512(v1));
        u = _mm512_gf2p8affine_epi64_epi8(I8, u, 0x00);
        _mm512_mask_storeu_epi64(&out[i * 8], k, u);
    }
}
#endif

#if defined(__AVX512BW__) && defined(__AVX512VL__) || BITSHUF_USE_IFUNC
IMPLEMENT_LOAD_FUNCTION(LOAD_MASK64, __mmask64)

ATTRIBUTE_TARGET("sse2")
static ALWAYS_INLINE __mmask64 MM_CVTSI128_MASK64(__m128i a) {
#if defined(__x86_64__) || defined(_M_X64)
    return _mm_cvtsi128_si64(a);
#else
    __mmask64 k;
    _mm_storel_epi64((__m128i*)&k, a);
    return k;
#endif
}

NO_INLINE
ATTRIBUTE_TARGET("avx512bw,avx512vl")
static void bitshuf_untrans_bit_avx512bw(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);
    size /= 8;

    const __m512i C0 = _mm512_set1_epi8(0x01);
    const __m512i C1 = _mm512_set1_epi8(0x02);
    const __m512i C2 = _mm512_set1_epi8(0x04);
    const __m512i C3 = _mm512_set1_epi8(0x08);
    const __m512i C4 = _mm512_set1_epi8(0x10);
    const __m512i C5 = _mm512_set1_epi8(0x20);
    const __m512i C6 = _mm512_set1_epi8(0x40);
    const __m512i C7 = _mm512_set1_epi8(-128);
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m512i u = _mm512_maskz_mov_epi8(LOAD_MASK64(&in[0 * size + i]), C0);
        u = _mm512_mask_add_epi8(X(u), LOAD_MASK64(&in[1 * size + i]), u, C1);
        u = _mm512_mask_add_epi8(X(u), LOAD_MASK64(&in[2 * size + i]), u, C2);
        u = _mm512_mask_add_epi8(X(u), LOAD_MASK64(&in[3 * size + i]), u, C3);
        u = _mm512_mask_add_epi8(X(u), LOAD_MASK64(&in[4 * size + i]), u, C4);
        u = _mm512_mask_add_epi8(X(u), LOAD_MASK64(&in[5 * size + i]), u, C5);
        u = _mm512_mask_add_epi8(X(u), LOAD_MASK64(&in[6 * size + i]), u, C6);
        u = _mm512_mask_add_epi8(X(u), LOAD_MASK64(&in[7 * size + i]), u, C7);
        _mm512_storeu_si512(&out[i * 8], u);
    }
    if (i < size) {
        const __mmask8 k = (1U << (size - i)) - 1;
        const __mmask64 a0 = MM_CVTSI128_MASK64(_mm_maskz_loadu_epi8(k, &in[0 * size + i]));
        const __mmask64 a1 = MM_CVTSI128_MASK64(_mm_maskz_loadu_epi8(k, &in[1 * size + i]));
        const __mmask64 a2 = MM_CVTSI128_MASK64(_mm_maskz_loadu_epi8(k, &in[2 * size + i]));
        const __mmask64 a3 = MM_CVTSI128_MASK64(_mm_maskz_loadu_epi8(k, &in[3 * size + i]));
        const __mmask64 a4 = MM_CVTSI128_MASK64(_mm_maskz_loadu_epi8(k, &in[4 * size + i]));
        const __mmask64 a5 = MM_CVTSI128_MASK64(_mm_maskz_loadu_epi8(k, &in[5 * size + i]));
        const __mmask64 a6 = MM_CVTSI128_MASK64(_mm_maskz_loadu_epi8(k, &in[6 * size + i]));
        const __mmask64 a7 = MM_CVTSI128_MASK64(_mm_maskz_loadu_epi8(k, &in[7 * size + i]));
        __m512i u = _mm512_maskz_mov_epi8(a0, C0);
        u = _mm512_mask_add_epi8(X(u), a1, u, C1);
        u = _mm512_mask_add_epi8(X(u), a2, u, C2);
        u = _mm512_mask_add_epi8(X(u), a3, u, C3);
        u = _mm512_mask_add_epi8(X(u), a4, u, C4);
        u = _mm512_mask_add_epi8(X(u), a5, u, C5);
        u = _mm512_mask_add_epi8(X(u), a6, u, C6);
        u = _mm512_mask_add_epi8(X(u), a7, u, C7);
        _mm512_mask_storeu_epi64(&out[i * 8], k, u);
    }
}
#endif

#if defined(__AVX2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("avx2")
static void bitshuf_untrans_bit_avx2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);
    size /= 8;

    const __m256i PERM = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    const __m256i MASK0 = _mm256_set1_epi64x(0x00aa00aa00aa00aa);
    const __m256i MASK1 = _mm256_set1_epi64x(0x0000cccc0000cccc);
    const __m256i MASK2 = _mm256_set1_epi64x(0x00000000f0f0f0f0);
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[1 * size + i]);
        const __m128i a2 = _mm_loadl_epi64((const __m128i*)&in[2 * size + i]);
        const __m128i a3 = _mm_loadl_epi64((const __m128i*)&in[3 * size + i]);
        const __m128i a4 = _mm_loadl_epi64((const __m128i*)&in[4 * size + i]);
        const __m128i a5 = _mm_loadl_epi64((const __m128i*)&in[5 * size + i]);
        const __m128i a6 = _mm_loadl_epi64((const __m128i*)&in[6 * size + i]);
        const __m128i a7 = _mm_loadl_epi64((const __m128i*)&in[7 * size + i]);
        __m256i u0 = MM256_SETR_M128I(_mm_unpacklo_epi8(a0, a1), _mm_unpacklo_epi8(a4, a5));
        __m256i u1 = MM256_SETR_M128I(_mm_unpacklo_epi8(a2, a3), _mm_unpacklo_epi8(a6, a7));
        __m256i v0 = _mm256_unpacklo_epi16(u0, u1);
        __m256i v1 = _mm256_unpackhi_epi16(u0, u1);
        u0 = _mm256_permutevar8x32_epi32(v0, PERM);
        u1 = _mm256_permutevar8x32_epi32(v1, PERM);
        v0 = _mm256_and_si256(_mm256_xor_si256(u0, _mm256_srli_epi64(u0, 07)), MASK0);
        v1 = _mm256_and_si256(_mm256_xor_si256(u1, _mm256_srli_epi64(u1, 07)), MASK0);
        u0 = _mm256_xor_si256(_mm256_xor_si256(u0, _mm256_slli_epi64(v0, 07)), v0);
        u1 = _mm256_xor_si256(_mm256_xor_si256(u1, _mm256_slli_epi64(v1, 07)), v1);
        v0 = _mm256_and_si256(_mm256_xor_si256(u0, _mm256_srli_epi64(u0, 14)), MASK1);
        v1 = _mm256_and_si256(_mm256_xor_si256(u1, _mm256_srli_epi64(u1, 14)), MASK1);
        u0 = _mm256_xor_si256(_mm256_xor_si256(u0, _mm256_slli_epi64(v0, 14)), v0);
        u1 = _mm256_xor_si256(_mm256_xor_si256(u1, _mm256_slli_epi64(v1, 14)), v1);
        v0 = _mm256_and_si256(_mm256_xor_si256(u0, _mm256_srli_epi64(u0, 28)), MASK2);
        v1 = _mm256_and_si256(_mm256_xor_si256(u1, _mm256_srli_epi64(u1, 28)), MASK2);
        u0 = _mm256_xor_si256(_mm256_xor_si256(u0, _mm256_slli_epi64(v0, 28)), v0);
        u1 = _mm256_xor_si256(_mm256_xor_si256(u1, _mm256_slli_epi64(v1, 28)), v1);
        _mm256_storeu_si256((__m256i*)&out[i * 8], u0);
        _mm256_storeu_si256((__m256i*)&out[i * 8 + 32], u1);
    }
    if (i < size)
        bitshuf_untrans_bit_tail(out, in, size * 8, i);
}
#endif

#if defined(__SSE2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("sse2")
static void bitshuf_untrans_bit_sse2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);
    size /= 8;

    const __m128i MASK0 = _mm_set1_epi64x(0x00aa00aa00aa00aa);
    const __m128i MASK1 = _mm_set1_epi64x(0x0000cccc0000cccc);
    const __m128i MASK2 = _mm_set1_epi64x(0x00000000f0f0f0f0);
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[1 * size + i]);
        const __m128i a2 = _mm_loadl_epi64((const __m128i*)&in[2 * size + i]);
        const __m128i a3 = _mm_loadl_epi64((const __m128i*)&in[3 * size + i]);
        const __m128i a4 = _mm_loadl_epi64((const __m128i*)&in[4 * size + i]);
        const __m128i a5 = _mm_loadl_epi64((const __m128i*)&in[5 * size + i]);
        const __m128i a6 = _mm_loadl_epi64((const __m128i*)&in[6 * size + i]);
        const __m128i a7 = _mm_loadl_epi64((const __m128i*)&in[7 * size + i]);
        __m128i u0 = _mm_unpacklo_epi8(a0, a1);
        __m128i u1 = _mm_unpacklo_epi8(a2, a3);
        __m128i u2 = _mm_unpacklo_epi8(a4, a5);
        __m128i u3 = _mm_unpacklo_epi8(a6, a7);
        __m128i v0 = _mm_unpacklo_epi16(u0, u1);
        __m128i v1 = _mm_unpackhi_epi16(u0, u1);
        __m128i v2 = _mm_unpacklo_epi16(u2, u3);
        __m128i v3 = _mm_unpackhi_epi16(u2, u3);
        u0 = _mm_unpacklo_epi32(v0, v2);
        u1 = _mm_unpackhi_epi32(v0, v2);
        u2 = _mm_unpacklo_epi32(v1, v3);
        u3 = _mm_unpackhi_epi32(v1, v3);
        v0 = _mm_and_si128(_mm_xor_si128(u0, _mm_srli_epi64(u0, 07)), MASK0);
        v1 = _mm_and_si128(_mm_xor_si128(u1, _mm_srli_epi64(u1, 07)), MASK0);
        v2 = _mm_and_si128(_mm_xor_si128(u2, _mm_srli_epi64(u2, 07)), MASK0);
        v3 = _mm_and_si128(_mm_xor_si128(u3, _mm_srli_epi64(u3, 07)), MASK0);
        u0 = _mm_xor_si128(_mm_xor_si128(u0, _mm_slli_epi64(v0, 07)), v0);
        u1 = _mm_xor_si128(_mm_xor_si128(u1, _mm_slli_epi64(v1, 07)), v1);
        u2 = _mm_xor_si128(_mm_xor_si128(u2, _mm_slli_epi64(v2, 07)), v2);
        u3 = _mm_xor_si128(_mm_xor_si128(u3, _mm_slli_epi64(v3, 07)), v3);
        v0 = _mm_and_si128(_mm_xor_si128(u0, _mm_srli_epi64(u0, 14)), MASK1);
        v1 = _mm_and_si128(_mm_xor_si128(u1, _mm_srli_epi64(u1, 14)), MASK1);
        v2 = _mm_and_si128(_mm_xor_si128(u2, _mm_srli_epi64(u2, 14)), MASK1);
        v3 = _mm_and_si128(_mm_xor_si128(u3, _mm_srli_epi64(u3, 14)), MASK1);
        u0 = _mm_xor_si128(_mm_xor_si128(u0, _mm_slli_epi64(v0, 14)), v0);
        u1 = _mm_xor_si128(_mm_xor_si128(u1, _mm_slli_epi64(v1, 14)), v1);
        u2 = _mm_xor_si128(_mm_xor_si128(u2, _mm_slli_epi64(v2, 14)), v2);
        u3 = _mm_xor_si128(_mm_xor_si128(u3, _mm_slli_epi64(v3, 14)), v3);
        v0 = _mm_and_si128(_mm_xor_si128(u0, _mm_srli_epi64(u0, 28)), MASK2);
        v1 = _mm_and_si128(_mm_xor_si128(u1, _mm_srli_epi64(u1, 28)), MASK2);
        v2 = _mm_and_si128(_mm_xor_si128(u2, _mm_srli_epi64(u2, 28)), MASK2);
        v3 = _mm_and_si128(_mm_xor_si128(u3, _mm_srli_epi64(u3, 28)), MASK2);
        u0 = _mm_xor_si128(_mm_xor_si128(u0, _mm_slli_epi64(v0, 28)), v0);
        u1 = _mm_xor_si128(_mm_xor_si128(u1, _mm_slli_epi64(v1, 28)), v1);
        u2 = _mm_xor_si128(_mm_xor_si128(u2, _mm_slli_epi64(v2, 28)), v2);
        u3 = _mm_xor_si128(_mm_xor_si128(u3, _mm_slli_epi64(v3, 28)), v3);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 0], u0);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 1], u1);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 2], u2);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 3], u3);
    }
    if (i < size)
        bitshuf_untrans_bit_tail(out, in, size * 8, i);
}
#endif

#if defined(__AVX512VBMI__) && defined(__AVX512VL__) && defined(__GFNI__)
#define bitshuf_untrans_bit bitshuf_untrans_bit_avx512vbmi_gfni
#elif BITSHUF_USE_IFUNC
IMPLEMENT_IFUNC(bitshuf_untrans_bit_ifunc,
                (char* restrict out, const char* restrict in, size_t size)) {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("gfni"))
    {
        return bitshuf_untrans_bit_avx512vbmi_gfni;
    }
#if defined(__AVX512BW__) && defined(__AVX512VL__)
    return bitshuf_untrans_bit_avx512bw;
#else
    if (__builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl"))
        return bitshuf_untrans_bit_avx512bw;
#if defined(__AVX2__)
    return bitshuf_untrans_bit_avx2;
#else
    if (__builtin_cpu_supports("avx2"))
        return bitshuf_untrans_bit_avx2;
#if defined(__SSE2__)
    return bitshuf_untrans_bit_sse2;
#else
    if (__builtin_cpu_supports("sse2"))
        return bitshuf_untrans_bit_sse2;

    return bitshuf_untrans_bit;
#endif
#endif
#endif
}
#define bitshuf_untrans_bit bitshuf_untrans_bit_ifunc
#elif defined(__AVX512BW__) && defined(__AVX512VL__)
#define bitshuf_untrans_bit bitshuf_untrans_bit_avx512bw
#elif defined(__AVX2__)
#define bitshuf_untrans_bit bitshuf_untrans_bit_avx2
#elif defined(__SSE2__)
#define bitshuf_untrans_bit bitshuf_untrans_bit_sse2
#endif

#if defined(__SSE2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("sse2")
static void bitshuf_untrans_byte_sse2(char* restrict out,
                                      const char* restrict in,
                                      size_t size,
                                      size_t elem_size) {
    assert(size % 8 == 0);

    size_t j = 0;
    for (; j + 8 <= elem_size; j += 8) {
        for (size_t i = 0; i < size; i += 8) {
            const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[(j + 0) * size + i]);
            const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[(j + 1) * size + i]);
            const __m128i a2 = _mm_loadl_epi64((const __m128i*)&in[(j + 2) * size + i]);
            const __m128i a3 = _mm_loadl_epi64((const __m128i*)&in[(j + 3) * size + i]);
            const __m128i a4 = _mm_loadl_epi64((const __m128i*)&in[(j + 4) * size + i]);
            const __m128i a5 = _mm_loadl_epi64((const __m128i*)&in[(j + 5) * size + i]);
            const __m128i a6 = _mm_loadl_epi64((const __m128i*)&in[(j + 6) * size + i]);
            const __m128i a7 = _mm_loadl_epi64((const __m128i*)&in[(j + 7) * size + i]);
            __m128i u0 = _mm_unpacklo_epi8(a0, a1);
            __m128i u1 = _mm_unpacklo_epi8(a2, a3);
            __m128i u2 = _mm_unpacklo_epi8(a4, a5);
            __m128i u3 = _mm_unpacklo_epi8(a6, a7);
            const __m128i v0 = _mm_unpacklo_epi16(u0, u1);
            const __m128i v1 = _mm_unpackhi_epi16(u0, u1);
            const __m128i v2 = _mm_unpacklo_epi16(u2, u3);
            const __m128i v3 = _mm_unpackhi_epi16(u2, u3);
            u0 = _mm_unpacklo_epi32(v0, v2);
            u1 = _mm_unpackhi_epi32(v0, v2);
            u2 = _mm_unpacklo_epi32(v1, v3);
            u3 = _mm_unpackhi_epi32(v1, v3);
            _mm_storel_epi64((__m128i*)&out[(i + 0) * elem_size + j], u0);
            _mm_storel_epi64((__m128i*)&out[(i + 1) * elem_size + j], _mm_srli_si128(u0, 8));
            _mm_storel_epi64((__m128i*)&out[(i + 2) * elem_size + j], u1);
            _mm_storel_epi64((__m128i*)&out[(i + 3) * elem_size + j], _mm_srli_si128(u1, 8));
            _mm_storel_epi64((__m128i*)&out[(i + 4) * elem_size + j], u2);
            _mm_storel_epi64((__m128i*)&out[(i + 5) * elem_size + j], _mm_srli_si128(u2, 8));
            _mm_storel_epi64((__m128i*)&out[(i + 6) * elem_size + j], u3);
            _mm_storel_epi64((__m128i*)&out[(i + 7) * elem_size + j], _mm_srli_si128(u3, 8));
        }
    }
    if (j < elem_size) {
        const size_t j0 = (j + 0) * size;
        const size_t j1 = (j + 1) < elem_size ? (j + 1) * size : 1;
        const size_t j2 = (j + 2) % elem_size * size + (j + 2) / elem_size;
        const size_t j3 = (j + 3) % elem_size * size + (j + 3) / elem_size;
        const size_t j4 = (j + 4) % elem_size * size + (j + 4) / elem_size;
        const size_t j5 = (j + 5) % elem_size * size + (j + 5) / elem_size;
        const size_t j6 = (j + 6) % elem_size * size + (j + 6) / elem_size;
        const size_t j7 = (j + 7) % elem_size * size + (j + 7) / elem_size;
        for (size_t i = 0; i + 8 < size; i += 8) {
            const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[j0 + i]);
            const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[j1 + i]);
            const __m128i a2 = _mm_loadl_epi64((const __m128i*)&in[j2 + i]);
            const __m128i a3 = _mm_loadl_epi64((const __m128i*)&in[j3 + i]);
            const __m128i a4 = _mm_loadl_epi64((const __m128i*)&in[j4 + i]);
            const __m128i a5 = _mm_loadl_epi64((const __m128i*)&in[j5 + i]);
            const __m128i a6 = _mm_loadl_epi64((const __m128i*)&in[j6 + i]);
            const __m128i a7 = _mm_loadl_epi64((const __m128i*)&in[j7 + i]);
            __m128i u0 = _mm_unpacklo_epi8(a0, a1);
            __m128i u1 = _mm_unpacklo_epi8(a2, a3);
            __m128i u2 = _mm_unpacklo_epi8(a4, a5);
            __m128i u3 = _mm_unpacklo_epi8(a6, a7);
            const __m128i v0 = _mm_unpacklo_epi16(u0, u1);
            const __m128i v1 = _mm_unpackhi_epi16(u0, u1);
            const __m128i v2 = _mm_unpacklo_epi16(u2, u3);
            const __m128i v3 = _mm_unpackhi_epi16(u2, u3);
            u0 = _mm_unpacklo_epi32(v0, v2);
            u1 = _mm_unpackhi_epi32(v0, v2);
            u2 = _mm_unpacklo_epi32(v1, v3);
            u3 = _mm_unpackhi_epi32(v1, v3);
            _mm_storel_epi64((__m128i*)&out[(i + 0) * elem_size + j], u0);
            _mm_storel_epi64((__m128i*)&out[(i + 1) * elem_size + j], _mm_srli_si128(u0, 8));
            _mm_storel_epi64((__m128i*)&out[(i + 2) * elem_size + j], u1);
            _mm_storel_epi64((__m128i*)&out[(i + 3) * elem_size + j], _mm_srli_si128(u1, 8));
            _mm_storel_epi64((__m128i*)&out[(i + 4) * elem_size + j], u2);
            _mm_storel_epi64((__m128i*)&out[(i + 5) * elem_size + j], _mm_srli_si128(u2, 8));
            _mm_storel_epi64((__m128i*)&out[(i + 6) * elem_size + j], u3);
            _mm_storel_epi64((__m128i*)&out[(i + 7) * elem_size + j], _mm_srli_si128(u3, 8));
        }
        for (; j < elem_size; j++) {
            for (size_t i = size - 8; i < size; i++)
                out[i * elem_size + j] = in[j * size + i];
        }
    }
}
#endif

#if defined(__SSE2__)
#define bitshuf_untrans_byte bitshuf_untrans_byte_sse2
#elif BITSHUF_USE_IFUNC
IMPLEMENT_IFUNC(bitshuf_untrans_byte_ifunc,
                (char* restrict out, const char* restrict in, size_t size, size_t elem_size)) {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("sse2"))
        return bitshuf_untrans_byte_sse2;

    return bitshuf_untrans_byte;
}
#define bitshuf_untrans_byte bitshuf_untrans_byte_ifunc
#endif

#if defined(__AVX2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("avx2")
static void bitshuf_untrans_byte_2_avx2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);

    size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        const __m256i a0 = _mm256_loadu_si256((const __m256i*)&in[0 * size + i]);
        const __m256i a1 = _mm256_loadu_si256((const __m256i*)&in[1 * size + i]);
        const __m256i u0 = _mm256_permute4x64_epi64(a0, 0xd8);
        const __m256i u1 = _mm256_permute4x64_epi64(a1, 0xd8);
        const __m256i v0 = _mm256_unpacklo_epi8(u0, u1);
        const __m256i v1 = _mm256_unpackhi_epi8(u0, u1);
        _mm256_storeu_si256((__m256i*)&out[i * 2], v0);
        _mm256_storeu_si256((__m256i*)&out[i * 2 + 32], v1);
    }
    if (i + 16 <= size) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[1 * size + i]);
        const __m128i u0 = _mm_unpacklo_epi8(a0, a1);
        const __m128i u1 = _mm_unpackhi_epi8(a0, a1);
        _mm_storeu_si128((__m128i*)&out[i * 2], u0);
        _mm_storeu_si128((__m128i*)&out[i * 2 + 16], u1);
        i += 16;
    }
    if (i + 8 <= size) {
        const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[1 * size + i]);
        const __m128i u = _mm_unpacklo_epi8(a0, a1);
        _mm_storeu_si128((__m128i*)&out[i * 2], u);
    }
}
#endif

#if defined(__SSE2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("sse2")
static void bitshuf_untrans_byte_2_sse2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);

    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[1 * size + i]);
        const __m128i u0 = _mm_unpacklo_epi8(a0, a1);
        const __m128i u1 = _mm_unpackhi_epi8(a0, a1);
        _mm_storeu_si128((__m128i*)&out[i * 2], u0);
        _mm_storeu_si128((__m128i*)&out[i * 2 + 16], u1);
    }
    if (i + 8 <= size) {
        const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[1 * size + i]);
        const __m128i u = _mm_unpacklo_epi8(a0, a1);
        _mm_storeu_si128((__m128i*)&out[i * 2], u);
    }
}
#endif

#if defined(__AVX2__)
#define bitshuf_untrans_byte_2 bitshuf_untrans_byte_2_avx2
#elif BITSHUF_USE_IFUNC
IMPLEMENT_IFUNC(bitshuf_untrans_byte_2_ifunc,
                (char* restrict out, const char* restrict in, size_t size)) {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx2"))
        return bitshuf_untrans_byte_2_avx2;
#if defined(__SSE2__)
    return bitshuf_untrans_byte_2_sse2;
#else
    if (__builtin_cpu_supports("sse2"))
        return bitshuf_untrans_byte_2_sse2;

    return bitshuf_untrans_byte_2;
#endif
}
#define bitshuf_untrans_byte_2 bitshuf_untrans_byte_2_ifunc
#elif defined(__SSE2__)
#define bitshuf_untrans_byte_2 bitshuf_untrans_byte_2_sse2
#endif

#if defined(__AVX2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("avx2")
static void bitshuf_untrans_byte_4_avx2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);

    size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        const __m256i a0 = _mm256_loadu_si256((const __m256i*)&in[0 * size + i]);
        const __m256i a1 = _mm256_loadu_si256((const __m256i*)&in[1 * size + i]);
        const __m256i a2 = _mm256_loadu_si256((const __m256i*)&in[2 * size + i]);
        const __m256i a3 = _mm256_loadu_si256((const __m256i*)&in[3 * size + i]);
        __m256i u0 = _mm256_unpacklo_epi8(a0, a1);
        __m256i u1 = _mm256_unpackhi_epi8(a0, a1);
        __m256i u2 = _mm256_unpacklo_epi8(a2, a3);
        __m256i u3 = _mm256_unpackhi_epi8(a2, a3);
        const __m256i v0 = _mm256_unpacklo_epi16(u0, u2);
        const __m256i v1 = _mm256_unpackhi_epi16(u0, u2);
        const __m256i v2 = _mm256_unpacklo_epi16(u1, u3);
        const __m256i v3 = _mm256_unpackhi_epi16(u1, u3);
        u0 = _mm256_inserti128_si256(v0, _mm256_castsi256_si128(v1), 1);
        u1 = _mm256_inserti128_si256(v2, _mm256_castsi256_si128(v3), 1);
        u2 = _mm256_permute2x128_si256(v0, v1, 0x31);
        u3 = _mm256_permute2x128_si256(v2, v3, 0x31);
        _mm256_storeu_si256((__m256i*)&out[i * 4 + 32 * 0], u0);
        _mm256_storeu_si256((__m256i*)&out[i * 4 + 32 * 1], u1);
        _mm256_storeu_si256((__m256i*)&out[i * 4 + 32 * 2], u2);
        _mm256_storeu_si256((__m256i*)&out[i * 4 + 32 * 3], u3);
    }
    if (i + 16 <= size) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[1 * size + i]);
        const __m128i a2 = _mm_loadu_si128((const __m128i*)&in[2 * size + i]);
        const __m128i a3 = _mm_loadu_si128((const __m128i*)&in[3 * size + i]);
        const __m128i u0 = _mm_unpacklo_epi8(a0, a1);
        const __m128i u1 = _mm_unpackhi_epi8(a0, a1);
        const __m128i u2 = _mm_unpacklo_epi8(a2, a3);
        const __m128i u3 = _mm_unpackhi_epi8(a2, a3);
        const __m128i v0 = _mm_unpacklo_epi16(u0, u2);
        const __m128i v1 = _mm_unpackhi_epi16(u0, u2);
        const __m128i v2 = _mm_unpacklo_epi16(u1, u3);
        const __m128i v3 = _mm_unpackhi_epi16(u1, u3);
        _mm_storeu_si128((__m128i*)&out[i * 4 + 16 * 0], v0);
        _mm_storeu_si128((__m128i*)&out[i * 4 + 16 * 1], v1);
        _mm_storeu_si128((__m128i*)&out[i * 4 + 16 * 2], v2);
        _mm_storeu_si128((__m128i*)&out[i * 4 + 16 * 3], v3);
        i += 16;
    }
    if (i + 8 <= size) {
        const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[1 * size + i]);
        const __m128i a2 = _mm_loadl_epi64((const __m128i*)&in[2 * size + i]);
        const __m128i a3 = _mm_loadl_epi64((const __m128i*)&in[3 * size + i]);
        const __m128i u0 = _mm_unpacklo_epi8(a0, a1);
        const __m128i u1 = _mm_unpacklo_epi8(a2, a3);
        const __m128i v0 = _mm_unpacklo_epi16(u0, u1);
        const __m128i v1 = _mm_unpackhi_epi16(u0, u1);
        _mm_storeu_si128((__m128i*)&out[i * 4], v0);
        _mm_storeu_si128((__m128i*)&out[i * 4 + 16], v1);
    }
}
#endif

#if defined(__SSE2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("sse2")
static void bitshuf_untrans_byte_4_sse2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);

    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[1 * size + i]);
        const __m128i a2 = _mm_loadu_si128((const __m128i*)&in[2 * size + i]);
        const __m128i a3 = _mm_loadu_si128((const __m128i*)&in[3 * size + i]);
        const __m128i u0 = _mm_unpacklo_epi8(a0, a1);
        const __m128i u1 = _mm_unpackhi_epi8(a0, a1);
        const __m128i u2 = _mm_unpacklo_epi8(a2, a3);
        const __m128i u3 = _mm_unpackhi_epi8(a2, a3);
        const __m128i v0 = _mm_unpacklo_epi16(u0, u2);
        const __m128i v1 = _mm_unpackhi_epi16(u0, u2);
        const __m128i v2 = _mm_unpacklo_epi16(u1, u3);
        const __m128i v3 = _mm_unpackhi_epi16(u1, u3);
        _mm_storeu_si128((__m128i*)&out[i * 4 + 16 * 0], v0);
        _mm_storeu_si128((__m128i*)&out[i * 4 + 16 * 1], v1);
        _mm_storeu_si128((__m128i*)&out[i * 4 + 16 * 2], v2);
        _mm_storeu_si128((__m128i*)&out[i * 4 + 16 * 3], v3);
    }
    if (i + 8 <= size) {
        const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[1 * size + i]);
        const __m128i a2 = _mm_loadl_epi64((const __m128i*)&in[2 * size + i]);
        const __m128i a3 = _mm_loadl_epi64((const __m128i*)&in[3 * size + i]);
        const __m128i u0 = _mm_unpacklo_epi8(a0, a1);
        const __m128i u1 = _mm_unpacklo_epi8(a2, a3);
        const __m128i v0 = _mm_unpacklo_epi16(u0, u1);
        const __m128i v1 = _mm_unpackhi_epi16(u0, u1);
        _mm_storeu_si128((__m128i*)&out[i * 4], v0);
        _mm_storeu_si128((__m128i*)&out[i * 4 + 16], v1);
    }
}
#endif

#if defined(__AVX2__)
#define bitshuf_untrans_byte_4 bitshuf_untrans_byte_4_avx2
#elif BITSHUF_USE_IFUNC
IMPLEMENT_IFUNC(bitshuf_untrans_byte_4_ifunc,
                (char* restrict out, const char* restrict in, size_t size)) {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx2"))
        return bitshuf_untrans_byte_4_avx2;
#if defined(__SSE2__)
    return bitshuf_untrans_byte_4_sse2;
#else
    if (__builtin_cpu_supports("sse2"))
        return bitshuf_untrans_byte_4_sse2;

    return bitshuf_untrans_byte_4;
#endif
}
#define bitshuf_untrans_byte_4 bitshuf_untrans_byte_4_ifunc
#elif defined(__SSE2__)
#define bitshuf_untrans_byte_4 bitshuf_untrans_byte_4_sse2
#endif

#if defined(__SSE2__) || BITSHUF_USE_IFUNC
NO_INLINE
ATTRIBUTE_TARGET("sse2")
static void bitshuf_untrans_byte_8_sse2(char* restrict out, const char* restrict in, size_t size) {
    assert(size % 8 == 0);

    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadu_si128((const __m128i*)&in[1 * size + i]);
        const __m128i a2 = _mm_loadu_si128((const __m128i*)&in[2 * size + i]);
        const __m128i a3 = _mm_loadu_si128((const __m128i*)&in[3 * size + i]);
        const __m128i a4 = _mm_loadu_si128((const __m128i*)&in[4 * size + i]);
        const __m128i a5 = _mm_loadu_si128((const __m128i*)&in[5 * size + i]);
        const __m128i a6 = _mm_loadu_si128((const __m128i*)&in[6 * size + i]);
        const __m128i a7 = _mm_loadu_si128((const __m128i*)&in[7 * size + i]);
        __m128i u0 = _mm_unpacklo_epi8(a0, a1);
        __m128i u1 = _mm_unpackhi_epi8(a0, a1);
        __m128i u2 = _mm_unpacklo_epi8(a2, a3);
        __m128i u3 = _mm_unpackhi_epi8(a2, a3);
        __m128i u4 = _mm_unpacklo_epi8(a4, a5);
        __m128i u5 = _mm_unpackhi_epi8(a4, a5);
        __m128i u6 = _mm_unpacklo_epi8(a6, a7);
        __m128i u7 = _mm_unpackhi_epi8(a6, a7);
        const __m128i v0 = _mm_unpacklo_epi16(u0, u2);
        const __m128i v1 = _mm_unpackhi_epi16(u0, u2);
        const __m128i v2 = _mm_unpacklo_epi16(u1, u3);
        const __m128i v3 = _mm_unpackhi_epi16(u1, u3);
        const __m128i v4 = _mm_unpacklo_epi16(u4, u6);
        const __m128i v5 = _mm_unpackhi_epi16(u4, u6);
        const __m128i v6 = _mm_unpacklo_epi16(u5, u7);
        const __m128i v7 = _mm_unpackhi_epi16(u5, u7);
        u0 = _mm_unpacklo_epi32(v0, v4);
        u1 = _mm_unpackhi_epi32(v0, v4);
        u2 = _mm_unpacklo_epi32(v1, v5);
        u3 = _mm_unpackhi_epi32(v1, v5);
        u4 = _mm_unpacklo_epi32(v2, v6);
        u5 = _mm_unpackhi_epi32(v2, v6);
        u6 = _mm_unpacklo_epi32(v3, v7);
        u7 = _mm_unpackhi_epi32(v3, v7);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 0], u0);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 1], u1);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 2], u2);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 3], u3);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 4], u4);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 5], u5);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 6], u6);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 7], u7);
    }
    if (i + 8 <= size) {
        const __m128i a0 = _mm_loadl_epi64((const __m128i*)&in[0 * size + i]);
        const __m128i a1 = _mm_loadl_epi64((const __m128i*)&in[1 * size + i]);
        const __m128i a2 = _mm_loadl_epi64((const __m128i*)&in[2 * size + i]);
        const __m128i a3 = _mm_loadl_epi64((const __m128i*)&in[3 * size + i]);
        const __m128i a4 = _mm_loadl_epi64((const __m128i*)&in[4 * size + i]);
        const __m128i a5 = _mm_loadl_epi64((const __m128i*)&in[5 * size + i]);
        const __m128i a6 = _mm_loadl_epi64((const __m128i*)&in[6 * size + i]);
        const __m128i a7 = _mm_loadl_epi64((const __m128i*)&in[7 * size + i]);
        __m128i u0 = _mm_unpacklo_epi8(a0, a1);
        __m128i u1 = _mm_unpacklo_epi8(a2, a3);
        __m128i u2 = _mm_unpacklo_epi8(a4, a5);
        __m128i u3 = _mm_unpacklo_epi8(a6, a7);
        const __m128i v0 = _mm_unpacklo_epi16(u0, u1);
        const __m128i v1 = _mm_unpackhi_epi16(u0, u1);
        const __m128i v2 = _mm_unpacklo_epi16(u2, u3);
        const __m128i v3 = _mm_unpackhi_epi16(u2, u3);
        u0 = _mm_unpacklo_epi32(v0, v2);
        u1 = _mm_unpackhi_epi32(v0, v2);
        u2 = _mm_unpacklo_epi32(v1, v3);
        u3 = _mm_unpackhi_epi32(v1, v3);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 0], u0);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 1], u1);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 2], u2);
        _mm_storeu_si128((__m128i*)&out[i * 8 + 16 * 3], u3);
    }
}
#endif

#if defined(__SSE2__)
#define bitshuf_untrans_byte_8 bitshuf_untrans_byte_8_sse2
#elif BITSHUF_USE_IFUNC
IMPLEMENT_IFUNC(bitshuf_untrans_byte_8_ifunc,
                (char* restrict out, const char* restrict in, size_t size)) {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("sse2"))
        return bitshuf_untrans_byte_8_sse2;

    return bitshuf_untrans_byte_8;
}
#define bitshuf_untrans_byte_8 bitshuf_untrans_byte_8_ifunc
#endif

#endif

int bitshuf_encode_block(char* restrict out,
                         const char* restrict in,
                         char* restrict scratch,
                         size_t size,
                         size_t elem_size) {
    if (UNLIKELY(size & 7))
        return -1;

    if (elem_size == 1) {
        bitshuf_trans_bit(out, in, size);
    } else {
        if (UNLIKELY(!scratch && elem_size > 1))
            return -1;

        switch (elem_size) {
            case 2:
                bitshuf_trans_byte_2(scratch, in, size);
                break;
            case 4:
                bitshuf_trans_byte_4(scratch, in, size);
                break;
            case 8:
                bitshuf_trans_byte_8(scratch, in, size);
                break;
            default:
                bitshuf_trans_byte(scratch, in, size, elem_size);
                break;
        }
        for (size_t i = 0; i < elem_size; i++)
            bitshuf_trans_bit(&out[i * size], &scratch[i * size], size);
    }
    return 0;
}

int bitshuf_decode_block(char* restrict out,
                         const char* restrict in,
                         char* restrict scratch,
                         size_t size,
                         size_t elem_size) {
    if (UNLIKELY(size & 7))
        return -1;

    if (elem_size == 1) {
        bitshuf_untrans_bit(out, in, size);
    } else {
        if (UNLIKELY(!scratch && elem_size > 1))
            return -1;

        for (size_t i = 0; i < elem_size; i++)
            bitshuf_untrans_bit(&scratch[i * size], &in[i * size], size);

        switch (elem_size) {
            case 2:
                bitshuf_untrans_byte_2(out, scratch, size);
                break;
            case 4:
                bitshuf_untrans_byte_4(out, scratch, size);
                break;
            case 8:
                bitshuf_untrans_byte_8(out, scratch, size);
                break;
            default:
                bitshuf_untrans_byte(out, scratch, size, elem_size);
                break;
        }
    }
    return 0;
}

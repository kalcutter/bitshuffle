// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2023 Kal Conley
#include "src/bitshuffle.h"

#include <stddef.h>
#include <stdint.h>
#include <cassert>
#include <cstring>
#include <memory>

#include "benchmark/benchmark.h"
#include "bitshuffle/src/bitshuffle_internals.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4100) // unreferenced formal parameter
#pragma warning(disable : 4244) // conversion from 'type1' to 'type2', possible loss of data
#endif

template <int (*F)(char*, const char*, char*, const size_t, const size_t)>
static void BM_bitshuffle(benchmark::State& state) {
    const size_t elem_size = state.range(0);
    const size_t size_bytes =
        state.range(1) / (elem_size * BSHUF_BLOCKED_MULT) * (elem_size * BSHUF_BLOCKED_MULT);
    const size_t size = size_bytes / elem_size;
    if (size == 0) {
        state.SkipWithError("size is zero");
        return;
    }
    std::unique_ptr<char[]> out{new char[size_bytes]};
    std::unique_ptr<char[]> in{new char[size_bytes]};
    std::unique_ptr<char[]> scratch{new char[size_bytes]};
    for (auto _ : state) {
        benchmark::DoNotOptimize(out.get());
        const int r = F(&out[0], &in[0], &scratch[0], size, elem_size);
        assert(r == 0);
        (void)r;
        benchmark::ClobberMemory();
    }
}

int bm_memcpy(char* out, const char* in, char* scratch, size_t size, size_t elem_size) {
    std::memcpy(out, in, size * elem_size);
    return 0;
}

int bm_bshuf_trans_bit_elem(char* out,
                            const char* in,
                            char* scratch,
                            size_t size,
                            size_t elem_size) {
    const int64_t n = bshuf_trans_bit_elem(in, out, size, elem_size);
    return static_cast<size_t>(n) == size * elem_size ? 0 : -1;
}

int bm_bshuf_untrans_bit_elem(char* out,
                              const char* in,
                              char* scratch,
                              size_t size,
                              size_t elem_size) {
    const int64_t n = bshuf_untrans_bit_elem(in, out, size, elem_size);
    return static_cast<size_t>(n) == size * elem_size ? 0 : -1;
}

#define memcpy bm_memcpy
#define bshuf_trans_bit_elem bm_bshuf_trans_bit_elem
#define bshuf_untrans_bit_elem bm_bshuf_untrans_bit_elem
// clang-format off
#define ELEM_SIZES {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 63, 64, 65, 66}
#define SIZES {256, 1024, 4096, 8192, 16384, 1'000'000, 16'000'000, 256'000'000}
// clang-format on
BENCHMARK_TEMPLATE(BM_bitshuffle, memcpy)->ArgsProduct({{1}, SIZES});
BENCHMARK_TEMPLATE(BM_bitshuffle, bitshuf_encode_block)->ArgsProduct({ELEM_SIZES, SIZES});
BENCHMARK_TEMPLATE(BM_bitshuffle, bitshuf_decode_block)->ArgsProduct({ELEM_SIZES, SIZES});
BENCHMARK_TEMPLATE(BM_bitshuffle, bshuf_trans_bit_elem)->ArgsProduct({ELEM_SIZES, SIZES});
BENCHMARK_TEMPLATE(BM_bitshuffle, bshuf_untrans_bit_elem)->ArgsProduct({ELEM_SIZES, SIZES});

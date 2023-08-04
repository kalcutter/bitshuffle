// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2023 Kal Conley
#include "src/bitshuffle.h"

#include <stddef.h>
#include <stdint.h>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#define BSHUF_BLOCKED_MULT 8

template <typename T>
class unaligned_allocator {
public:
    using value_type = T;

    unaligned_allocator() = default;

    template <typename U>
    unaligned_allocator(unaligned_allocator<U> const&) noexcept {}

    value_type* allocate(size_t n) {
        return static_cast<value_type*>(::operator new((n + 1) * sizeof(value_type))) + 1;
    }

    void deallocate(value_type* p, size_t) noexcept { ::operator delete(p - 1); }
};

#define EXPECT_BITSHUF(ELEM_SIZE, ...)                                                        \
    [](const std::vector<uint8_t, unaligned_allocator<uint8_t>>& u,                           \
       const std::vector<uint8_t, unaligned_allocator<uint8_t>>& v, const size_t elem_size) { \
        assert(u.size() == v.size());                                                         \
        assert(u.size() % (BSHUF_BLOCKED_MULT * elem_size) == 0);                             \
        std::vector<uint8_t, unaligned_allocator<uint8_t>> buf(u.size());                     \
        std::vector<uint8_t, unaligned_allocator<uint8_t>> scratch(u.size());                 \
        {                                                                                     \
            std::memset(buf.data(), 0xcc, buf.size());                                        \
            std::memset(scratch.data(), 0xcc, scratch.size());                                \
            const int r = bitshuf_encode_block(                                               \
                reinterpret_cast<char*>(buf.data()), reinterpret_cast<const char*>(u.data()), \
                reinterpret_cast<char*>(scratch.data()), buf.size() / elem_size, elem_size);  \
            ASSERT_EQ(r, 0);                                                                  \
            if (std::memcmp(buf.data(), v.data(), buf.size()) != 0) {                         \
                ASSERT_THAT(buf, ::testing::ElementsAreArray(v));                             \
            }                                                                                 \
        }                                                                                     \
        {                                                                                     \
            std::memset(buf.data(), 0xcc, buf.size());                                        \
            std::memset(scratch.data(), 0xcc, scratch.size());                                \
            const int r = bitshuf_decode_block(                                               \
                reinterpret_cast<char*>(buf.data()), reinterpret_cast<const char*>(v.data()), \
                reinterpret_cast<char*>(scratch.data()), buf.size() / elem_size, elem_size);  \
            ASSERT_EQ(r, 0);                                                                  \
            if (std::memcmp(buf.data(), u.data(), buf.size()) != 0) {                         \
                ASSERT_THAT(buf, ::testing::ElementsAreArray(u));                             \
            }                                                                                 \
        }                                                                                     \
    }(__VA_ARGS__, ELEM_SIZE)

TEST(Bitshuffle, ScratchNull) {
    char out[8], in[8];
    std::memset(in, 0, sizeof(in));
    for (size_t elem_size : {0, 1}) {
        EXPECT_EQ(bitshuf_encode_block(out, in, NULL, 8, elem_size), 0);
        EXPECT_EQ(bitshuf_decode_block(out, in, NULL, 8, elem_size), 0);
    }
}

TEST(Bitshuffle, ScratchNullError) {
    char out[1], in[1];
    for (size_t elem_size = 2; elem_size <= 257; elem_size++) {
        EXPECT_EQ(bitshuf_encode_block(out, in, NULL, 8, elem_size), -1);
        EXPECT_EQ(bitshuf_decode_block(out, in, NULL, 8, elem_size), -1);
    }
}

TEST(Bitshuffle, SizeZero) {
    char out[1], in[1], scratch[1];
    for (size_t elem_size = 0; elem_size <= 257; elem_size++) {
        EXPECT_EQ(bitshuf_encode_block(out, in, scratch, 0, elem_size), 0);
        EXPECT_EQ(bitshuf_decode_block(out, in, scratch, 0, elem_size), 0);
    }
}

TEST(Bitshuffle, ErrorIfSizeNotMultipleOf8) {
    char out[1], in[1], scratch[1];
    for (size_t size : {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17}) {
        EXPECT_EQ(bitshuf_encode_block(out, in, scratch, size, 1), -1);
        EXPECT_EQ(bitshuf_decode_block(out, in, scratch, size, 1), -1);
    }
}

TEST(Bitshuffle, ElemSizeZero) {
    char out[1], in[1], scratch[1];
    for (size_t size = 0; size <= 256; size += BSHUF_BLOCKED_MULT) {
        EXPECT_EQ(bitshuf_encode_block(out, in, scratch, size, 0), 0);
        EXPECT_EQ(bitshuf_decode_block(out, in, scratch, size, 0), 0);
    }
}

TEST(Bitshuffle, OneBitSetExhaustive) {
    const size_t N = 120;
    static_assert(N % BSHUF_BLOCKED_MULT == 0, "");

    for (size_t size = 0; size <= N; size += BSHUF_BLOCKED_MULT) {
        for (size_t elem_size = 0; elem_size <= 17; elem_size++) {
            std::vector<uint8_t, unaligned_allocator<uint8_t>> u(size * elem_size);
            std::vector<uint8_t, unaligned_allocator<uint8_t>> v(size * elem_size);
            for (size_t i = 0; i < size; i++) {
                for (size_t b = 0; b < elem_size * 8; b++) {
                    u.assign(u.size(), 0);
                    v.assign(v.size(), 0);
                    u[(i * elem_size) + b / 8] = 1U << (b % 8);
                    v[(b * size + i) / 8] = 1U << (i % 8);
                    EXPECT_BITSHUF(elem_size, u, v);
                }
            }
        }
    }
}

TEST(Bitshuffle, OneBitZeroExhaustive) {
    const size_t N = 120;
    static_assert(N % BSHUF_BLOCKED_MULT == 0, "");

    for (size_t size = 0; size <= N; size += BSHUF_BLOCKED_MULT) {
        for (size_t elem_size = 0; elem_size <= 17; elem_size++) {
            std::vector<uint8_t, unaligned_allocator<uint8_t>> u(size * elem_size);
            std::vector<uint8_t, unaligned_allocator<uint8_t>> v(size * elem_size);
            for (size_t i = 0; i < size; i++) {
                for (size_t b = 0; b < elem_size * 8; b++) {
                    u.assign(u.size(), 0xff);
                    v.assign(v.size(), 0xff);
                    u[(i * elem_size) + b / 8] = ~(1U << (b % 8));
                    v[(b * size + i) / 8] = ~(1U << (i % 8));
                    EXPECT_BITSHUF(elem_size, u, v);
                }
            }
        }
    }
}

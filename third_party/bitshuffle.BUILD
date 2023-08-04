load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "bitshuffle_core",
    srcs = [
        "src/bitshuffle_core.c",
        "src/iochain.c",
    ],
    hdrs = [
        "src/bitshuffle_core.h",
        "src/bitshuffle_internals.h",
        "src/iochain.h",
    ],
)

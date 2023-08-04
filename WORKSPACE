load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "benchmark",
    sha256 = "6430e4092653380d9dc4ccb45a1e2dc9259d581f4866dc0759713126056bc1d7",
    strip_prefix = "benchmark-1.7.1",
    url = "https://github.com/google/benchmark/archive/refs/tags/v1.7.1.tar.gz",
)

http_archive(
    name = "bitshuffle",
    build_file = "//third_party:bitshuffle.BUILD",
    sha256 = "2631aaa5d4c24e51415c7b1827d4f9dcf505ad8db03738210da9ce6dab8f5870",
    strip_prefix = "bitshuffle-0.5.1",
    url = "https://github.com/kiyo-masui/bitshuffle/archive/refs/tags/0.5.1.tar.gz",
)

http_archive(
    name = "googletest",
    sha256 = "ad7fdba11ea011c1d925b3289cf4af2c66a352e18d4c7264392fead75e919363",
    strip_prefix = "googletest-1.13.0",
    url = "https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz",
)

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

load("//bazel:otel_cc_benchmark.bzl", "otel_cc_benchmark")

cc_test(
    name = "baggage_test",
    srcs = [
        "baggage_test.cc",
    ],
    tags = [
        "api",
        "test",
    ],
    deps = [
        "//api",
        "@com_google_googletest//:gtest_main",
    ],
)

otel_cc_benchmark(
    name = "baggage_benchmark",
    srcs = ["baggage_benchmark.cc"],
    tags = [
        "api",
        "benchmark",
        "test",
    ],
    deps = ["//api"],
)

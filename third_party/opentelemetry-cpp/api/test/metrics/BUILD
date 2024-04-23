# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

load("//bazel:otel_cc_benchmark.bzl", "otel_cc_benchmark")

cc_test(
    name = "noop_sync_instrument_test",
    srcs = [
        "noop_sync_instrument_test.cc",
    ],
    tags = [
        "metrics",
        "test",
    ],
    deps = [
        "//api",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "meter_provider_test",
    srcs = [
        "meter_provider_test.cc",
    ],
    tags = [
        "metrics",
        "test",
    ],
    deps = [
        "//api",
        "@com_google_googletest//:gtest_main",
    ],
)

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

load("//bazel:otel_cc_benchmark.bzl", "otel_cc_benchmark")

cc_test(
    name = "provider_test",
    srcs = [
        "provider_test.cc",
    ],
    tags = [
        "api",
        "logs",
        "test",
    ],
    deps = [
        "//api",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "logger_test",
    srcs = [
        "logger_test.cc",
    ],
    tags = [
        "api",
        "logs",
        "test",
    ],
    deps = [
        "//api",
        "@com_google_googletest//:gtest_main",
    ],
)

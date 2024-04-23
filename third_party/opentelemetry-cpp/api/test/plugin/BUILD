# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

cc_test(
    name = "dynamic_load_test",
    srcs = [
        "dynamic_load_test.cc",
    ],
    linkopts = [
        "-ldl",
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

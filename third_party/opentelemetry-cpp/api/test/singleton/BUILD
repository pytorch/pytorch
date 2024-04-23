# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

DEFAULT_WIN_COPTS = [
]

# gcc and clang, assumed to be used on this platform
DEFAULT_NOWIN_COPTS = [
    "-fvisibility=default",
]

HIDDEN_WIN_COPTS = [
]

# gcc and clang, assumed to be used on this platform
HIDDEN_NOWIN_COPTS = [
    "-fvisibility=hidden",
]

cc_library(
    name = "component_a",
    srcs = [
        "component_a.cc",
    ],
    hdrs = [
        "component_a.h",
    ],
    linkstatic = True,
    deps = [
        "//api",
    ],
)

cc_library(
    name = "component_b",
    srcs = [
        "component_b.cc",
    ],
    hdrs = [
        "component_b.h",
    ],
    linkstatic = True,
    deps = [
        "//api",
    ],
)

cc_library(
    name = "component_c",
    srcs = [
        "component_c.cc",
    ],
    hdrs = [
        "component_c.h",
    ],
    copts = select({
        "//bazel:windows": DEFAULT_WIN_COPTS,
        "//conditions:default": DEFAULT_NOWIN_COPTS,
    }),
    linkstatic = False,
    deps = [
        "//api",
    ],
)

cc_library(
    name = "component_d",
    srcs = [
        "component_d.cc",
    ],
    hdrs = [
        "component_d.h",
    ],
    copts = select({
        "//bazel:windows": HIDDEN_WIN_COPTS,
        "//conditions:default": HIDDEN_NOWIN_COPTS,
    }),
    linkstatic = False,
    deps = [
        "//api",
    ],
)

cc_library(
    name = "component_e",
    srcs = [
        "component_e.cc",
    ],
    hdrs = [
        "component_e.h",
    ],
    copts = select({
        "//bazel:windows": DEFAULT_WIN_COPTS,
        "//conditions:default": DEFAULT_NOWIN_COPTS,
    }),
    linkstatic = False,
    deps = [
        "//api",
    ],
)

cc_library(
    name = "component_f",
    srcs = [
        "component_f.cc",
    ],
    hdrs = [
        "component_f.h",
    ],
    copts = select({
        "//bazel:windows": HIDDEN_WIN_COPTS,
        "//conditions:default": HIDDEN_NOWIN_COPTS,
    }),
    linkstatic = False,
    deps = [
        "//api",
    ],
)

# no cc_shared_library in bazel 4.2
cc_binary(
    name = "component_g",
    srcs = [
        "component_g.cc",
    ],
    copts = select({
        "//bazel:windows": DEFAULT_WIN_COPTS,
        "//conditions:default": DEFAULT_NOWIN_COPTS,
    }),
    linkshared = True,
    deps = [
        "//api",
    ],
)

# no cc_shared_library in bazel 4.2
cc_binary(
    name = "component_h",
    srcs = [
        "component_h.cc",
    ],
    copts = select({
        "//bazel:windows": HIDDEN_WIN_COPTS,
        "//conditions:default": HIDDEN_NOWIN_COPTS,
    }),
    linkshared = True,
    deps = [
        "//api",
    ],
)

#
# To build this test alone:
# - bazel build //api/test/singleton:singleton_test
# - bazel build //api/test/singleton:component_g
# - bazel build //api/test/singleton:component_h
#
# Note that singleton_test does not depend on
# component_g and component_h, on purpose.
#
# To run this test:
# bazel test //api/test/singleton:singleton_test
#

cc_test(
    name = "singleton_test",
    srcs = [
        "singleton_test.cc",
    ],
    defines = ["BAZEL_BUILD"],
    linkopts = [
        "-ldl",
    ],
    linkstatic = False,
    tags = [
        "api",
        "test",
    ],
    deps = [
        "component_a",
        "component_b",
        "component_c",
        "component_d",
        "component_e",
        "component_f",
        "//api",
        "@com_google_googletest//:gtest_main",
    ],
)

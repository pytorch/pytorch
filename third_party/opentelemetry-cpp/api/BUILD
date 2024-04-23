# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "string_flag")

package(default_visibility = ["//visibility:public"])

bool_flag(
    name = "with_abseil",
    build_setting_default = False,
)

CPP_STDLIBS = [
    "none",
    "best",
    "2014",
    "2017",
    "2020",
    "2023",
]

string_flag(
    name = "with_cxx_stdlib",
    build_setting_default = "best",
    values = CPP_STDLIBS,
)

cc_library(
    name = "api",
    hdrs = glob(["include/**/*.h"]),
    defines = select({
        ":with_external_abseil": ["HAVE_ABSEIL"],
        "//conditions:default": [],
    }) + select({
        ":set_cxx_stdlib_none": [],
        ### automatic selection
        ":set_cxx_stdlib_best": ["OPENTELEMETRY_STL_VERSION=(__cplusplus/100)"],
        # See https://learn.microsoft.com/en-us/cpp/build/reference/zc-cplusplus
        ":set_cxx_stdlib_best_and_msvc": ["OPENTELEMETRY_STL_VERSION=(_MSVC_LANG/100)"],
        ### manual selection
        ":set_cxx_stdlib_2014": ["OPENTELEMETRY_STL_VERSION=2014"],
        ":set_cxx_stdlib_2017": ["OPENTELEMETRY_STL_VERSION=2017"],
        ":set_cxx_stdlib_2020": ["OPENTELEMETRY_STL_VERSION=2020"],
        ":set_cxx_stdlib_2023": ["OPENTELEMETRY_STL_VERSION=2023"],
        "//conditions:default": [],
    }),
    strip_include_prefix = "include",
    tags = ["api"],
    deps = select({
        ":with_external_abseil": [
            "@com_google_absl//absl/base",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/types:variant",
        ],
        "//conditions:default": [],
    }),
)

config_setting(
    name = "with_external_abseil",
    flag_values = {":with_abseil": "true"},
)

[config_setting(
    name = "set_cxx_stdlib_%s" % v,
    flag_values = {":with_cxx_stdlib": v},
) for v in CPP_STDLIBS]

config_setting(
    name = "set_cxx_stdlib_best_and_msvc",
    constraint_values = ["@bazel_tools//tools/cpp:msvc"],
    flag_values = {":with_cxx_stdlib": "best"},
)

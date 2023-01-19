# Copyright 2018 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Go rules
load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

# Python rules
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "legacy_fields_migrator",
    srcs = ["legacy_fields_migrator.py"],
    python_version = "PY3",
    deps = [
        ":legacy_fields_migration_lib",
        "//third_party/com/github/bazelbuild/bazel/src/main/protobuf:crosstool_config_py_pb2",
        "@io_abseil_py//absl:app",
        "@io_abseil_py//absl/flags",
    ],
)

py_library(
    name = "legacy_fields_migration_lib",
    srcs = ["legacy_fields_migration_lib.py"],
    deps = [
        "//third_party/com/github/bazelbuild/bazel/src/main/protobuf:crosstool_config_py_pb2",
    ],
)

py_test(
    name = "legacy_fields_migration_lib_test",
    srcs = ["legacy_fields_migration_lib_test.py"],
    python_version = "PY3",
    deps = [
        ":legacy_fields_migration_lib",
        "//third_party/com/github/bazelbuild/bazel/src/main/protobuf:crosstool_config_py_pb2",
    ],
)

py_binary(
    name = "crosstool_query",
    srcs = ["crosstool_query.py"],
    python_version = "PY3",
    deps = [
        "//third_party/com/github/bazelbuild/bazel/src/main/protobuf:crosstool_config_py_pb2",
        "@io_abseil_py//absl:app",
        "@io_abseil_py//absl/flags",
    ],
)

py_binary(
    name = "ctoolchain_comparator",
    srcs = ["ctoolchain_comparator.py"],
    python_version = "PY3",
    deps = [
        ":ctoolchain_comparator_lib",
        "//third_party/com/github/bazelbuild/bazel/src/main/protobuf:crosstool_config_py_pb2",
        "@io_abseil_py//absl:app",
        "@io_abseil_py//absl/flags",
    ],
)

py_library(
    name = "ctoolchain_comparator_lib",
    srcs = ["ctoolchain_comparator_lib.py"],
    deps = [
        "//third_party/com/github/bazelbuild/bazel/src/main/protobuf:crosstool_config_py_pb2",
    ],
)

py_test(
    name = "ctoolchain_comparator_lib_test",
    srcs = ["ctoolchain_comparator_lib_test.py"],
    python_version = "PY3",
    deps = [
        ":ctoolchain_comparator_lib",
        "//third_party/com/github/bazelbuild/bazel/src/main/protobuf:crosstool_config_py_pb2",
        "@py_mock//py/mock",
    ],
)

go_binary(
    name = "convert_crosstool_to_starlark",
    srcs = ["convert_crosstool_to_starlark.go"],
    deps = [
        ":crosstooltostarlarklib",
        "//third_party/com/github/bazelbuild/bazel/src/main/protobuf:crosstool_config_go_proto",
        "@com_github_golang_protobuf//proto:go_default_library",
    ],
)

go_library(
    name = "crosstooltostarlarklib",
    srcs = ["crosstool_to_starlark_lib.go"],
    importpath = "tools/migration/crosstooltostarlarklib",
    deps = ["//third_party/com/github/bazelbuild/bazel/src/main/protobuf:crosstool_config_go_proto"],
)

go_test(
    name = "crosstooltostarlarklib_test",
    size = "small",
    srcs = ["crosstool_to_starlark_lib_test.go"],
    embed = [":crosstooltostarlarklib"],
    deps = [
        "//third_party/com/github/bazelbuild/bazel/src/main/protobuf:crosstool_config_go_proto",
        "@com_github_golang_protobuf//proto:go_default_library",
    ],
)

filegroup(
    name = "bazel_osx_p4deps",
    srcs = [
        "BUILD",
        "ctoolchain_compare.bzl",
    ],
)

exports_files(["ctoolchain_compare.bzl"])

bzl_library(
    name = "ctoolchain_compare_bzl",
    srcs = ["ctoolchain_compare.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "cc_toolchain_config_comparator_bzl",
    srcs = ["cc_toolchain_config_comparator.bzl"],
    visibility = ["//visibility:private"],
)

# Copyright 2019 The Bazel Authors. All rights reserved.
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

# Example showing how to create a custom Starlark rule that rules_cc can depend on

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//examples/my_c_archive:my_c_archive.bzl", "my_c_archive")
load("//examples/my_c_compile:my_c_compile.bzl", "my_c_compile")

licenses(["notice"])

cc_binary(
    name = "main",
    srcs = ["main.c"],
    deps = [":archive"],
)

my_c_archive(
    name = "archive",
    object = ":object",
    deps = [":bar"],
)

my_c_compile(
    name = "object",
    src = "foo.c",
)

cc_library(
    name = "bar",
    srcs = ["bar.c"],
)

bzl_library(
    name = "my_c_archive_bzl",
    srcs = ["my_c_archive.bzl"],
    visibility = ["//visibility:private"],
)

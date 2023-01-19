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

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

# A collection of examples showing the usage of rules_cc
licenses(["notice"])

bool_flag(
    name = "incompatible_link_once",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "experimental_debug",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "experimental_cc_shared_library_bzl",
    srcs = ["experimental_cc_shared_library.bzl"],
    visibility = ["//visibility:private"],
)

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

"""Setup for rules_cc tests and tools."""

load("@bazel_federation//setup:rules_pkg.bzl", "rules_pkg_setup")
load("@bazel_federation//setup:rules_python.bzl", "rules_python_setup")

# TODO(fweikert): Add setup.bzl file for skylib to the federation and load it instead of workspace.bzl
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

# TODO(fweikert): Also load rules_go's setup.bzl file from the federation once it exists
load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

def rules_cc_internal_setup():
    """Setup of dependencies of tests and development-only tools used in rules_cc repository."""
    bazel_skylib_workspace()
    go_rules_dependencies()
    go_register_toolchains()
    rules_pkg_setup()
    rules_python_setup()

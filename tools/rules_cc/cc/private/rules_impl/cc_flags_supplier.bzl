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
"""Rule that provides the CC_FLAGS Make variable."""

load("//cc:action_names.bzl", "CC_FLAGS_MAKE_VARIABLE_ACTION_NAME")
load("//cc:find_cc_toolchain.bzl", "find_cc_toolchain")
load("//cc/private/rules_impl:cc_flags_supplier_lib.bzl", "build_cc_flags")

def _cc_flags_supplier_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    cc_flags = build_cc_flags(ctx, cc_toolchain, CC_FLAGS_MAKE_VARIABLE_ACTION_NAME)
    variables = platform_common.TemplateVariableInfo({
        "CC_FLAGS": cc_flags,
    })
    return [variables]

cc_flags_supplier = rule(
    implementation = _cc_flags_supplier_impl,
    attrs = {
        "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    incompatible_use_toolchain_transition = True,
    fragments = ["cpp"],
)

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

"""Example showing how to get CcToolchainInfo in a custom rule."""

load("@rules_cc//cc:toolchain_utils.bzl", "find_cpp_toolchain")

def _write_cc_toolchain_cpu_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    output = ctx.actions.declare_file(ctx.label.name + "_cpu")
    ctx.actions.write(output, cc_toolchain.cpu)
    return [DefaultInfo(files = depset([output]))]

# This rule does nothing, just writes the target_cpu from the cc_toolchain used for this build.
write_cc_toolchain_cpu = rule(
    implementation = _write_cc_toolchain_cpu_impl,
    attrs = {
        "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    incompatible_use_toolchain_transition = True,
)

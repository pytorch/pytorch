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
"""Library of functions that provide the CC_FLAGS Make variable."""

# This should match the logic in CcCommon.computeCcFlags:
def build_cc_flags(ctx, cc_toolchain, action_name):
    """Determine the value for CC_FLAGS based on the given toolchain.

    Args:
      ctx: The rule context.
      cc_toolchain: CcToolchainInfo instance.
      action_name: Name of the action.
    Returns:
      string containing flags separated by a space.
    """

    # Get default cc flags from toolchain's make_variables.
    legacy_cc_flags = cc_common.legacy_cc_flags_make_variable_do_not_use(
        cc_toolchain = cc_toolchain,
    )

    # Determine the sysroot flag.
    sysroot_cc_flags = _from_sysroot(cc_toolchain)

    # Flags from feature config.
    feature_config_cc_flags = _from_features(ctx, cc_toolchain, action_name)

    # Combine the different sources, but only add the sysroot flag if nothing
    # else adds sysroot.
    # If added, it must appear before the feature config flags.
    cc_flags = []
    if legacy_cc_flags:
        cc_flags.append(legacy_cc_flags)
    if sysroot_cc_flags and not _contains_sysroot(feature_config_cc_flags):
        cc_flags.append(sysroot_cc_flags)
    cc_flags.extend(feature_config_cc_flags)

    return " ".join(cc_flags)

def _contains_sysroot(flags):
    for flag in flags:
        if "--sysroot=" in flag:
            return True
    return False

def _from_sysroot(cc_toolchain):
    sysroot = cc_toolchain.sysroot
    if sysroot:
        return "--sysroot=%s" % sysroot
    else:
        return None

def _from_features(ctx, cc_toolchain, action_name):
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    variables = cc_common.empty_variables()

    cc_flags = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = action_name,
        variables = variables,
    )
    return cc_flags

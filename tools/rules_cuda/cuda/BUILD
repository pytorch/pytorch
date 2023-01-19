load(":defs.bzl", "cuda_targets")
load(
    "//private:build.bzl",
    "cuda_targets_flag",
    "cuda_toolchain_info",
    "detect_cuda_toolchain",
    "report_error",
)
load("@bazel_skylib//lib:selects.bzl", "selects")
load(
    "@bazel_skylib//rules:common_settings.bzl",
    "bool_flag",
    "string_flag",
    "string_list_flag",
)
load("@local_cuda//:defs.bzl", "if_local_cuda")

package(default_visibility = ["//visibility:public"])

# Command line flag to set ":is_cuda_enabled" config setting.
#
# Set with --@rules_cuda//cuda:enable_cuda
bool_flag(
    name = "enable_cuda",
    build_setting_default = False,
)

# This config setting can be used in select() statements to e.g. conditionally
# depend on cuda_library() targets or conditionally mark targets compatible
# (see 'requires_cuda_enabled()' macro in @rules_cuda//cuda:defs.bzl).
#
# It is not used in this package, but is rather intended as a central switch for
# different unrelated users.
#
# Set with --@rules_cuda//cuda:enable_cuda
config_setting(
    name = "is_cuda_enabled",
    flag_values = {":enable_cuda": "True"},
)

# Command line flag to specify the list of CUDA compute architectures to
# compile for. For details, please consult the '--cuda-gpu-arch' clang flag.
#
# Provides CudaTargetsInfo of the list of cuda_targets to build.
#
# Example usage: --@rules_cuda//cuda:cuda_targets=sm_60,sm_70
cuda_targets_flag(
    name = "cuda_targets",
    build_setting_default = ["sm_52"],
)

# Config setting whether building cuda_target (sm_xy) has been requested.
[
    config_setting(
        name = "cuda_target_%s_enabled" % cuda_target,
        flag_values = {":cuda_targets": cuda_target},
    )
    for cuda_target in cuda_targets
]

# Config setting whether building any cuda_targets has been requested.
#
# Do not use as a condition whether your target should support CUDA, use
# :is_cuda_enabled instead. The intended use of this config setting is to verify
# that at least one target has been requested. See :requires_cuda_targets for
# an example.
selects.config_setting_group(
    name = "has_cuda_targets_any",
    match_any = [
        "cuda_target_%s_enabled" % cuda_target
        for cuda_target in cuda_targets
    ],
)

# Command line flag to select compiler for cuda_library() code.
string_flag(
    name = "compiler",
    build_setting_default = "nvcc",
    values = [
        "nvcc",
        "clang",
    ],
)

# Command line flag for copts to add to cuda_library() compile command.
string_list_flag(
    name = "copts",
    build_setting_default = [],
)

# Command line flag to specify the CUDA runtime. Use this target as CUDA
# runtime dependency.
#
# This target is implicitly added as a dependency to cuda_library() targets.
#
# Example usage: --@rules_cuda//cuda:cuda_runtime=@local_cuda//:cuda_runtime
label_flag(
    name = "cuda_runtime",
    build_setting_default = if_local_cuda(
        "@local_cuda//:cuda_runtime_static",
        ":empty",
    ),
)

# This target fails to execute and reports an error message that the @local_cuda
# repository does not provide the CUDA toolkit. The target passes analysis so
# that it doesn't break e.g. bazel query.
report_error(
    name = "no_cuda_toolkit_error",
    out = "no_cuda_toolkit_error.h",
    message = (
        "The @local_cuda repository is empty because the CUDA_PATH " +
        "environment variable does not point to a CUDA toolkit directory."
    ),
    tags = ["manual"],
)

# This target fails to execute and reports an error message that the toolchain
# does not support CUDA. The target passes analysis so that it doesn't break
# e.g. bazel query.
report_error(
    name = "unsupported_cuda_toolchain_error",
    out = "unsupported_cuda_toolchain_error.h",
    message = "The current cc_toolchain does not support feature cuda.",
    tags = ["manual"],
)

# Target to pass through a cc_toolchain_config rule and use as first argument
# to cuda_toolchain_config() to retrieve the 'cuda' feature configuration.
cuda_toolchain_info(
    name = "cuda_toolchain_info",
)

# Detects whether the current cc_toolchain supports feature 'cuda'.
detect_cuda_toolchain(
    name = "detect_cuda_toolchain",
    visibility = ["//visibility:private"],
)

# Config setting whether the current cc_toolchain supports feature 'cuda'.
config_setting(
    name = "cuda_toolchain_detected",
    flag_values = {":detect_cuda_toolchain": "True"},
)

# Empty cc_library to use as cuda_runtime when @local_cuda has no targets.
cc_library(
    name = "empty",
    visibility = ["//visibility:private"],
)

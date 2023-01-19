"""Private code for CUDA rules."""

load("//cuda:defs.bzl", "CudaTargetsInfo", "cuda_targets")
load("//cuda:toolchain.bzl", "CudaToolchainInfo")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@local_cuda//:defs.bzl", "if_local_cuda")

def _cuda_targets_flag_impl(ctx):
    for cuda_target in ctx.build_setting_value:
        if cuda_target not in cuda_targets:
            fail("%s is not a supported %s value." % (cuda_target, ctx.label))
    return CudaTargetsInfo(cuda_targets = ctx.build_setting_value)

cuda_targets_flag = rule(
    implementation = _cuda_targets_flag_impl,
    build_setting = config.string_list(flag = True),
    provides = [CudaTargetsInfo],
)

def _detect_cuda_toolchain_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ["cuda"],
    )
    is_enabled = cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "cuda",
    )
    return [config_common.FeatureFlagInfo(value = str(is_enabled))]

# Rule providing whether the current cc_toolchain supports feature 'cuda'.
detect_cuda_toolchain = rule(
    implementation = _detect_cuda_toolchain_impl,
    attrs = {
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    incompatible_use_toolchain_transition = True,
    fragments = ["cpp"],
)

def _report_error_impl(ctx):
    ctx.actions.run_shell(
        outputs = [ctx.outputs.out],
        progress_message = "\n%s\n" % ctx.attr.message,
        command = "false \"%s\"" % ctx.attr.message,
    )

# Rule which passes anlysis phase, but fails during execution with message.
report_error = rule(
    implementation = _report_error_impl,
    attrs = {
        "message": attr.string(),
        "out": attr.output(mandatory = True),
    },
)

def _cuda_toolchain_info_impl(ctx):
    return [
        DefaultInfo(files = depset([ctx.file._nvcc] if ctx.file._nvcc else [])),
        CudaToolchainInfo(
            nvcc = ctx.file._nvcc,
            compiler = ctx.attr._compiler[BuildSettingInfo].value,
            cuda_targets = ctx.attr._cuda_targets[CudaTargetsInfo].cuda_targets,
            copts = ctx.attr._copts[BuildSettingInfo].value,
        ),
    ]

# A rule that encapsulates the information to pass to cuda_toolchain_config.
# Specifically, it combines //cuda:cuda_targets, @local_cuda//:cuda/bin/nvcc
# and //cuda:compiler.
cuda_toolchain_info = rule(
    implementation = _cuda_toolchain_info_impl,
    attrs = {
        "_cuda_targets": attr.label(
            default = Label("//cuda:cuda_targets"),
        ),
        "_nvcc": attr.label(
            default = if_local_cuda(Label("@local_cuda//:cuda/bin/nvcc"), None),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
        "_compiler": attr.label(default = Label("//cuda:compiler")),
        "_copts": attr.label(default = Label("//cuda:copts")),
    },
    provides = [CudaToolchainInfo],
)

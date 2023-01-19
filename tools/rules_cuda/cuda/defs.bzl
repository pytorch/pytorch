"""cuda_library() macro wrapping a cc_library() target."""

load("@local_cuda//:defs.bzl", "if_local_cuda")

cuda_targets = [
    "sm_30",
    "sm_32",
    "sm_35",
    "sm_37",
    "sm_50",
    "sm_52",
    "sm_53",
    "sm_60",
    "sm_61",
    "sm_62",
    "sm_70",
    "sm_72",
    "sm_75",
    "sm_80",
]

CudaTargetsInfo = provider(
    "Provides a list of CUDA targets to compile for.",
    fields = {"cuda_targets": "List of CUDA targets to compile for."},
)

def cuda_library(name, **kwargs):
    """Macro wrapping a cc_library which can contain CUDA device code.

    Args:
      name: target name.
      **kwargs: forwarded to cc_library.
    """

    # Add targets to 'srcs' that fail to execute with a descriptive error
    # message if the current configuration doesn't support cuda_library targets.
    srcs = kwargs.pop("srcs", []) + if_local_cuda(
        select({
            "@rules_cuda//cuda:cuda_toolchain_detected": [],
            "//conditions:default": [
                "@rules_cuda//cuda:unsupported_cuda_toolchain_error",
            ],
        }),
        ["@rules_cuda//cuda:no_cuda_toolkit_error"],
    )

    deps = kwargs.pop("deps", []) + ["@rules_cuda//cuda:cuda_runtime"]

    features = kwargs.pop("features", []) + ["cuda", "-use_header_modules"]
    if kwargs.get("textual_hdrs", None):
        features += ["-layering_check", "-parse_headers"]

    native.cc_library(
        name = name,
        srcs = srcs,
        deps = deps,
        features = features,
        **kwargs
    )

def requires_cuda_enabled():
    """Returns constraint_setting that is not satisfied unless :is_cuda_enabled.

    Add to 'target_compatible_with' attribute to mark a target incompatible when
    @rules_cuda//cuda:enable_cuda is not set. Incompatible targets are excluded
    from bazel target wildcards and fail to build if requested explicitly."""
    return select({
        "@rules_cuda//cuda:is_cuda_enabled": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

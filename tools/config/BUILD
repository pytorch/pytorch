load("@bazel_skylib//lib:selects.bzl", "selects")

config_setting(
    name = "cuda",
    define_values = {
        "cuda": "true",
    },
)

# Even when building with --config=cuda, host targets should be built with cuda disabled
# as these targets will run on CI machines that have no GPUs.
selects.config_setting_group(
    name = "cuda_enabled_and_capable",
    match_all = [
        ":cuda",
    ],
)

# Configures the system to build with cuda using clang.
config_setting(
    name = "cuda_clang",
    define_values = {
        "cuda_clang": "true",
    },
)

# Indicates that cuda code should be compiled with nvcc
# Mostly exists to support _analysis_ of tensorflow; more work is needed to actually make this
# setting work.
config_setting(
    name = "cuda_nvcc",
    define_values = {
        "cuda_nvcc": "true",
    },
)

config_setting(
    name = "thread_sanitizer",
    define_values = {"thread_sanitizer": "1"},
    visibility = ["//visibility:public"],
)

# Indicates whether we should use codegen unboxing + static dispatch
# Currently only being supported on mobile build
config_setting(
    name = "lightweight_dispatch",
    define_values = {"lightweight_dispatch": "true"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "lightweight_dispatch_enabled",
    match_all = [
        ":lightweight_dispatch",
    ],
)
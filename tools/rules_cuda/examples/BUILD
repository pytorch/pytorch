load("//cuda:defs.bzl", "cuda_library", "requires_cuda_enabled")

cuda_library(
    name = "kernel",
    srcs = ["kernel.cu.cc"],
    hdrs = ["kernel.h"],
    target_compatible_with = requires_cuda_enabled(),
)

# Run with 'bazel run --cuda //examples:hello_cuda'
cc_binary(
    name = "hello_cuda",
    srcs = ["main.cc"],
    deps = [":kernel"],
)

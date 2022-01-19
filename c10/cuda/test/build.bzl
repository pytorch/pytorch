load("@rules_cuda//cuda:defs.bzl", "requires_cuda_enabled")

def define_targets(rules):
    rules.cc_test(
        name = "test",
        srcs = ["impl/CUDATest.cpp"],
        deps = [
            "@com_google_googletest//:gtest_main",
            "//c10/cuda",
        ],
        target_compatible_with = requires_cuda_enabled(),
    )

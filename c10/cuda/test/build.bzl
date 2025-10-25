def define_targets(rules, gtest_deps):
    rules.cc_test(
        name = "test",
        srcs = [
            "impl/CUDATest.cpp",
        ],
        target_compatible_with = rules.requires_cuda_enabled(),
        deps = [
            "//c10/cuda",
        ] + gtest_deps,
    )

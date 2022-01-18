def define_targets(rules):
    rules.cc_test(
        name = "test",
        srcs = ["impl/CUDATest.cpp"],
        deps = [
            "@com_google_googletest//:gtest_main",
            "//c10/cuda",
        ],
    )

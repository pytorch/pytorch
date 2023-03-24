def define_targets(rules):
    rules.cc_test(
        name = "try_ensure_test",
        size = "small",
        srcs = ["try_ensure_test.cpp"],
        deps = [
            "//c10/core:CPUAllocator",
            "//c10/core:base",
            "//c10/core/impl/cow:try_ensure",
            "@com_google_googletest//:gtest_main",
        ],
    )

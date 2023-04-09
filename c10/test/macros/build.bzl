def define_targets(rules):
    rules.cc_binary(
        name = "cmake_macros_tester",
        srcs = ["cmake_macros_tester.cpp"],
        testonly = True,
        deps = [
            "//c10",
            "@com_google_googletest//:gtest_main",
        ],
    )

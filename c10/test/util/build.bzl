def define_rules(rules):
    rules.cc_test(
        name = "C++17_test",
        srcs = ["C++17_test.cpp"],
        deps = [
            "@com_google_googletest//:gtest_main",
            "//c10/util:C++17",
        ],
    )

def define_targets(rules):
    rules.cc_test(
        name = "tests",
        size = "small",
        srcs = rules.glob([
            "util/*.cpp",
            "core/*.cpp",
            "core/impl/*.cpp",
        ]),
        copts = ["-Wno-deprecated-declarations"],
        deps = [
            ":Macros",
            ":complex_math_test_common",
            ":complex_test_common",
            "@com_google_googletest//:gtest_main",
            "//c10/core:base",
            "//c10/macros",
            "//c10/util:base",
            "//c10/util:typeid",
        ],
        visibility = ["//:__pkg__"],
    )

    rules.cc_library(
        name = "Macros",
        hdrs = ["util/Macros.h"],
        testonly = True,
        visibility = ["//:__subpackages__"],
    )

    rules.cc_library(
        name = "complex_math_test_common",
        hdrs = ["util/complex_math_test_common.h"],
        deps = [
            "@com_google_googletest//:gtest",
            "//c10/util:base",
        ],
        testonly = True,
    )

    rules.cc_library(
        name = "complex_test_common",
        hdrs = ["util/complex_test_common.h"],
        deps = [
            "@com_google_googletest//:gtest",
            "//c10/macros",
            "//c10/util:base",
        ],
        testonly = True,
    )

def define_targets(rules):
    rules.test_suite(
        name = "tests",
        tests = [
            ":core_tests",
            ":typeid_test",
            ":util_base_tests",
        ],
        visibility = ["//:__pkg__"],
    )

    rules.cc_test(
        name = "core_tests",
        size = "small",
        srcs = rules.glob([
            "core/*.cpp",
            "core/impl/*.cpp",
        ]),
        copts = ["-Wno-deprecated-declarations"],
        deps = [
            "@com_google_googletest//:gtest_main",
            "//c10/core:base",
            "//c10/util:base",
        ],
    )

    rules.cc_test(
        name = "typeid_test",
        size = "small",
        srcs = ["util/typeid_test.cpp"],
        copts = ["-Wno-deprecated-declarations"],
        deps = [
            "@com_google_googletest//:gtest_main",
            "//c10/util:typeid",
        ],
    )

    rules.cc_test(
        name = "util_base_tests",
        srcs = rules.glob(
            ["util/*.cpp"],
            exclude = [
                "util/ssize_test.cpp",
                "util/typeid_test.cpp",
            ],
        ),
        copts = ["-Wno-deprecated-declarations"],
        deps = [
            ":Macros",
            ":complex_math_test_common",
            ":complex_test_common",
            "@com_google_googletest//:gtest_main",
            "//c10/macros:macros",
            "//c10/util:base",
        ],
    )

    rules.cc_test(
        name = "util/ssize_test",
        srcs = ["util/ssize_test.cpp"],
        deps = [
            "@com_google_googletest//:gtest_main",
            "//c10/util:ssize",
        ],
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
            "//c10/macros:macros",
            "//c10/util:base",
        ],
        testonly = True,
    )

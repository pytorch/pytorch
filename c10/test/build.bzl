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
            "//c10/core:base",
            "//c10/util:base",
            "//c10/core:CPUAllocator",
            "@com_google_googletest//:gtest_main",
        ],
    )

    rules.cc_test(
        name = "typeid_test",
        size = "small",
        srcs = ["util/typeid_test.cpp"],
        copts = ["-Wno-deprecated-declarations"],
        deps = [
            "//c10/util:typeid",
            "@com_google_googletest//:gtest_main",
        ],
    )

    rules.cc_test(
        name = "util_base_tests",
        srcs = rules.glob(
            ["util/*.cpp"],
            exclude = [
                "util/bit_cast_test.cpp",
                "util/ssize_test.cpp",
                "util/typeid_test.cpp",
            ],
        ),
        copts = ["-Wno-deprecated-declarations", "-Wno-ctad-maybe-unsupported"],
        deps = [
            ":Macros",
            ":complex_math_test_common",
            ":complex_test_common",
            "//c10/macros",
            "//c10/util:base",
            "@com_google_googletest//:gtest_main",
        ],
    )

    rules.cc_test(
        name = "util/bit_cast_test",
        srcs = ["util/bit_cast_test.cpp"],
        deps = [
            "//c10/util:bit_cast",
            "@com_google_googletest//:gtest_main",
        ],
    )

    rules.cc_test(
        name = "util/nofatal_test",
        srcs = ["util/nofatal_test.cpp"],
        deps = [
            "//c10/util:base",
            "@com_google_googletest//:gtest_main",
        ],
    )

    rules.cc_test(
        name = "util/ssize_test",
        srcs = ["util/ssize_test.cpp"],
        deps = [
            "//c10/util:ssize",
            "@com_google_googletest//:gtest_main",
        ],
    )

    rules.cc_library(
        name = "Macros",
        testonly = True,
        hdrs = ["util/Macros.h"],
        visibility = ["//:__subpackages__"],
    )

    rules.cc_library(
        name = "complex_math_test_common",
        testonly = True,
        hdrs = ["util/complex_math_test_common.h"],
        deps = [
            "//c10/util:base",
            "@com_google_googletest//:gtest",
        ],
    )

    rules.cc_library(
        name = "complex_test_common",
        testonly = True,
        hdrs = ["util/complex_test_common.h"],
        deps = [
            "//c10/macros",
            "//c10/util:base",
            "@com_google_googletest//:gtest",
        ],
    )

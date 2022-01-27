def define_targets(rules):
    rules.cc_test(
        name = "tests",
        size = "small",
        srcs = [
            "core/CompileTimeFunctionPointer_test.cpp",
            "core/DeviceGuard_test.cpp",
            "core/Device_test.cpp",
            "core/DispatchKeySet_test.cpp",
            "core/StreamGuard_test.cpp",
            "core/impl/InlineDeviceGuard_test.cpp",
            "core/impl/InlineStreamGuard_test.cpp",
            "core/impl/SizesAndStrides_test.cpp",
            "util/Array_test.cpp",
            "util/Bitset_test.cpp",
            "util/C++17_test.cpp",
            "util/ConstexprCrc_test.cpp",
            "util/Half_test.cpp",
            "util/LeftRight_test.cpp",
            "util/Metaprogramming_test.cpp",
            "util/SmallVectorTest.cpp",
            "util/ThreadLocal_test.cpp",
            "util/TypeIndex_test.cpp",
            "util/TypeList_test.cpp",
            "util/TypeTraits_test.cpp",
            "util/accumulate_test.cpp",
            "util/bfloat16_test.cpp",
            "util/complex_math_test.cpp",
            "util/complex_test.cpp",
            "util/either_test.cpp",
            "util/exception_test.cpp",
            "util/flags_test.cpp",
            "util/intrusive_ptr_test.cpp",
            "util/irange_test.cpp",
            "util/logging_test.cpp",
            "util/optional_test.cpp",
            "util/ordered_preserving_dict_test.cpp",
            "util/registry_test.cpp",
            "util/string_view_test.cpp",
            "util/tempfile_test.cpp",
            "util/typeid_test.cpp",
        ],
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

def define_targets(rules):
    rules.cc_library(
        name = "conjugate_fallback",
        visibility = ["//:__pkg__"],
        srcs = ["ConjugateFallback.cpp"],
        alwayslink = True,
        deps = [
            ":math_bits_fallthrough_lists",
            ":unary_involution_fallback",
        ],
    )

    rules.cc_library(
        name = "math_bits_fallthrough_lists",
        visibility = ["//:__pkg__"],
        hdrs = ["MathBitFallThroughLists.h"],
    )

    rules.cc_library(
        name = "negate_fallback",
        visibility = ["//:__pkg__"],
        srcs = ["NegateFallback.cpp"],
        alwayslink = True,
        deps = [
            ":math_bits_fallthrough_lists",
            ":unary_involution_fallback",
        ],
    )

    rules.cc_library(
        name = "transform_fallback",
        hdrs = ["TransformFallback.h"],
        srcs = ["TransformFallback.cpp"],
        deps = [
            ":math_bits_fallthrough_lists",
            "//:ATen-core",
            "//:ATen-native",
            "//:torch_library_headers",
            "//c10/core:base",
        ],
    )

    rules.cc_library(
        name = "unary_involution_fallback",
        hdrs = ["UnaryInvolutionFallback.h"],
        srcs = ["UnaryInvolutionFallback.cpp"],
        deps = [
            ":transform_fallback",
            "//:ATen-core",
            "//c10/core:base",
        ],
    )

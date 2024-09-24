def define_targets(rules):
    rules.cc_library(
        name = "TypeCast",
        srcs = ["TypeCast.cpp"],
        hdrs = ["TypeCast.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":base",
            "//c10/core:ScalarType",
            "//c10/macros",
        ],
    )

    rules.cc_library(
        name = "base",
        srcs = rules.glob(
            ["*.cpp"],
            exclude = [
                "TypeCast.cpp",
                "typeid.cpp",
            ],
        ),
        hdrs = rules.glob(
            ["*.h"],
            exclude = [
                "TypeCast.h",
                "typeid.h",
            ],
        ),
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":bit_cast",
            "//c10/macros",
            "@fmt",
        ] + rules.select({
            "//c10:using_gflags": ["@com_github_gflags_gflags//:gflags"],
            "//conditions:default": [],
        }) + rules.select({
            "//c10:using_glog": ["@com_github_glog//:glog"],
            "//conditions:default": [],
        }),
        linkopts = rules.select({
            "@bazel_tools//src/conditions:windows": [],
            "//conditions:default": ["-ldl"],
        }),
        # This library uses flags and registration. Do not let the
        # linker remove them.
        alwayslink = True,
    )

    rules.cc_library(
        name = "bit_cast",
        hdrs = ["bit_cast.h"],
        visibility = ["//:__subpackages__"],
    )

    rules.cc_library(
        name = "ssize",
        hdrs = ["ssize.h"],
        linkstatic = True,
        visibility = ["//:__subpackages__"],
        deps = [":base"],
    )

    rules.cc_library(
        name = "typeid",
        srcs = ["typeid.cpp"],
        hdrs = ["typeid.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":base",
            "//c10/core:ScalarType",
            "//c10/macros",
        ],
    )

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(
            ["*.h"],
            exclude = [
                "bit_cast.h",
                "ssize.h",
            ],
        ),
        visibility = [
            "//:__pkg__",
            "//c10:__pkg__",
        ],
    )

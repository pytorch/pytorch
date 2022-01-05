def define_targets(rules):
    rules.cc_library(
        name = "TypeCast",
        srcs = ["TypeCast.cpp"],
        hdrs = ["TypeCast.h"],
        visibility = ["//visibility:public"],
        deps = [
            ":base",
            "//c10/core:ScalarType",
            "//c10/macros:macros",
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
        visibility = ["//visibility:public"],
        deps = [
            "@fmt",
            "//c10/macros:macros",
        ] + rules.select({
            "//c10:using_gflags": ["@com_github_gflags_gflags//:gflags"],
            "//conditions:default": [],
        }) + rules.select({
            "//c10:using_glog": ["@com_github_glog//:glog"],
            "//conditions:default": [],
        }),
    )

    rules.cc_library(
        name = "typeid",
        srcs = ["typeid.cpp"],
        hdrs = ["typeid.h"],
        visibility = ["//visibility:public"],
        deps = [
            ":base",
            "//c10/core:ScalarType",
            "//c10/macros:macros",
        ],
    )

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(
            ["*.h"],
            exclude = [
            ],
        ),
        visibility = ["//c10:__pkg__"],
    )

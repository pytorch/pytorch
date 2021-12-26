def define_targets(rules):
    rules.cc_library(
        name = "base",
        srcs = glob(
            ["*.cpp"],
            exclude = [
                "TypeCast.cpp",
                "typeid.cpp",
            ],
        ),
        hdrs = glob(
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

    rules.exports_files(
        glob(["*.h"]) + # for //c10:headers
        [
            "TypeCast.cpp",
            "typeid.cpp",
        ],
        visibility = ["//c10:__pkg__"],
    )

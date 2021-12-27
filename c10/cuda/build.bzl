def define_targets(rules):
    rules.cc_library(
        name = "cuda",
        srcs = glob(
            [
                "*.cpp",
                "impl/*.cpp",
            ],
            exclude = [
            ],
        ),
        hdrs = glob(
            [
                "*.h",
                "impl/*.h",
            ],
            exclude = [
            ],
        ),
        visibility = ["//visibility:public"],
        deps = [
            "@cuda",
            "//c10/core:base",
            "//c10/macros:macros",
            "//c10/util:base",
        ],
    )

    rules.filegroup(
        name = "headers",
        srcs = glob(
            [
                "*.h",
                "impl/*.h",
            ],
            exclude = [
            ],
        ),
        visibility = ["//c10:__pkg__"],
    )

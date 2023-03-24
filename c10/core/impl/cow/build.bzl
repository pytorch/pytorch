def define_targets(rules):
    rules.cc_library(
        name = "context",
        srcs = ["context.cpp"],
        hdrs = ["context.h"],
        visibility = ["//c10/test/core/impl/cow:__pkg__"],
        deps = [
            "//c10/core:base",
            "//c10/util:base",
        ],
    )

    rules.cc_library(
        name = "materialize",
        srcs = ["materialize.cpp"],
        hdrs = ["materialize.h"],
        deps = [
            ":context",
            "//c10/core:base",
            "//c10/macros",
        ],
        visibility = [
            "//:__pkg__",
        ],
    )

    rules.cc_library(
        name = "try_ensure",
        srcs = ["try_ensure.cpp"],
        hdrs = ["try_ensure.h"],
        deps = [
            ":context",
            "//c10/core:base",
            "//c10/macros",
            "//c10/util:base",
        ],
        visibility = [
            "//:__pkg__",
            "//c10/test/core/impl/cow:__pkg__",
        ],
    )

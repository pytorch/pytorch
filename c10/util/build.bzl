def define_rules(rules):
    rules.package(default_visibility = ["//:__subpackages__"])

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(
            ["*.h"],
            exclude=[
                "Array.h",
                "C++17.h",
            ]),
        visibility = ["//:__pkg__"],
    )

    rules.filegroup(
        name = "sources",
        srcs = rules.glob(
            ["*.cpp"],
            exclude=[
                "Array.cpp",
                "C++17.cpp",
            ],
        ),
        visibility = ["//:__pkg__"],
    )

    rules.cc_library(
        name = "Array",
        hdrs = ["Array.h"],
        srcs = ["Array.cpp"],
        deps = [":C++17"],
    )

    rules.cc_library(
        name = "C++17",
        hdrs = ["C++17.h"],
        srcs = ["C++17.cpp"],
        deps = ["//c10/macros:Macros"],
    )

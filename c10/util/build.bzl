def define_rules(rules):
    rules.package(default_visibility = ["//:__subpackages__"])

    rules.cc_library(
        name = "headers",
        hdrs = rules.glob(["*.h"],
                     exclude=[
                         "C++17.h",
                     ]),
        deps = ["//c10/macros:Macros"],
        visibility = ["//:__pkg__"],
    )

    rules.cc_library(
        name = "util",
        hdrs = rules.glob(["*.h"],
                     exclude=[
                         "C++17.h",
                     ]),
        srcs = rules.glob(["*.cpp"],
                     exclude=[
                         "C++17.cpp",
                     ]),
        deps = ["//c10/macros:Macros"],
        visibility = ["//:__pkg__"],
    )

    rules.cc_library(
        name = "C++17",
        hdrs = ["C++17.h"],
        srcs = ["C++17.cpp"],
        deps = ["//c10/macros:Macros"],
    )

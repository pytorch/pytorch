def define_rules(rules):
    rules.package(default_visibility = ["//:__subpackages__"])

    rules.cc_library(
        name = "C++17",
        hdrs = ["C++17.h"],
        srcs = ["C++17.cpp"],
        deps = ["//c10/macros:Macros"],
    )

    # Temporary targets to export the headers and sources that are not
    # in libraries but are still needed for the //:c10 target that we
    # are slowly replacing.

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(["*.h"]),
        visibility = ["//:__pkg__"],
    )

    rules.filegroup(
        name = "sources",
        srcs = rules.glob(
            ["*.cpp"],
            exclude=[
                "C++17.cpp",
            ],
        ),
        visibility = ["//:__pkg__"],
    )

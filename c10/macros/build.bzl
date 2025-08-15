def define_targets(rules):
    rules.cc_library(
        name = "macros",
        hdrs = [
            "Macros.h",
            # Despite the documentation in Macros.h, Export.h is included
            # directly by many downstream files. Thus, we declare it as a
            # public header in this file.
            "Export.h",
            "cmake_macros.h",
        ],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            "//torch/headeronly:torch_headeronly",
        ],
    )

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(
            ["*.h"],
            exclude = [
            ],
        ),
        visibility = ["//:__pkg__"],
    )

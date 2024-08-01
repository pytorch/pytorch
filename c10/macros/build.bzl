def define_targets(rules):
    rules.cc_library(
        name = "macros",
        srcs = [":cmake_macros_h"],
        hdrs = [
            "Macros.h",
            # Despite the documentation in Macros.h, Export.h is included
            # directly by many downstream files. Thus, we declare it as a
            # public header in this file.
            "Export.h",
        ],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
    )

    rules.cmake_configure_file(
        name = "cmake_macros_h",
        src = "cmake_macros.h.in",
        out = "cmake_macros.h",
        definitions = [
            "C10_BUILD_SHARED_LIBS",
            "C10_USE_MSVC_STATIC_RUNTIME",
        ] + rules.select({
            "//c10:using_gflags": ["C10_USE_GFLAGS"],
            "//conditions:default": [],
        }) + rules.select({
            "//c10:using_glog": ["C10_USE_GLOG"],
            "//conditions:default": [],
        }),
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

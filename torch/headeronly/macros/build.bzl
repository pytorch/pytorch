def define_targets(rules):
    rules.cc_library(
        name = "macros",
        srcs = [":cmake_macros_h"],
        hdrs = [
            # Following the example from c10
            "Export.h",
            "Macros.h",
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

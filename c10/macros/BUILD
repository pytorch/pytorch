load("@rules//:cmake_configure_file.bzl", "cmake_configure_file")
load("@rules//:cxx_library.bzl", "cxx_library")

cxx_library(
    name = "macros",
    exported_headers = [
        "Macros.h",
        # Despite the documentation in Macros.h, Export.h is included
        # directly by many downstream files. Thus, we declare it as a
        # public header in this file.
        "Export.h",
    ],
    headers = {
        "cmake_macros.h": ":cmake_macros_h",
    },
    visibility = ["PUBLIC"],
)

cmake_configure_file(
    name = "cmake_macros_h",
    src = "cmake_macros.h.in",
    out = "cmake_macros.h",
    definitions = [
        "C10_BUILD_SHARED_LIBS",
        "C10_USE_MSVC_STATIC_RUNTIME",
    ] + select({
        "//c10:using_gflags": ["C10_USE_GFLAGS"],
        "//conditions:default": [],
    }) + select({
        "//c10:using_glog": ["C10_USE_GLOG"],
        "//conditions:default": [],
    }),
)

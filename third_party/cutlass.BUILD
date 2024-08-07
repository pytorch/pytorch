# Description:
#   CUDA Templates for Linear Algebra Subroutines

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "cutlass",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
        "tools/util/include/**/*.h",
        "tools/util/include/**/*.hpp",
        "tools/util/include/**/*.inl",
    ]),
    includes = [
        "include/",
        "tools/util/include/",
    ],
    visibility = ["//visibility:public"],
)

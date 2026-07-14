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
    defines = [
        "CUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES=1",
        "CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    ],
    includes = [
        "include/",
        "tools/util/include/",
    ],
    visibility = ["//visibility:public"],
)

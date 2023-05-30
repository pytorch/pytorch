# Description:
#   CUDA Templates for Linear Algebra Subroutines

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "cutlass",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"]),
    includes = ["include/"],
    visibility = ["//visibility:public"],
)

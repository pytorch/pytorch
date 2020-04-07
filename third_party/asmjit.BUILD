load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "asmjit",
    srcs = glob([
        "src/asmjit/core/*.cpp",
        "src/asmjit/x86/*.cpp",
    ]),
    hdrs = glob([
        "src/asmjit/x86/*.h",
        "src/asmjit/core/*.h",
        "src/asmjit/*.h",
    ]),
    copts = [
        "-DASMJIT_STATIC",
        "-fno-tree-vectorize",
        "-std=c++17",
        "-fmerge-all-constants",
        "-std=gnu++11",
        "-DTH_BLAS_MKL",
    ],
    includes = [
        "asmjit/",
        "src/",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
)

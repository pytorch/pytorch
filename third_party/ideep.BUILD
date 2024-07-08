load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "ideep",
    hdrs = glob([
        "include/**/*.hpp",
        "include/**/*.h",
    ]),
    defines = [
        "IDEEP_USE_MKL",
    ],
    includes = [
        "include/",
    ],
    visibility = ["//visibility:public"],
    deps = ["@mkl_dnn//:mkl-dnn"],
)

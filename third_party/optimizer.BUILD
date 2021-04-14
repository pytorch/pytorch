load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "optimizer",
    srcs = [
        "optimizer/onnxoptimizer/*.cc",
    ],
    hdrs = glob([
        "optimizer/onnxoptimizer/*.h",
    ]),
    includes = [
        "optimizer/onnxoptimizer/",
        "optimizer/onnxoptimizer/passes/",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

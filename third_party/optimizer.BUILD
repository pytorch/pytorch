load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "optimizer",
    srcs = glob(
        [
            "onnxoptimizer/*.cc",
        ],
        exclude = [
            "onnxoptimizer/cpp2py_export.cc",
        ],
    ),
    hdrs = glob([
        "onnxoptimizer/*.h",
        "onnxoptimizer/passes/*.h",
    ]),
    includes = [
        "onnxoptimizer/",
        "onnxoptimizer/passes/",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@onnx",
    ]
)

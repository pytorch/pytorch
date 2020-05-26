load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "foxi",
    srcs = [
        "foxi/onnxifi_loader.c",
    ],
    hdrs = glob([
        "foxi/*.h",
    ]),
    includes = [
        ".",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

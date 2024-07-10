load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "nlohmann",
    hdrs = glob(["include/**/*.hpp"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nlohmann_single_include",
    hdrs = glob(["single_include/nlohmann/*.hpp"]),
    visibility = ["//visibility:public"],
)


load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(name = "nlohmann",
    includes = ["include"],
    deps = ["nlohmann-internal"],
    visibility = ["//visibility:public"],
)

cc_import(name = "nlohmann-internal",
     hdrs = glob(["include/**/*.hpp"]),
     visibility = ["//visibility:private"],
)

cc_library(
    name = "nlohmann_single_include",
    hdrs = glob(["single_include/nlohmann/*.hpp"]),
    visibility = ["//visibility:public"],
)

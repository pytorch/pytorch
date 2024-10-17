load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "mkl_headers",
    hdrs = glob(["include/*.h"]),
    includes = ["include/"],
    visibility = ["//visibility:public"],
)

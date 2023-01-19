load("@rules_cc//cc:defs.bzl", "cc_library")
load("//examples:experimental_cc_shared_library.bzl", "cc_shared_library_permissions")

cc_library(
    name = "bar",
    srcs = ["bar.cc"],
    hdrs = ["bar.h"],
    visibility = ["//visibility:public"],
)

cc_shared_library_permissions(
    name = "permissions",
    targets = [
        "//examples/test_cc_shared_library3:bar",
    ],
    visibility = ["//examples/test_cc_shared_library:__pkg__"],
)

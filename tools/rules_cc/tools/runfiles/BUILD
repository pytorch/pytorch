load("//cc:defs.bzl", "cc_library", "cc_test")

licenses(["notice"])

cc_library(
    name = "runfiles",
    testonly = 1,
    srcs = ["runfiles_src.cc"],
    hdrs = ["runfiles_src.h"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "runfiles_test",
    srcs = ["runfiles_test.cc"],
    deps = [
        ":runfiles",
        "@com_google_googletest//:gtest_main",
    ],
)

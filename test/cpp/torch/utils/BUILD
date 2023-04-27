load("@rules_cc//cc:defs.bzl", "cc_test")

cc_test(
    name = "reprfunc_test",
    srcs = ["reprfunc_test.cpp"],
    deps = [
        "//torch/csrc/utils:reprfunc",
        "@com_google_googletest//:gtest_main",
        "@local_config_python//:python_embed",
    ],
)

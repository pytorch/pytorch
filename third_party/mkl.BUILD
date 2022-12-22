load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "mkl",
    srcs = [
        "libmkl_avx2.so",
        "libmkl_core.so",
        "libmkl_def.so",
        "libmkl_intel_lp64.so",
        "libmkl_rt.so",
        "libmkl_sequential.so",
        "libmkl_vml_avx2.so",
        "libmkl_vml_avx512.so",
        "libmkl_vml_def.so",
    ] + select({
        "@pytorch//tools/config:thread_sanitizer": [],
        "//conditions:default": ["libmkl_tbb_thread.so"],
    }),
    visibility = ["//visibility:public"],
    deps = ["@mkl_headers"],
)

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "mkl",
    srcs = [
        "libmkl_avx2.dylib",
        "libmkl_core.dylib",
        # "libmkl_def.dylib",
        "libmkl_intel_lp64.dylib",
        "libmkl_rt.dylib",
        "libmkl_sequential.dylib",
        "libmkl_vml_avx2.dylib",
        "libmkl_vml_avx512.dylib",
        # "libmkl_vml_def.dylib",
    ] + select({
        "@//tools/config:thread_sanitizer": [],
        "//conditions:default": [], # ["libmkl_tbb_thread.dylib"],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "@platforms//os:linux": ["@mkl_headers_linux//:mkl_headers"],
        "@platforms//os:macos": ["@mkl_headers_macos//:mkl_headers"],
    }),
)

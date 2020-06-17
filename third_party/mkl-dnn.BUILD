load("@rules_cc//cc:defs.bzl", "cc_library")
load("@//third_party:substitution.bzl", "template_rule")

template_rule(
    name = "include_dnnl_version",
    src = "include/dnnl_version.h.in",
    out = "include/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "1",
        "@DNNL_VERSION_MINOR@": "5",
        "@DNNL_VERSION_PATCH@": "0",
        "@DNNL_VERSION_HASH@": "e2ac1fac44c5078ca927cb9b90e1b3066a0b2ed0",
    },
)

template_rule(
    name = "include_dnnl_config",
    src = "include/dnnl_config.h.in",
    out = "include/dnnl_config.h",
    substitutions = {
        "cmakedefine": "define",
        "${DNNL_CPU_THREADING_RUNTIME}": "OMP",
        "${DNNL_CPU_RUNTIME}": "OMP",
        "${DNNL_GPU_RUNTIME}": "NONE",
    },
)

cc_library(
    name = "mkl-dnn",
    srcs = glob([
        "src/common/*.cpp",
        "src/cpu/**/*.cpp",
    ]),
    hdrs = glob([
        "include/*.h",
        "include/*.hpp",
        "src/*.hpp",
        "src/cpu/**/*.hpp",
        "src/cpu/**/*.h",
        "src/common/*.hpp",
        "src/cpu/rnn/*.hpp",
    ]) + [
        "include/dnnl_version.h",
        "include/dnnl_config.h",
    ],
    copts = [
        "-DUSE_AVX",
        "-DUSE_AVX2",
        "-DDNNL_DLL",
        "-DDNNL_DLL_EXPORTS",
        "-DDNNL_ENABLE_CONCURRENT_EXEC",
        "-DTH_BLAS_MKL",
        "-D__STDC_CONSTANT_MACROS",
        "-D__STDC_LIMIT_MACROS",
        "-fno-strict-overflow",
        "-fopenmp",
    ] + select({
        "@//tools/config:thread_sanitizer": ["-DMKLDNN_THR=0"],
        "//conditions:default": ["-DMKLDNN_THR=2"],
    }),
    includes = [
        "include/",
        "src/",
        "src/common/",
        "src/cpu/",
        "src/cpu/x64/xbyak/",
    ],
    visibility = ["//visibility:public"],
    linkopts = [
        "-lgomp",
    ],
    deps = [
        "@mkl",
    ] + select({
        "@//tools/config:thread_sanitizer": [],
        "//conditions:default": ["@tbb"],
    }),
)

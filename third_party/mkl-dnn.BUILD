load("@rules_cc//cc:defs.bzl", "cc_library")
load("@//third_party:substitution.bzl", "template_rule")

template_rule(
    name = "include_dnnl_version",
    src = "include/dnnl_version.h.in",
    out = "include/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "1",
        "@DNNL_VERSION_MINOR@": "2",
        "@DNNL_VERSION_PATCH@": "0",
        "@DNNL_VERSION_HASH@": "70f8b879ea7a0c38caedb3320b7c85e8497ff50d",
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
        "src/cpu/*.cpp",
        "src/cpu/binary/*.cpp",
        "src/cpu/gemm/*.cpp",
        "src/cpu/gemm/bf16/*.cpp",
        "src/cpu/gemm/f32/*.cpp",
        "src/cpu/gemm/s8x8s32/*.cpp",
        "src/cpu/jit_utils/*.cpp",
        "src/cpu/jit_utils/jitprofiling/*.c",
        "src/cpu/jit_utils/linux_perf/*.cpp",
        "src/cpu/matmul/*.cpp",
        "src/cpu/resampling/*.cpp",
        "src/cpu/rnn/*.cpp",
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
        "src/cpu/xbyak/",
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

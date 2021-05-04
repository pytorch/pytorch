load("@rules_cc//cc:defs.bzl", "cc_library")
load("@//third_party:substitution.bzl", "template_rule")

_DNNL_RUNTIME_OMP = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
    "#cmakedefine DNNL_WITH_SYCL": "/* #undef DNNL_WITH_SYCL */",
    "#cmakedefine DNNL_WITH_LEVEL_ZERO": "/* #undef DNNL_WITH_LEVEL_ZERO */",
    "#cmakedefine DNNL_SYCL_CUDA": "/* #undef DNNL_SYCL_CUDA */",
}

template_rule(
    name = "include_dnnl_version",
    src = "include/oneapi/dnnl/dnnl_version.h.in",
    out = "include/oneapi/dnnl/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "2",
        "@DNNL_VERSION_MINOR@": "1",
        "@DNNL_VERSION_PATCH@": "2",
        "@DNNL_VERSION_HASH@": "98be7e8afa711dc9b66c8ff3504129cb82013cdb",
    },
)

template_rule(
    name = "include_dnnl_config",
    src = "include/oneapi/dnnl/dnnl_config.h.in",
    out = "include/oneapi/dnnl/dnnl_config.h",
    substitutions = _DNNL_RUNTIME_OMP,
)

cc_library(
    name = "mkl-dnn",
    srcs = glob([
        "src/common/*.cpp",
        "src/cpu/**/*.cpp",
    ], exclude=[
        "src/cpu/aarch64/**/*.cpp",
    ]),
    hdrs = glob([
        "include/oneapi/dnnl/*.h",
        "include/oneapi/dnnl/*.hpp",
        "include/*.h",
        "include/*.hpp",
        "src/cpu/**/*.hpp",
        "src/cpu/**/*.h",
        "src/common/*.hpp",
    ], exclude=[
        "src/cpu/aarch64/**/*.hpp",
        "src/cpu/aarch64/**/*.h",
    ]) + [
        "include/oneapi/dnnl/dnnl_config.h",
        "include/oneapi/dnnl/dnnl_version.h",
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
        "@//tools/config:thread_sanitizer": ["-DDNNL_CPU_RUNTIME=0"],
        "//conditions:default": ["-DDNNL_CPU_RUNTIME=2"],
    }),
    includes = [
        "include/",
        "include/oneapi/",
        "include/oneapi/dnnl/",
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

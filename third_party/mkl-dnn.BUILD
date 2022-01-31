load("@rules_cc//cc:defs.bzl", "cc_library")
load("@//third_party:substitution.bzl", "template_rule")

_DNNL_RUNTIME_OMP = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
    "#cmakedefine DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE": "/* undef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE */",
    "#cmakedefine DNNL_WITH_SYCL": "/* #undef DNNL_WITH_SYCL */",
    "#cmakedefine DNNL_WITH_LEVEL_ZERO": "/* #undef DNNL_WITH_LEVEL_ZERO */",
    "#cmakedefine DNNL_SYCL_CUDA": "/* #undef DNNL_SYCL_CUDA */",
}

template_rule(
    name = "third_party/oneDNN/include_dnnl_version",
    src = "third_party/oneDNN/include/oneapi/dnnl/dnnl_version.h.in",
    out = "third_party/oneDNN/include/oneapi/dnnl/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "2",
        "@DNNL_VERSION_MINOR@": "3",
        "@DNNL_VERSION_PATCH@": "3",
        "@DNNL_VERSION_HASH@": "f40443c413429c29570acd6cf5e3d1343cf647b4",
    },
)

template_rule(
    name = "third_party/oneDNN/include_dnnl_config",
    src = "third_party/oneDNN/include/oneapi/dnnl/dnnl_config.h.in",
    out = "third_party/oneDNN/include/oneapi/dnnl/dnnl_config.h",
    substitutions = _DNNL_RUNTIME_OMP,
)

cc_library(
    name = "mkl-dnn",
    srcs = glob([
        "third_party/oneDNN/src/common/*.cpp",
        "third_party/oneDNN/src/cpu/**/*.cpp",
    ], exclude=[
        "third_party/oneDNN/src/cpu/aarch64/**/*.cpp",
    ]),
    hdrs = glob([
        "third_party/oneDNN/include/oneapi/dnnl/*.h",
        "third_party/oneDNN/include/oneapi/dnnl/*.hpp",
        "third_party/oneDNN/include/*.h",
        "third_party/oneDNN/include/*.hpp",
        "third_party/oneDNN/src/cpu/**/*.hpp",
        "third_party/oneDNN/src/cpu/**/*.h",
        "third_party/oneDNN/src/common/*.hpp",
        "third_party/oneDNN/src/common/ittnotify/jitprofiling.h",
    ], exclude=[
        "third_party/oneDNN/src/cpu/aarch64/**/*.hpp",
        "third_party/oneDNN/src/cpu/aarch64/**/*.h",
    ]) + [
        "third_party/oneDNN/include/oneapi/dnnl/dnnl_config.h",
        "third_party/oneDNN/include/oneapi/dnnl/dnnl_version.h",
    ],
    copts = [
        "-DUSE_AVX",
        "-DUSE_AVX2",
        "-DDNNL_DLL",
        "-DDNNL_DLL_EXPORTS",
        "-DDNNL_ENABLE_CONCURRENT_EXEC",
        "-D__STDC_CONSTANT_MACROS",
        "-D__STDC_LIMIT_MACROS",
        "-fno-strict-overflow",
        "-fopenmp",
    ] + select({
        "@//tools/config:thread_sanitizer": ["-DDNNL_CPU_RUNTIME=0"],
        "//conditions:default": ["-DDNNL_CPU_RUNTIME=2"],
    }),
    includes = [
        "third_party/oneDNN/include/",
        "third_party/oneDNN/include/oneapi/",
        "third_party/oneDNN/include/oneapi/dnnl/",
        "third_party/oneDNN/src/",
        "third_party/oneDNN/src/common/",
        "third_party/oneDNN/src/cpu/",
        "third_party/oneDNN/src/cpu/x64/xbyak/",
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

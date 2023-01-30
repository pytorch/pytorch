load("@rules_cc//cc:defs.bzl", "cc_library")
load("@pytorch//third_party:substitution.bzl", "template_rule")

_DNNL_RUNTIME_OMP = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
    "#cmakedefine DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE": "/* undef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE */",
    "#cmakedefine DNNL_WITH_SYCL": "/* #undef DNNL_WITH_SYCL */",
    "#cmakedefine DNNL_WITH_LEVEL_ZERO": "/* #undef DNNL_WITH_LEVEL_ZERO */",
    "#cmakedefine DNNL_SYCL_CUDA": "/* #undef DNNL_SYCL_CUDA */",
    "#cmakedefine DNNL_SYCL_HIP": "/* #undef DNNL_SYCL_HIP */",
    "#cmakedefine DNNL_ENABLE_STACK_CHECKER": "#undef DNNL_ENABLE_STACK_CHECKER",
    "#cmakedefine DNNL_EXPERIMENTAL": "#undef DNNL_EXPERIMENTAL",
    "#cmakedefine01 BUILD_TRAINING": "#define BUILD_TRAINING 1",
    "#cmakedefine01 BUILD_INFERENCE": "#define BUILD_INFERENCE 0",
    "#cmakedefine01 BUILD_PRIMITIVE_ALL": "#define BUILD_PRIMITIVE_ALL 1",
    "#cmakedefine01 BUILD_BATCH_NORMALIZATION": "#define BUILD_BATCH_NORMALIZATION 0",
    "#cmakedefine01 BUILD_BINARY": "#define BUILD_BINARY 0",
    "#cmakedefine01 BUILD_CONCAT": "#define BUILD_CONCAT 0",
    "#cmakedefine01 BUILD_CONVOLUTION": "#define BUILD_CONVOLUTION 0",
    "#cmakedefine01 BUILD_DECONVOLUTION": "#define BUILD_DECONVOLUTION 0",
    "#cmakedefine01 BUILD_ELTWISE": "#define BUILD_ELTWISE 0",
    "#cmakedefine01 BUILD_INNER_PRODUCT": "#define BUILD_INNER_PRODUCT 0",
    "#cmakedefine01 BUILD_LAYER_NORMALIZATION": "#define BUILD_LAYER_NORMALIZATION 0",
    "#cmakedefine01 BUILD_LRN": "#define BUILD_LRN 0",
    "#cmakedefine01 BUILD_MATMUL": "#define BUILD_MATMUL 0",
    "#cmakedefine01 BUILD_POOLING": "#define BUILD_POOLING 0",
    "#cmakedefine01 BUILD_PRELU": "#define BUILD_PRELU 0",
    "#cmakedefine01 BUILD_REDUCTION": "#define BUILD_REDUCTION 0",
    "#cmakedefine01 BUILD_REORDER": "#define BUILD_REORDER 0",
    "#cmakedefine01 BUILD_RESAMPLING": "#define BUILD_RESAMPLING 0",
    "#cmakedefine01 BUILD_RNN": "#define BUILD_RNN 0",
    "#cmakedefine01 BUILD_SHUFFLE": "#define BUILD_SHUFFLE 0",
    "#cmakedefine01 BUILD_SOFTMAX": "#define BUILD_SOFTMAX 0",
    "#cmakedefine01 BUILD_SUM": "#define BUILD_SUM 0",
    "#cmakedefine01 BUILD_PRIMITIVE_CPU_ISA_ALL": "#define BUILD_PRIMITIVE_CPU_ISA_ALL 1",
    "#cmakedefine01 BUILD_SSE41": "#define BUILD_SSE41 0",
    "#cmakedefine01 BUILD_AVX2": "#define BUILD_AVX2 0",
    "#cmakedefine01 BUILD_AVX512": "#define BUILD_AVX512 0",
    "#cmakedefine01 BUILD_AMX": "#define BUILD_AMX 0",
    "#cmakedefine01 BUILD_PRIMITIVE_GPU_ISA_ALL": "#define BUILD_PRIMITIVE_GPU_ISA_ALL 1",
    "#cmakedefine01 BUILD_GEN9": "#define BUILD_GEN9 0",
    "#cmakedefine01 BUILD_GEN11": "#define BUILD_GEN11 0",
    "#cmakedefine01 BUILD_XELP": "#define BUILD_XELP 0",
    "#cmakedefine01 BUILD_XEHPG": "#define BUILD_XEHPG 0",
    "#cmakedefine01 BUILD_XEHPC": "#define BUILD_XEHPC 0",
    "#cmakedefine01 BUILD_XEHP": "#define BUILD_XEHP 0",
}

template_rule(
    name = "third_party/oneDNN/include_dnnl_version",
    src = "third_party/oneDNN/include/oneapi/dnnl/dnnl_version.h.in",
    out = "third_party/oneDNN/include/oneapi/dnnl/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "2",
        "@DNNL_VERSION_MINOR@": "7",
        "@DNNL_VERSION_PATCH@": "3",
        "@DNNL_VERSION_HASH@": "7710bdc92064a08b985c5cbdb09de773b19cba1f",
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
        "-DDNNL_DLL",
        "-DDNNL_DLL_EXPORTS",
        "-DDNNL_ENABLE_CONCURRENT_EXEC",
        "-D__STDC_CONSTANT_MACROS",
        "-D__STDC_LIMIT_MACROS",
        "-fno-strict-overflow",
        "-fopenmp",
    ] + select({
        "@pytorch//tools/config:thread_sanitizer": ["-DDNNL_CPU_RUNTIME=0"],
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
        "@pytorch//tools/config:thread_sanitizer": [],
        "//conditions:default": ["@tbb"],
    }),
)

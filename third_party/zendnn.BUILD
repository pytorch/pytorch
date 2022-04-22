load("@rules_cc//cc:defs.bzl", "cc_library")
load("@//third_party:substitution.bzl", "template_rule")

template_rule(
    name = "include_zendnn_version",
    src = "include/zendnn_version.h.in",
    out = "include/zendnn_version.h",
    substitutions = {
        "@ZENDNN_VERSION_MAJOR@": "3",
        "@ZENDNN_VERSION_MINOR@": "2",
        "@ZENDNN_VERSION_PATCH@": "0",
    },
)

template_rule(
    name = "include_zendnn_config",
    src = "include/zendnn_config.h.in",
    out = "include/zendnn_config.h",
    substitutions = {
        "cmakedefine": "define",
        "${ZENDNN_CPU_THREADING_RUNTIME}": "OMP",
        "${ZENDNN_GPU_RUNTIME}": "NONE",
    },
)

cc_library(
    name = "zendnn",
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
        "include/zendnn_version.h",
        "include/zendnn_config.h",
    ],
    copts = [
        "-DUSE_AVX",
        "-DUSE_AVX2",
        "-DZENDNN_DLL",
        "-DZENDNN_DLL_EXPORTS",
        "-DZENDNN_ENABLE_CONCURRENT_EXEC",
        "-D__STDC_CONSTANT_MACROS",
        "-D__STDC_LIMIT_MACROS",
        "-fno-strict-overflow",
        "-fopenmp",
    ],
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
        "@zendnn",
    ] + select({
        "@//tools/config:thread_sanitizer": [],
        "//conditions:default": ["@tbb"],
    }),
)

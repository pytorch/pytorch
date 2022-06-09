load("@rules_cc//cc:defs.bzl", "cc_library")
load("@//third_party:substitution.bzl", "template_rule")

licenses(["notice"])  # Apache 2.0

template_rule(
    name = "version_string",
    src = "@//:aten/src/ATen/cpu/tbb/extra/version_string.ver.in",
    out = "version_string.h",
    substitutions = {
        "@CMAKE_SYSTEM_NAME@": "Unknown",
        "@CMAKE_SYSTEM@": "Unknown",
        "@CMAKE_SYSTEM_VERSION@": "Unknown",
        "@CMAKE_CXX_COMPILER_ID@": "Unknown",
        "@_configure_date@": "Unknown",
    }
)

cc_library(
    name = "tbb",
    srcs = [":version_string"] + glob(
        [
            "src/old/*.h",
            "src/rml/client/*.h",
            "src/rml/include/*.h",
            "src/rml/server/*.h",
            "src/tbb/*.h",
            "src/tbb/tools_api/*.h",
            "src/tbb/tools_api/legacy/*.h",
            "src/old/*.cpp",
            "src/tbb/*.cpp",
        ],
        exclude = ["src/old/test_*.cpp"],
    ) + ["src/rml/client/rml_tbb.cpp"],
    hdrs = glob(
        [
            "include/tbb/*",
            "include/tbb/compat/*",
            "include/tbb/internal/*",
            "include/tbb/machine/*",
        ],
        exclude = ["include/tbb/scalable_allocator.h"],
    ),
    copts = [
        "-Iexternal/tbb/src/rml/include",
        "-Iexternal/tbb/src",
        "-pthread",
        "-DDO_ITT_NOTIFY=1",
        "-DUSE_PTHREAD=1",
        "-D__TBB_BUILD=1",
        "-D__TBB_DYNAMIC_LOAD_ENABLED=0",
        "-D__TBB_SOURCE_DIRECTLY_INCLUDED=1",
        "-fno-sanitize=vptr",
        "-fno-sanitize=thread",
    ],
    defines = [
        # TBB Cannot detect the standard library version when using clang with libstdc++.
        # See https://github.com/01org/tbb/issues/22
        "TBB_USE_GLIBCXX_VERSION=(_GLIBCXX_RELEASE*10000)",
        "TBB_PREVIEW_GLOBAL_CONTROL=1",
        "TBB_PREVIEW_LOCAL_OBSERVER=1",
        "__TBB_ALLOW_MUTABLE_FUNCTORS=1",
    ],
    includes = [
        "include",
        "src/tbb/tools_api",
    ],
    linkopts = [
        "-ldl",
        "-lpthread",
        "-lrt",
    ],
    textual_hdrs = ["src/tbb/tools_api/ittnotify_static.c"],
    visibility = ["//visibility:public"],
)

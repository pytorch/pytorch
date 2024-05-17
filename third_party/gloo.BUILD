load("@rules_cc//cc:defs.bzl", "cc_library")
load("@pytorch//tools/rules:cu.bzl", "cu_library")
load("@pytorch//third_party:substitution.bzl", "template_rule")
load("@pytorch//tools/config:defs.bzl", "if_cuda")

template_rule(
    name = "gloo_config_cmake_macros",
    src = "gloo/config.h.in",
    out = "gloo/config.h",
    substitutions = {
        "@GLOO_VERSION_MAJOR@": "0",
        "@GLOO_VERSION_MINOR@": "5",
        "@GLOO_VERSION_PATCH@": "0",
        "cmakedefine01 GLOO_USE_CUDA": "define GLOO_USE_CUDA 1",
        "cmakedefine01 GLOO_USE_NCCL": "define GLOO_USE_NCCL 0",
        "cmakedefine01 GLOO_USE_ROCM": "define GLOO_USE_ROCM 0",
        "cmakedefine01 GLOO_USE_RCCL": "define GLOO_USE_RCCL 0",
        "cmakedefine01 GLOO_USE_REDIS": "define GLOO_USE_REDIS 0",
        "cmakedefine01 GLOO_USE_IBVERBS": "define GLOO_USE_IBVERBS 0",
        "cmakedefine01 GLOO_USE_MPI": "define GLOO_USE_MPI 0",
        "cmakedefine01 GLOO_USE_AVX": "define GLOO_USE_AVX 0",
        "cmakedefine01 GLOO_USE_LIBUV": "define GLOO_USE_LIBUV 0",
        # The `GLOO_HAVE_TRANSPORT_TCP_TLS` line should go above the `GLOO_HAVE_TRANSPORT_TCP` in order to properly substitute the template.
        "cmakedefine01 GLOO_HAVE_TRANSPORT_TCP_TLS": "define GLOO_HAVE_TRANSPORT_TCP_TLS 1",
        "cmakedefine01 GLOO_HAVE_TRANSPORT_TCP": "define GLOO_HAVE_TRANSPORT_TCP 1",
        "cmakedefine01 GLOO_HAVE_TRANSPORT_IBVERBS": "define GLOO_HAVE_TRANSPORT_IBVERBS 0",
        "cmakedefine01 GLOO_HAVE_TRANSPORT_UV": "define GLOO_HAVE_TRANSPORT_UV 0",
    },
)

cc_library(
    name = "gloo_headers",
    hdrs = glob(
        [
            "gloo/*.h",
            "gloo/common/*.h",
            "gloo/rendezvous/*.h",
            "gloo/transport/*.h",
            "gloo/transport/tcp/*.h",
            "gloo/transport/tcp/tls/*.h",
        ],
        exclude = [
            "gloo/rendezvous/redis_store.h",
        ],
    ) + ["gloo/config.h"],
    includes = [
        ".",
    ],
)

cu_library(
    name = "gloo_cuda",
    srcs = [
        "gloo/cuda.cu",
        "gloo/cuda_private.cu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":gloo_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "gloo",
    srcs = glob(
        [
            "gloo/*.cc",
            "gloo/common/*.cc",
            "gloo/rendezvous/*.cc",
            "gloo/transport/*.cc",
            "gloo/transport/tcp/*.cc",
        ],
        exclude = [
            "gloo/cuda*.cc",
            "gloo/common/win.cc",
            "gloo/rendezvous/redis_store.cc",
        ]
    ) + if_cuda(glob(["gloo/cuda*.cc"])),
    copts = [
        "-std=c++17",
    ],
    visibility = ["//visibility:public"],
    deps = [":gloo_headers"] + if_cuda(
        [":gloo_cuda"],
        [],
    ),
)

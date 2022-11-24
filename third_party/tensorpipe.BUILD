load("@rules_cc//cc:defs.bzl", "cc_library")
load("@pytorch//third_party:substitution.bzl", "header_template_rule")

LIBUV_COMMON_SRCS = [
    "third_party/libuv/src/fs-poll.c",
    "third_party/libuv/src/idna.c",
    "third_party/libuv/src/inet.c",
    "third_party/libuv/src/random.c",
    "third_party/libuv/src/strscpy.c",
    "third_party/libuv/src/threadpool.c",
    "third_party/libuv/src/timer.c",
    "third_party/libuv/src/uv-common.c",
    "third_party/libuv/src/uv-data-getter-setters.c",
    "third_party/libuv/src/version.c",
]

LIBUV_POSIX_SRCS = [
    "third_party/libuv/src/unix/async.c",
    "third_party/libuv/src/unix/core.c",
    "third_party/libuv/src/unix/dl.c",
    "third_party/libuv/src/unix/fs.c",
    "third_party/libuv/src/unix/getaddrinfo.c",
    "third_party/libuv/src/unix/getnameinfo.c",
    "third_party/libuv/src/unix/loop.c",
    "third_party/libuv/src/unix/loop-watcher.c",
    "third_party/libuv/src/unix/pipe.c",
    "third_party/libuv/src/unix/poll.c",
    "third_party/libuv/src/unix/process.c",
    "third_party/libuv/src/unix/random-devurandom.c",
    "third_party/libuv/src/unix/signal.c",
    "third_party/libuv/src/unix/stream.c",
    "third_party/libuv/src/unix/tcp.c",
    "third_party/libuv/src/unix/thread.c",
    "third_party/libuv/src/unix/tty.c",
    "third_party/libuv/src/unix/udp.c",
]

LIBUV_LINUX_SRCS = LIBUV_POSIX_SRCS + [
    "third_party/libuv/src/unix/proctitle.c",
    "third_party/libuv/src/unix/linux-core.c",
    "third_party/libuv/src/unix/linux-inotify.c",
    "third_party/libuv/src/unix/linux-syscalls.c",
    "third_party/libuv/src/unix/procfs-exepath.c",
    "third_party/libuv/src/unix/random-getrandom.c",
    "third_party/libuv/src/unix/random-sysctl-linux.c",
]

cc_library(
    name = "libuv",
    srcs = LIBUV_COMMON_SRCS + LIBUV_LINUX_SRCS,
    includes = [
        "third_party/libuv/include",
        "third_party/libuv/src",
    ],
    hdrs = glob(
        [
            "third_party/libuv/include/*.h",
            "third_party/libuv/include/uv/*.h",
            "third_party/libuv/src/*.h",
            "third_party/libuv/src/unix/*.h",
        ],
    ),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libnop",
    srcs = [],
    includes = ["third_party/libnop/include"],
    hdrs = glob(["third_party/libnop/include/**/*.h"]),
)

header_template_rule(
    name = "tensorpipe_cpu_config_header",
    src = "tensorpipe/config.h.in",
    out = "tensorpipe/config.h",
    substitutions = {
        "#cmakedefine01 TENSORPIPE_HAS_SHM_TRANSPORT": "#define TENSORPIPE_HAS_SHM_TRANSPORT 1",
        "#cmakedefine01 TENSORPIPE_HAS_IBV_TRANSPORT": "#define TENSORPIPE_HAS_IBV_TRANSPORT 1",
        "#cmakedefine01 TENSORPIPE_HAS_CMA_CHANNEL": "#define TENSORPIPE_HAS_CMA_CHANNEL 1",
    },
)

header_template_rule(
    name = "tensorpipe_cuda_config_header",
    src = "tensorpipe/config_cuda.h.in",
    out = "tensorpipe/config_cuda.h",
    substitutions = {
        "#cmakedefine01 TENSORPIPE_HAS_CUDA_IPC_CHANNEL": "#define TENSORPIPE_HAS_CUDA_IPC_CHANNEL 1",
        "#cmakedefine01 TENSORPIPE_HAS_CUDA_GDR_CHANNEL": "#define TENSORPIPE_HAS_CUDA_GDR_CHANNEL 1",
    },
)

# We explicitly list the CUDA headers & sources, and we consider everything else
# as CPU (using a catch-all glob). This is both because there's fewer CUDA files
# (thus making it easier to list them exhaustively) and because it will make it
# more likely to catch a misclassified file: if we forget to mark a file as CUDA
# we'll try to build it on CPU and that's likely to fail.

TENSORPIPE_CUDA_HEADERS = [
    "tensorpipe/tensorpipe_cuda.h",
    "tensorpipe/channel/cuda_basic/*.h",
    "tensorpipe/channel/cuda_gdr/*.h",
    "tensorpipe/channel/cuda_ipc/*.h",
    "tensorpipe/channel/cuda_xth/*.h",
    "tensorpipe/common/cuda.h",
    "tensorpipe/common/cuda_buffer.h",
    "tensorpipe/common/cuda_lib.h",
    "tensorpipe/common/cuda_loop.h",
    "tensorpipe/common/nvml_lib.h",
]

TENSORPIPE_CUDA_SOURCES = [
    "tensorpipe/channel/cuda_basic/*.cc",
    "tensorpipe/channel/cuda_gdr/*.cc",
    "tensorpipe/channel/cuda_ipc/*.cc",
    "tensorpipe/channel/cuda_xth/*.cc",
    "tensorpipe/common/cuda_buffer.cc",
    "tensorpipe/common/cuda_loop.cc",
]

TENSORPIPE_CPU_HEADERS = glob(
    [
        "tensorpipe/*.h",
        "tensorpipe/channel/*.h",
        "tensorpipe/channel/*/*.h",
        "tensorpipe/common/*.h",
        "tensorpipe/core/*.h",
        "tensorpipe/transport/*.h",
        "tensorpipe/transport/*/*.h",
    ],
    exclude=TENSORPIPE_CUDA_HEADERS)

TENSORPIPE_CPU_SOURCES = glob(
    [
        "tensorpipe/*.cc",
        "tensorpipe/channel/*.cc",
        "tensorpipe/channel/*/*.cc",
        "tensorpipe/common/*.cc",
        "tensorpipe/core/*.cc",
        "tensorpipe/transport/*.cc",
        "tensorpipe/transport/*/*.cc",
    ],
    exclude=TENSORPIPE_CUDA_SOURCES)

cc_library(
    name = "tensorpipe_cpu",
    srcs = TENSORPIPE_CPU_SOURCES,
    hdrs = TENSORPIPE_CPU_HEADERS + [":tensorpipe_cpu_config_header"],
    includes = [
        ".",
    ],
    copts = [
        "-std=c++14",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":libnop",
        ":libuv",
    ],
)

cc_library(
    name = "tensorpipe_cuda",
    srcs = glob(TENSORPIPE_CUDA_SOURCES),
    hdrs = glob(TENSORPIPE_CUDA_HEADERS) + [":tensorpipe_cuda_config_header"],
    includes = [
        ".",
    ],
    copts = [
        "-std=c++14",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":tensorpipe_cpu",
        "@cuda",
    ],
)

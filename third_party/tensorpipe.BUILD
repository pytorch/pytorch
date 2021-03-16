load("@rules_cc//cc:defs.bzl", "cc_library")
load("@//third_party:substitution.bzl", "header_template_rule")

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
    name = "tensorpipe_config_header",
    src = "tensorpipe/config.h.in",
    out = "tensorpipe/config.h",
    substitutions = {
        "#cmakedefine01 TENSORPIPE_HAS_SHM_TRANSPORT": "",
        "#cmakedefine01 TENSORPIPE_HAS_CMA_CHANNEL": "",
        "#cmakedefine01 TENSORPIPE_HAS_CUDA_IPC_CHANNEL": "",
        "#cmakedefine01 TENSORPIPE_HAS_IBV_TRANSPORT": "",
        "#cmakedefine01 TENSORPIPE_SUPPORTS_CUDA": "",
    },
)

TENSORPIPE_HEADERS = glob([
    "tensorpipe/*.h",
    "tensorpipe/channel/*.h",
    "tensorpipe/channel/*/*.h",
    "tensorpipe/common/*.h",
    "tensorpipe/core/*.h",
    "tensorpipe/transport/*.h",
    "tensorpipe/transport/*/*.h",
    "tensorpipe/util/*/*.h",
])

TENSORPIPE_BASE_SRCS = glob([
    "tensorpipe/*.cc",
    "tensorpipe/channel/*.cc",
    "tensorpipe/common/address.cc",
    "tensorpipe/common/epoll_loop.cc",
    "tensorpipe/common/error.cc",
    "tensorpipe/common/fd.cc",
    "tensorpipe/common/ibv.cc",
    "tensorpipe/common/socket.cc",
    "tensorpipe/common/system.cc",
    "tensorpipe/core/*.cc",
    "tensorpipe/transport/*.cc",
    "tensorpipe/util/*/*.cc",
])

TENSORPIPE_SRCS = TENSORPIPE_BASE_SRCS + glob([
    "tensorpipe/channel/basic/*.cc",
    "tensorpipe/channel/mpt/*.cc",
    "tensorpipe/channel/xth/*.cc",
    "tensorpipe/transport/uv/*.cc",
])

TENSORPIPE_SRCS_CUDA = TENSORPIPE_SRCS + glob([
    "tensorpipe/common/cuda_loop.cc",
    "tensorpipe/channel/cuda_basic/*.cc",
    "tensorpipe/channel/cuda_ipc/*.cc",
    "tensorpipe/channel/cuda_xth/*.cc",
])

cc_library(
    name = "tensorpipe",
    srcs = TENSORPIPE_SRCS + [":tensorpipe_config_header"],
    hdrs = TENSORPIPE_HEADERS,
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
    srcs = TENSORPIPE_SRCS_CUDA + [":tensorpipe_config_header"],
    hdrs = TENSORPIPE_HEADERS,
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
        "@cuda",
    ],
)

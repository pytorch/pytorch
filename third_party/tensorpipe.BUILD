load("@rules_cc//cc:defs.bzl", "cc_library")
load("@//third_party:substitution.bzl", "template_rule")

LIBUV_COMMON_SRCS = [
    "third_party/libuv/src/fs-poll.c",
    "third_party/libuv/src/idna.c",
    "third_party/libuv/src/inet.c",
    "third_party/libuv/src/strscpy.c",
    "third_party/libuv/src/threadpool.c",
    "third_party/libuv/src/timer.c",
    "third_party/libuv/src/uv-data-getter-setters.c",
    "third_party/libuv/src/uv-common.c",
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
    "third_party/libuv/src/unix/signal.c",
    "third_party/libuv/src/unix/stream.c",
    "third_party/libuv/src/unix/tcp.c",
    "third_party/libuv/src/unix/thread.c",
    "third_party/libuv/src/unix/tty.c",
    "third_party/libuv/src/unix/udp.c",
]

LIBUV_LINUX_SRCS = LIBUV_POSIX_SRCS + [
    "third_party/libuv/src/unix/linux-core.c",
    "third_party/libuv/src/unix/linux-inotify.c",
    "third_party/libuv/src/unix/linux-syscalls.c",
    "third_party/libuv/src/unix/procfs-exepath.c",
    "third_party/libuv/src/unix/sysinfo-loadavg.c",
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

proto_library(
    name = "tensorpipe_proto_source",
    srcs = glob([
        "tensorpipe/proto/*.proto",
        "tensorpipe/proto/*/*.proto",
    ]),
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "tensorpipe_protos",
    deps = [":tensorpipe_proto_source"],
)

template_rule(
    name = "tensorpipe_header_template",
    src = "tensorpipe/tensorpipe.h.in",
    out = "tensorpipe/tensorpipe.h",
    substitutions = {
        "cmakedefine01 TENSORPIPE_HAS_SHM_TRANSPORT": "define TENSORPIPE_HAS_SHM_TRANSPORT 0",
        "cmakedefine01 TENSORPIPE_HAS_CMA_CHANNEL": "define TENSORPIPE_HAS_CMA_CHANNEL 0",
    },
)

cc_library(
    name = "tensorpipe",
    srcs = glob(
        [
            "tensorpipe/*.cc",
            "tensorpipe/channel/*.cc",
            "tensorpipe/channel/*/*.cc",
            "tensorpipe/common/*.cc",
            "tensorpipe/core/*.cc",
            "tensorpipe/transport/*.cc",
            "tensorpipe/transport/*/*.cc",
            "tensorpipe/util/*/*.cc",
        ],
    ),
    hdrs = glob(
        [
            "tensorpipe/*.h",
            "tensorpipe/channel/*.h",
            "tensorpipe/channel/*/*.h",
            "tensorpipe/common/*.h",
            "tensorpipe/core/*.h",
            "tensorpipe/transport/*.h",
            "tensorpipe/transport/*/*.h",
            "tensorpipe/util/*/*.h",
        ],
    ),
    includes = [
        ".",
    ],
    copts = [
        "-std=c++14",
    ],
    visibility = ["//visibility:public"],
    deps = [":tensorpipe_protos", ":libuv"],
)

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "clog",
    srcs = [
        "deps/clog/src/clog.c",
    ],
    hdrs = glob([
        "deps/clog/include/*.h",
    ]),
    includes = [
        "deps/clog/include/",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cpuinfo",
    srcs = glob(
        [
            "src/*.c",
            "src/linux/*.c",
            "src/x86/*.c",
            "src/x86/cache/*.c",
            "src/x86/linux/*.c",
        ],
        exclude = [
            "src/x86/mockcpuid.c",
            "src/linux/mockfile.c",
        ],
    ),
    hdrs = glob([
        "include/*.h",
        "src/*.h",
        "src/cpuinfo/*.h",
        "src/include/*.h",
        "src/x86/*.h",
        "src/x86/linux/*.h",
        "src/linux/*.h",
    ]),
    copts = [
        "-DCPUINFO_LOG_LEVEL=2",
        "-DTH_BLAS_MKL",
        "-D_GNU_SOURCE=1",
    ],
    includes = [
        "include",
        "src",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":clog",
    ],
)

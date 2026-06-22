def define_targets(rules):
    rules.cc_library(
        name = "CPUCachingAllocator",
        srcs = ["CPUCachingAllocator.cpp"],
        hdrs = ["CPUCachingAllocator.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            "//c10/core:alloc_cpu",
            "//c10/util:base",
        ],
    )

    rules.cc_library(
        name = "CPUProfilingAllocator",
        srcs = ["CPUProfilingAllocator.cpp"],
        hdrs = ["CPUProfilingAllocator.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            "//c10/core:alloc_cpu",
            "//c10/util:base",
        ],
    )

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(["*.h"]),
        visibility = ["//c10:__pkg__"],
    )

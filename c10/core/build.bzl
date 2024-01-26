def define_targets(rules):
    rules.cc_library(
        name = "CPUAllocator",
        srcs = ["CPUAllocator.cpp"],
        hdrs = ["CPUAllocator.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":alignment",
            ":base",
            "//c10/mobile:CPUCachingAllocator",
            "//c10/mobile:CPUProfilingAllocator",
            "//c10/util:base",
        ],
        # This library defines a flag, The use of alwayslink keeps it
        # from being stripped.
        alwayslink = True,
    )

    rules.cc_library(
        name = "MapAllocator",
        srcs = ["MapAllocator.cpp"],
        hdrs = ["MapAllocator.h"],
        linkstatic = True,
        local_defines = [
            "C10_BUILD_MAIN_LIB",
            "HAVE_MMAP=1",
            "HAVE_SHM_OPEN=1",
            "HAVE_SHM_UNLINK=1",
        ],
        visibility = ["//visibility:public"],
        deps = [
            ":CPUAllocator",
            ":base",
            "//c10/util:base",
        ],
        linkopts = ["-lrt"],
        alwayslink = True,
    )

    rules.cc_library(
        name = "ScalarType",
        hdrs = ["ScalarType.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = ["//c10/util:base"],
    )

    rules.cc_library(
        name = "alignment",
        hdrs = ["alignment.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
    )

    rules.cc_library(
        name = "alloc_cpu",
        srcs = ["impl/alloc_cpu.cpp"],
        hdrs = ["impl/alloc_cpu.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":alignment",
            "//c10/macros",
            "//c10/util:base",
        ],
        # This library defines flags, The use of alwayslink keeps them
        # from being stripped.
        alwayslink = True,
    )

    rules.cc_library(
        name = "base",
        srcs = rules.glob(
            [
                "*.cpp",
                "impl/*.cpp",
            ],
            exclude = [
                "CPUAllocator.cpp",
                "MapAllocator.cpp",
                "impl/alloc_cpu.cpp",
            ],
        ),
        hdrs = rules.glob(
            [
                "*.h",
                "impl/*.h",
            ],
            exclude = [
                "CPUAllocator.h",
                "MapAllocator.h",
                "impl/alloc_cpu.h",
            ],
        ),
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":ScalarType",
            "//third_party/cpuinfo",
            "//c10/macros",
            "//c10/util:TypeCast",
            "//c10/util:base",
            "//c10/util:typeid",
        ],
        # This library uses flags and registration. Do not let the
        # linker remove them.
        alwayslink = True,
    )

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(
            [
                "*.h",
                "impl/*.h",
            ],
            exclude = [
                "alignment.h",
            ],
        ),
        visibility = ["//c10:__pkg__"],
    )

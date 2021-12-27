def define_targets(rules):
    rules.cc_library(
        name = "ScalarType",
        hdrs = ["ScalarType.h"],
        visibility = ["//visibility:public"],
        deps = ["//c10/util:base"],
    )

    rules.cc_library(
        name = "alignment",
        hdrs = ["alignment.h"],
        visibility = ["//visibility:public"],
    )

    rules.cc_library(
        name = "alloc_cpu",
        srcs = ["impl/alloc_cpu.cpp"],
        hdrs = ["impl/alloc_cpu.h"],
        visibility = ["//visibility:public"],
        deps = [
            ":alignment",
            "//c10/macros:macros",
            "//c10/util:base",
        ],
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
                "impl/alloc_cpu.h",
            ],
        ),
        visibility = ["//visibility:public"],
        deps = [
            ":ScalarType",
            "//c10/macros:macros",
            "//c10/util:TypeCast",
            "//c10/util:base",
            "//c10/util:typeid",
        ],
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

    rules.filegroup(
        name = "sources",
        srcs = ["CPUAllocator.cpp"],
        visibility = ["//c10:__pkg__"],
    )

def define_targets(rules, c10_name = "c10"):
    rules.cc_library(
        name = c10_name,
        visibility = ["//visibility:public"],
        deps = [
            "//c10/core:CPUAllocator",
            "//c10/core:ScalarType",
            "//c10/core:alignment",
            "//c10/core:alloc_cpu",
            "//c10/core:base",
            "//c10/macros",
            "//c10/mobile:CPUCachingAllocator",
            "//c10/mobile:CPUProfilingAllocator",
            "//c10/util:TypeCast",
            "//c10/util:base",
            "//c10/util:typeid",
        ] + rules.if_cuda(
            [
                "//c10/cuda:cuda",
                "//c10/cuda:Macros",
            ],
            [],
        ),
    )

    rules.cc_library(
        name = c10_name + "_headers",
        deps = [
            "//c10/core:base_headers",
            "//c10/macros",
            "//c10/util:base_headers",
            "//c10/util:bit_cast",
        ],
        visibility = ["//visibility:public"],
    )

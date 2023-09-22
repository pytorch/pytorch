def define_targets(rules):
    rules.cc_library(
        name = "c10",
        visibility = ["//visibility:public"],
        deps = [
            "//c10/core:CPUAllocator",
            "//c10/core:MapAllocator",
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

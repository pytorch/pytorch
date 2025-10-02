def define_targets(rules):
    rules.filegroup(
        name = "version.h.in",
        srcs = ["version.h.in"],
        visibility = ["//visibility:public"],
    )

    rules.cc_library(
        name = "torch_headeronly",
        hdrs = rules.glob([
            "**/*.h"
        ]),
        visibility = ["//visibility:public"],
        deps = [
            "//torch/headeronly/macros",
        ],
    )

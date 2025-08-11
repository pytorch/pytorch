def define_targets(rules):
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

def define_targets(rules):
    rules.cc_library(
        name = "torch_headeronly",
        visibility = ["//visibility:public"],
        deps = [
            "//torch/headeronly/macros",
        ],
    )

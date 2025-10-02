def define_targets(rules):
    rules.filegroup(
        name = "version.h.in",
        srcs = ["version.h.in"],
        visibility = ["//visibility:public"],
    )

    # Alias for the generated version.h from the root package
    rules.alias(
        name = "version.h",
        actual = "//:version_h",
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

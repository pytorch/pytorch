def define_targets(rules):
    rules.filegroup(
        name = "pyi.in",
        srcs = rules.glob(["*.pyi.in"]),
        visibility = ["//visibility:public"],
    )

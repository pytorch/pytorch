def define_targets(rules):
    rules.py_library(
        name = "torchgen",
        srcs = rules.glob(["**/*.py"]),
        visibility = ["//visibility:public"],
        deps = [
            rules.requirement("pyyaml"),
            rules.requirement("typing-extensions"),
        ],
    )

    rules.py_binary(
        name = "gen",
        srcs = [":torchgen"],
        visibility = ["//visibility:public"],
        deps = [
            rules.requirement("pyyaml"),
            rules.requirement("typing-extensions"),
        ],
    )

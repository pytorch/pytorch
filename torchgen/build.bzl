def define_targets(rules):
    rules.py_library(
        name = "torchgen",
        srcs = rules.glob(["**/*.py"]),
        deps = [
            rules.requirement("PyYAML"),
            rules.requirement("typing-extensions"),
            rules.requirement("mypy-extensions"),
        ],
        visibility = ["//visibility:public"],
    )

    rules.py_binary(
        name = "gen",
        srcs = [":torchgen"],
        visibility = ["//visibility:public"],
    )

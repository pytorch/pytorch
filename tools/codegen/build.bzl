def define_targets(rules):
    rules.py_library(
        name = "codegen",
        srcs = rules.glob(["**/*.py"]),
        deps = [
            rules.requirement("PyYAML"),
            rules.requirement("typing-extensions"),
        ],
        visibility = ["//visibility:public"],
    )

    rules.py_binary(
        name = "gen",
        srcs = [":codegen"],
        visibility = ["//visibility:public"],
    )

def define_targets(rules):
    rules.py_library(
        name = "codegen",
        srcs = glob(["**/*.py"]),
        deps = [
            rules.requirement("PyYAML"),
            rules.requirement("typing-extensions"),
        ],
        visibility = ["//visibility:public"],
    )

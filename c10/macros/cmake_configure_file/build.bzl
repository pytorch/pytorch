def define_targets(rules):
    rules.py_binary(
        name = "tool",
        srcs = ["tool.py"],
        visibility = ["//:__subpackages__"],
    )

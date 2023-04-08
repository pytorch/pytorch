def define_targets(rules):
    rules.filegroup(
        name = "tool",
        srcs = ["tool.py"],
        visibility = [":__subpackages__"],
    )

def define_targets(rules):
    rules.py_library(
        name = "autograd",
        srcs = rules.glob(["*.py"]),
        data = rules.glob([
            "*.yaml",
            "templates/*",
        ]),
        visibility = ["//:__subpackages__"],
        deps = [
            rules.requirement("PyYAML"),
            "//torchgen",
        ],
    )

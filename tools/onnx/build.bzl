def define_targets(rules):
    rules.py_library(
        name = "onnx",
        srcs = rules.glob(["gen_diagnostics.py"]),
        data = rules.glob([
            "templates/*",
        ]),
        visibility = ["//:__subpackages__"],
        deps = [
            rules.requirement("PyYAML"),
            "//torchgen:torchgen",
        ],
    )

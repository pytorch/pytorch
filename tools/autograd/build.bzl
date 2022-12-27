load("@//tools:cruise_rules.bzl", "cruise_py_library")
load("@//tools/py_hermetic_rules:py_hermetic_rules.bzl", "pip3_dep")

def define_targets(rules):
    cruise_py_library(
        name = "autograd",
        srcs = rules.glob(["*.py"]),
        data = rules.glob([
            "*.yaml",
            "templates/*",
        ]),
        visibility = ["//:__subpackages__"],
        deps = [
            pip3_dep("pyyaml"),
            "//torchgen:torchgen",
        ],
        import_from_root = True,
    )

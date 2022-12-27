load("@//tools:cruise_rules.bzl", "cruise_py_binary", "cruise_py_library")
load("@//tools/py_hermetic_rules:py_hermetic_rules.bzl", "pip3_dep")

def define_targets(rules):
    cruise_py_library(
        name = "torchgen",
        srcs = rules.glob(
            ["**/*.py"],
        ),
        deps = [
            # rules.requirement("PyYAML"),
            # rules.requirement("typing-extensions"),
            pip3_dep("pyyaml"),
            pip3_dep("typing_extensions"),
        ],
        import_from_root = False,
        strip_import_prefix = "pytorch",
        absolute_import_prefix = True,
        visibility = ["//visibility:public"],
    )

    cruise_py_binary(
        name = "gen",
        srcs = ["gen.py"],
        deps = [":torchgen"],
        visibility = ["//visibility:public"],
    )

    rules.py_binary(
        name = "gen_executorch",
        srcs = [":torchgen"],
        visibility = ["//visibility:public"],
    )

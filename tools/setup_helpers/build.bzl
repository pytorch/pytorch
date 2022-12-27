load("@//tools:cruise_rules.bzl", "cruise_py_binary")
load("@//tools/py_hermetic_rules:py_hermetic_rules.bzl", "pip3_dep")

def define_targets(rules):
    cruise_py_binary(
        name = "generate_code",
        srcs = ["generate_code.py"],
        visibility = ["//:__pkg__"],
        deps = [
            pip3_dep("pyyaml"),
            "//tools/autograd",
            "//torchgen",
        ],
        data = [
            "@pytorch//:torch/csrc/lazy/ts_backend/ts_native_functions.cpp"
        ],
    )

    cruise_py_binary(
        name = "gen_version_header",
        srcs = ["gen_version_header.py"],
        visibility = ["//:__pkg__"],
    )

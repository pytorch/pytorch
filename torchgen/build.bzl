def define_targets(rules):
    rules.py_library(
        name = "torchgen",
        srcs = rules.glob(["**/*.py"]),
        data = [
            "//caffe2:native_functions.yaml",
            "//caffe2:tags.yaml",
        ],
        deps = [
            rules.requirement("PyYAML"),
            rules.requirement("typing-extensions"),
        ],
        visibility = ["//visibility:public"],
    )

    rules.py_binary(
        name = "gen",
        srcs = [":torchgen"],
        visibility = ["//visibility:public"],
    )

def define_targets(rules):
    rules.py_binary(
        name = "generate_code",
        srcs = ["generate_code.py"],
        visibility = ["//:__pkg__"],
        deps = [
            rules.requirement("PyYAML"),
            rules.requirement("setuptools"),
            "//tools/autograd",
            "//tools/codegen",
        ],
    )

    rules.py_binary(
        name = "gen_version_header",
        srcs = ["gen_version_header.py"],
        visibility = ["//:__pkg__"],
    )

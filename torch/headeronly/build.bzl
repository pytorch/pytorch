def define_targets(rules):
    rules.genrule(
        name = "version_h",
        srcs = [
            "version.h.in",
            "//:version.txt",
        ],
        outs = ["version.h"],
        cmd = "$(execpath //tools/setup_helpers:gen_version_header) " +
              "--template-path version.h.in " +
              "--version-path $(location //:version.txt) --output-path $@ ",
        tools = ["//tools/setup_helpers:gen_version_header"],
    )

    rules.cc_library(
        name = "torch_headeronly",
        hdrs = rules.glob([
            "**/*.h"
        ]),
        visibility = ["//visibility:public"],
        deps = [
            "//torch/headeronly/macros",
        ],
    )

def define_targets(rules):
    # workaround issue where open source bazel requires $(location ...)
    # for filepaths but the buck conversion requires no $(location ...)
    # for filepaths.
    is_buck = hasattr(native, "read_config")
    template_arg = "version.h.in" if is_buck else "$(location version.h.in)"

    genrule_args = {
        "name": "version_h",
        "srcs": [
            "version.h.in",
            "//:version.txt",
        ],
        "outs": ["version.h"],
        "cmd": "$(execpath //tools/setup_helpers:gen_version_header) " +
               "--template-path " + template_arg + " " +
               "--version-path $(location //:version.txt) --output-path $@ ",
        "tools": ["//tools/setup_helpers:gen_version_header"],
    }

    # Add visibility only for Bazel, buck genrule in fbcode.bzl does not
    # support this argument
    if not is_buck:
        genrule_args["visibility"] = ["//visibility:public"]

    rules.genrule(**genrule_args)

    rules.cc_library(
        name = "torch_headeronly",
        hdrs = rules.glob([
            "**/*.h"
        ]) + ["version.h.in"],
        visibility = ["//visibility:public"],
        deps = [
            "//torch/headeronly/macros",
        ],
    )

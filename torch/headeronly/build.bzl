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

    # Generate enum_tag.h using torchgen
    enum_tag_genrule_args = {
        "name": "enum_tag_h",
        "srcs": [
            "//aten/src/ATen/native:native_functions.yaml",
            "//aten/src/ATen/native:tags.yaml",
            "templates/enum_tag.h",
        ],
        "outs": ["core/enum_tag.h"],
        "cmd": "$(execpath //torchgen:gen) " +
               "--source-path aten/src/ATen " +
               "--headeronly " +
               "--headeronly-install-dir $(@D)/..",
        "tools": ["//torchgen:gen"],
    }

    if not is_buck:
        enum_tag_genrule_args["visibility"] = ["//visibility:public"]

    rules.genrule(**enum_tag_genrule_args)

    rules.cc_library(
        name = "torch_headeronly",
        hdrs = rules.glob([
            "**/*.h"
        ]) + ["version.h.in", "core/enum_tag.h"],
        visibility = ["//visibility:public"],
        deps = [
            "//torch/headeronly/macros",
        ],
    )

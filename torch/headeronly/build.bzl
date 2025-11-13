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

    # Generate enum_tag.h from tags.yaml using torchgen
    # This file is generated in the torch/headeronly subpackage to be at the correct
    # path (torch/headeronly/core/enum_tag.h) that the ATen forwarding header expects.
    enum_tag_genrule_args = {
        "name": "enum_tag_h",
        "srcs": [
            "//:aten/src/ATen/native/tags.yaml",
        ] + rules.glob(["templates/*"]),
        "outs": ["core/enum_tag.h"],
        "cmd": " && ".join([
            # Create a temporary source directory structure
            "SRCDIR=$$(mktemp -d)",
            "mkdir -p $$SRCDIR/aten/src/ATen/native",
            "cp $(location //:aten/src/ATen/native/tags.yaml) $$SRCDIR/aten/src/ATen/native/",
            # Create template directory
            "mkdir -p $$SRCDIR/torch/headeronly/templates",
            "cp $(location templates/enum_tag.h) $$SRCDIR/torch/headeronly/templates/",
            # Run torchgen to generate only headeronly files
            "$(execpath //torchgen:gen) " +
            "--source-path $$SRCDIR/aten/src/ATen " +
            "--install_dir $$(dirname $$(dirname $$(@D))) " +
            "--generate headeronly",
            # Clean up
            "rm -rf $$SRCDIR",
        ]),
        "tools": ["//torchgen:gen"],
    }

    if not is_buck:
        enum_tag_genrule_args["visibility"] = ["//visibility:public"]

    rules.genrule(**enum_tag_genrule_args)

    rules.cc_library(
        name = "torch_headeronly",
        hdrs = rules.glob([
            "**/*.h"
        ]) + [
            "version.h.in",
            ":enum_tag_h",  # Reference the locally generated enum_tag.h
        ],
        visibility = ["//visibility:public"],
        deps = [
            "//torch/headeronly/macros",
        ],
    )

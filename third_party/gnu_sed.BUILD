load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

genrule(
    name = "gnu_sed",
    # Produce a relative symlink from the output to the built binary.
    cmd = "srcs='$<' && cd $(RULEDIR) && ln -s $${srcs#$(RULEDIR)/} $$(basename $(OUTS))",
    srcs = [":filegroup"],
    outs = ["sed.bin"],
    executable = True,
    visibility = ["//visibility:public"],
)

# Create a filegroup that solely consists of the "sed" binary output
# from the sed build rule.
filegroup(
    name = "filegroup",
    srcs = [":sed"],
    output_group = "sed",
)

configure_make(
    name = "sed",
    lib_source = ":srcs",
    out_binaries = ["sed"],
    # TODO Only do this for macOS.
    env = {
        "AR": "",
    },
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

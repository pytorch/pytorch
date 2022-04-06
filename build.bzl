def define_targets(rules):
    rules.genrule(
        name = "version_h",
        srcs = [
            ":torch/csrc/api/include/torch/version.h.in",
            ":version.txt"
        ],
        outs = ["torch/csrc/api/include/torch/version.h"],
        cmd = "$(location //caffe2/tools/setup_helpers:gen_version_header) " +
              "--template-path $(location :torch/csrc/api/include/torch/version.h.in) " +
              "--version-path $(location :version.txt) --output-path $@ ",
        tools = ['//caffe2/tools/setup_helpers:gen_version_header'],
    )

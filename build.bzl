def define_targets(rules):
    rules.cc_library(
        name = "caffe2_serialize",
        srcs = [
            "caffe2/serialize/file_adapter.cc",
            "caffe2/serialize/inline_container.cc",
            "caffe2/serialize/istream_adapter.cc",
            "caffe2/serialize/read_adapter_interface.cc",
        ],
        tags = [
            "supermodule:android/default/pytorch",
            "supermodule:ios/default/public.pytorch",
            "xplat",
        ],
        visibility = ["//visibility:public"],
        deps = [
            ":caffe2_headers",
            "@com_github_glog//:glog",
            "//c10",
            "//third_party/miniz-2.0.8:miniz",
        ],
    )

    rules.genrule(
        name = "version_h",
        srcs = [
            ":torch/csrc/api/include/torch/version.h.in",
            ":version.txt",
        ],
        outs = ["torch/csrc/api/include/torch/version.h"],
        cmd = "$(location //tools/setup_helpers:gen_version_header) " +
              "--template-path $(location :torch/csrc/api/include/torch/version.h.in) " +
              "--version-path $(location :version.txt) --output-path $@ ",
        tools = ["//tools/setup_helpers:gen_version_header"],
    )

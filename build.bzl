def define_targets(rules):
    rules.cc_library(
        name = "caffe2_serialize",
        srcs = [
            "caffe2/serialize/file_adapter.cc",
            "caffe2/serialize/inline_container.cc",
            "caffe2/serialize/istream_adapter.cc",
            "caffe2/serialize/read_adapter_interface.cc",
        ],
        # Flags that end with '()' are converted to a list of strings
        # by calling the function when translating in to buck/bazel.
        # TODO: find a better way to do this.
        copts = ["get_c2_fbandroid_xplat_compiler_flags()", "-frtti"],
        compatible_with = [],
        tags = [
            "supermodule:android/default/pytorch",
            "supermodule:ios/default/public.pytorch",
        ],
        visibility = ["//visibility:public"],
        deps = [
            ":caffe2_headers",
            ":miniz",
            "//third-party/glog:glog",
            "//xplat/caffe2/c10:c10",
        ],
    )

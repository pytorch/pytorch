# Adopted from: https://github.com/tensorflow/tensorflow/blob/master/third_party/cudnn_frontend.BUILD

# Description:
# The cuDNN Frontend API is a C++ header-only library that demonstrates how
# to use the cuDNN C backend API.

load("@rules_cc//cc:defs.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["LICENSE.txt"])

cc_library(
    name = "cudnn_frontend",
    hdrs = glob(["include/**"]),
    includes = ["include/"],
    include_prefix = "third_party/cudnn_frontend",
)

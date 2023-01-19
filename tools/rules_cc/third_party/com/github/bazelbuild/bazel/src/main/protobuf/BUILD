load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

licenses(["notice"])  # Apache 2.0

py_proto_library(
    name = "crosstool_config_py_pb2",
    srcs = ["crosstool_config.proto"],
    visibility = [
        "//tools/migration:__pkg__",
    ],
)

proto_library(
    name = "crosstool_config_pb2",
    srcs = ["crosstool_config.proto"],
    visibility = [
        "//tools/migration:__pkg__",
    ],
)

go_proto_library(
    name = "crosstool_config_go_proto",
    importpath = "third_party/com/github/bazelbuild/bazel/src/main/protobuf/crosstool_config_go_proto",
    proto = ":crosstool_config_pb2",
    visibility = [
        "//tools/migration:__pkg__",
    ],
)

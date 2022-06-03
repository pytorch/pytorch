# @nolint
load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library", "oxx_test")
load("//arvr/tools/build_defs:oxx_python.bzl", "oxx_python_binary", "oxx_python_library")
load("//arvr/tools/build_defs:genrule_utils.bzl", "gen_cmake_header")
load("//arvr/tools/build_defs:protobuf.bzl", "proto_cxx_library")
load("@bazel_skylib//lib:paths.bzl", "paths")

def define_caffe2_proto():
    proto_cxx_library(
        name = "caffe2_proto_ovrsource",
        protos = [
            "caffe2/proto/caffe2.proto",
            "caffe2/proto/caffe2_legacy.proto",
            "caffe2/proto/hsm.proto",
            "caffe2/proto/metanet.proto",
            "caffe2/proto/predictor_consts.proto",
            "caffe2/proto/prof_dag.proto",
            "caffe2/proto/torch.proto",
        ],
    )

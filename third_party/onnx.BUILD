load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "gen_proto",
    srcs = ["onnx/gen_proto.py"],
    data = [
        "onnx/onnx.in.proto",
        "onnx/onnx-operators.in.proto",
        "onnx/onnx-data.in.proto",
    ],
)

genrule(
    name = "generate_onnx_proto",
    outs = [
        "onnx/onnx_onnx_torch-ml.proto",
        "onnx/onnx-ml.pb.h",
    ],
    cmd = "$(location :gen_proto) -p onnx_torch -o $(@D)/onnx onnx -m >/dev/null && sed -i 's/onnx_onnx_torch-ml.pb.h/onnx\\/onnx_onnx_torch-ml.pb.h/g' $(@D)/onnx/onnx-ml.pb.h",
    tools = [":gen_proto"],
)

genrule(
    name = "generate_onnx_operators_proto",
    outs = [
        "onnx/onnx-operators_onnx_torch-ml.proto",
        "onnx/onnx-operators-ml.pb.h",
    ],
    cmd = "$(location :gen_proto) -p onnx_torch -o $(@D)/onnx onnx-operators -m >/dev/null && sed -i 's/onnx-operators_onnx_torch-ml.pb.h/onnx\\/onnx-operators_onnx_torch-ml.pb.h/g' $(@D)/onnx/onnx-operators-ml.pb.h",
    tools = [":gen_proto"],
)

genrule(
    name = "generate_onnx_data_proto",
    outs = [
        "onnx/onnx-data_onnx_torch.proto",
        "onnx/onnx-data.pb.h",
    ],
    cmd = "$(location :gen_proto) -p onnx_torch -o $(@D)/onnx onnx-data -m >/dev/null && sed -i 's/onnx-data_onnx_torch.pb.h/onnx\\/onnx-data_onnx_torch.pb.h/g' $(@D)/onnx/onnx-data.pb.h",
    tools = [":gen_proto"],
)

cc_library(
    name = "onnx",
    srcs = glob(
        [
            "onnx/*.cc",
            "onnx/common/*.cc",
            "onnx/defs/*.cc",
            "onnx/defs/controlflow/*.cc",
            "onnx/defs/experiments/*.cc",
            "onnx/defs/generator/*.cc",
            "onnx/defs/logical/*.cc",
            "onnx/defs/math/*.cc",
            "onnx/defs/nn/*.cc",
            "onnx/defs/object_detection/*.cc",
            "onnx/defs/optional/*.cc",
            "onnx/defs/quantization/*.cc",
            "onnx/defs/reduction/*.cc",
            "onnx/defs/rnn/*.cc",
            "onnx/defs/sequence/*.cc",
            "onnx/defs/tensor/*.cc",
            "onnx/defs/traditionalml/*.cc",
            "onnx/defs/training/defs.cc",
            "onnx/shape_inference/*.cc",
            "onnx/version_converter/*.cc",
        ],
        exclude = [
            "onnx/cpp2py_export.cc",
        ],
    ),
    hdrs = glob([
        "onnx/*.h",
        "onnx/version_converter/*.h",
        "onnx/common/*.h",
        "onnx/defs/*.h",
        "onnx/defs/math/*.h",
        "onnx/defs/controlflow/*.h",
        "onnx/defs/reduction/*.h",
        "onnx/defs/tensor/*.h",
        "onnx/shape_inference/*.h",
        "onnx/version_converter/adapters/*.h",
    ]) + [
        "onnx/onnx-ml.pb.h",
        "onnx/onnx-operators-ml.pb.h",
        "onnx/onnx-data.pb.h",
    ],
    defines = [
        "ONNX_ML=1",
        "ONNX_NAMESPACE=onnx_torch",
    ],
    includes = [
        ".",
        "onnx/",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":onnx_proto_lib",
    ],
)

cc_library(
    name = "onnx_proto_headers",
    hdrs = glob([
        "onnx/*_pb.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":onnx_proto_lib",
    ],
)

proto_library(
    name = "onnx_proto",
    srcs = [
        "onnx/onnx-operators_onnx_torch-ml.proto",
        "onnx/onnx_onnx_torch-ml.proto",
        "onnx/onnx-data_onnx_torch.proto",
    ],
)

cc_proto_library(
    name = "onnx_proto_lib",
    deps = [":onnx_proto"],
)

def define_targets(rules):
    rules.cc_library(
        name = "aten_core",
        srcs = ["aten/src/ATen/core/SymIntArrayRef.cpp"],
        hdrs = [
            "aten/src/ATen/core/ATenGeneral.h",
            "aten/src/ATen/core/blob.h",
            "aten/src/ATen/core/DimVector.h",
            "aten/src/ATen/core/grad_mode.h",
            "aten/src/ATen/core/UndefinedTensorImpl.h",
            "aten/src/ATen/core/SymIntArrayRef.h",
            "aten/src/ATen/core/SymInt.h",
        ],
        strip_include_prefix = "aten/src/",
        deps = [
            "//c10",
        ],
        tags = [
            "no-caffe2-headers",
            "xplat",
        ],
        visibility = ["//:__subpackages__"],
    )

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
            "-fbcode",
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
        name = "generate-code",
        srcs = [
            ":DispatchKeyNativeFunctions.cpp",
            ":DispatchKeyNativeFunctions.h",
            ":LazyIr.h",
            ":RegisterDispatchKey.cpp",
            ":native_functions.yaml",
            ":shape_inference.h",
            ":tags.yaml",
            ":ts_native_functions.cpp",
            ":ts_native_functions.yaml",
        ],
        tools = ["//tools/setup_helpers:generate_code"],
        outs = GENERATED_AUTOGRAD_CPP + GENERATED_AUTOGRAD_PYTHON + GENERATED_TESTING_PY,
        cmd = "$(location //tools/setup_helpers:generate_code) " +
              "--gen-dir=$(RULEDIR) " +
              "--native-functions-path $(location :native_functions.yaml) " +
              "--tags-path=$(location :tags.yaml) " +
              "--gen_lazy_ts_backend",
    )

    rules.cc_library(
        name = "generated-autograd-headers",
        hdrs = [":{}".format(h) for h in _GENERATED_AUTOGRAD_CPP_HEADERS + _GENERATED_AUTOGRAD_PYTHON_HEADERS],
        visibility = ["//visibility:public"],
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

# These lists are temporarily living in and exported from the shared
# structure so that an internal build that lives under a different
# root can access them. These could technically live in a separate
# file in the same directory but that would require extra work to
# ensure that file is synced to both Meta internal repositories and
# GitHub. This problem will go away when the targets downstream of
# generate-code that use these lists are moved into the shared
# structure as well.

_GENERATED_AUTOGRAD_PYTHON_HEADERS = [
    "torch/csrc/autograd/generated/python_functions.h",
]

_GENERATED_AUTOGRAD_CPP_HEADERS = [
    "torch/csrc/autograd/generated/Functions.h",
    "torch/csrc/autograd/generated/VariableType.h",
    "torch/csrc/autograd/generated/variable_factories.h",
]

GENERATED_TESTING_PY = [
    "torch/testing/_internal/generated/annotated_fn_args.py",
]

GENERATED_LAZY_H = [
    "torch/csrc/lazy/generated/LazyIr.h",
    "torch/csrc/lazy/generated/LazyNativeFunctions.h",
]

_GENERATED_AUTOGRAD_PYTHON_CPP = [
    "torch/csrc/autograd/generated/python_functions_0.cpp",
    "torch/csrc/autograd/generated/python_functions_1.cpp",
    "torch/csrc/autograd/generated/python_functions_2.cpp",
    "torch/csrc/autograd/generated/python_functions_3.cpp",
    "torch/csrc/autograd/generated/python_functions_4.cpp",
    "torch/csrc/autograd/generated/python_nn_functions.cpp",
    "torch/csrc/autograd/generated/python_fft_functions.cpp",
    "torch/csrc/autograd/generated/python_linalg_functions.cpp",
    "torch/csrc/autograd/generated/python_return_types.cpp",
    "torch/csrc/autograd/generated/python_sparse_functions.cpp",
    "torch/csrc/autograd/generated/python_special_functions.cpp",
    "torch/csrc/autograd/generated/python_torch_functions_0.cpp",
    "torch/csrc/autograd/generated/python_torch_functions_1.cpp",
    "torch/csrc/autograd/generated/python_torch_functions_2.cpp",
    "torch/csrc/autograd/generated/python_variable_methods.cpp",
]

GENERATED_AUTOGRAD_PYTHON = _GENERATED_AUTOGRAD_PYTHON_HEADERS + _GENERATED_AUTOGRAD_PYTHON_CPP

GENERATED_AUTOGRAD_CPP = [
    "torch/csrc/autograd/generated/Functions.cpp",
    "torch/csrc/autograd/generated/VariableType_0.cpp",
    "torch/csrc/autograd/generated/VariableType_1.cpp",
    "torch/csrc/autograd/generated/VariableType_2.cpp",
    "torch/csrc/autograd/generated/VariableType_3.cpp",
    "torch/csrc/autograd/generated/VariableType_4.cpp",
    "torch/csrc/autograd/generated/TraceType_0.cpp",
    "torch/csrc/autograd/generated/TraceType_1.cpp",
    "torch/csrc/autograd/generated/TraceType_2.cpp",
    "torch/csrc/autograd/generated/TraceType_3.cpp",
    "torch/csrc/autograd/generated/TraceType_4.cpp",
    "torch/csrc/autograd/generated/ADInplaceOrViewType_0.cpp",
    "torch/csrc/autograd/generated/ADInplaceOrViewType_1.cpp",
    "torch/csrc/lazy/generated/LazyNativeFunctions.cpp",
    "torch/csrc/lazy/generated/RegisterAutogradLazy.cpp",
    "torch/csrc/lazy/generated/RegisterLazy.cpp",
] + _GENERATED_AUTOGRAD_CPP_HEADERS + GENERATED_LAZY_H

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
            "aten/src/ATen/native/ts_native_functions.yaml",
            "aten/src/ATen/templates/DispatchKeyNativeFunctions.cpp",
            "aten/src/ATen/templates/DispatchKeyNativeFunctions.h",
            "aten/src/ATen/templates/LazyIr.h",
            "aten/src/ATen/templates/RegisterDispatchKey.cpp",
            "torch/csrc/lazy/core/shape_inference.h",
            "torch/csrc/lazy/ts_backend/ts_native_functions.cpp",
            ":native_functions.yaml",
        ],
        tools = ["//tools/setup_helpers:generate_code"],
        outs = _GENERATED_CPP + GENERATED_AUTOGRAD_H + GENERATED_LAZY_H + GENERATED_TESTING_PY,
        cmd = "$(location //tools/setup_helpers:generate_code) " +
              "--install_dir $(RULEDIR) " +
              "--native-functions-path $(location :native_functions.yaml) " +
              "--gen_lazy_ts_backend",
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

# In the open-source build, these are generated into
# torch/csrc/autograd/generated
GENERATED_AUTOGRAD_H = [
    "autograd/generated/Functions.h",
    "autograd/generated/VariableType.h",
    "autograd/generated/python_functions.h",
    "autograd/generated/variable_factories.h",
]

# In the open-source build, these are generated into
# torch/testing/_internal/generated
GENERATED_TESTING_PY = [
    "annotated_fn_args.py",
]

GENERATED_LAZY_H = [
    "lazy/generated/LazyIr.h",
    "lazy/generated/LazyNativeFunctions.h",
]

# In both open-source and fbcode builds, these are generated into
# torch/csrc/{autograd,jit}/generated.i
_GENERATED_CPP = [
    "autograd/generated/Functions.cpp",
    "autograd/generated/VariableType_0.cpp",
    "autograd/generated/VariableType_1.cpp",
    "autograd/generated/VariableType_2.cpp",
    "autograd/generated/VariableType_3.cpp",
    "autograd/generated/VariableType_4.cpp",
    "autograd/generated/TraceType_0.cpp",
    "autograd/generated/TraceType_1.cpp",
    "autograd/generated/TraceType_2.cpp",
    "autograd/generated/TraceType_3.cpp",
    "autograd/generated/TraceType_4.cpp",
    "autograd/generated/ADInplaceOrViewType_0.cpp",
    "autograd/generated/ADInplaceOrViewType_1.cpp",
    "autograd/generated/python_functions_0.cpp",
    "autograd/generated/python_functions_1.cpp",
    "autograd/generated/python_functions_2.cpp",
    "autograd/generated/python_functions_3.cpp",
    "autograd/generated/python_functions_4.cpp",
    "autograd/generated/python_nn_functions.cpp",
    "autograd/generated/python_fft_functions.cpp",
    "autograd/generated/python_linalg_functions.cpp",
    "autograd/generated/python_return_types.cpp",
    "autograd/generated/python_sparse_functions.cpp",
    "autograd/generated/python_special_functions.cpp",
    "autograd/generated/python_torch_functions_0.cpp",
    "autograd/generated/python_torch_functions_1.cpp",
    "autograd/generated/python_torch_functions_2.cpp",
    "autograd/generated/python_variable_methods.cpp",
]

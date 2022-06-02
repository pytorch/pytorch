load(
    ":ufunc_defs.bzl",
    "aten_ufunc_generated_cpu_kernel_sources",
    "aten_ufunc_generated_cpu_sources",
    "aten_ufunc_generated_cuda_sources",
)

def define_targets(rules):
    rules.cc_library(
        name = "aten_core_headers",
        hdrs = aten_core_hdrs(rules = rules),
        strip_include_prefix = "aten/src/",
        deps = ["//c10"],
        visibility = ["//visibility:private"],
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

    #
    # ATen generated code
    # You need to keep this is sync with the files written out
    # by gen.py (in the cmake build system, we track generated files
    # via generated_cpp.txt and generated_cpp.txt-cuda
    #
    # Sure would be nice to use gen.py to create this list dynamically
    # instead of hardcoding, no? Well, we can't, as discussed in this
    # thread:
    # https://fb.facebook.com/groups/askbuck/permalink/1924258337622772/

    gen_aten_srcs = [
        "aten/src/ATen/native/native_functions.yaml",
        "aten/src/ATen/native/tags.yaml",
    ] + rules.glob(["aten/src/ATen/templates/*"])

    gen_aten_cmd = " ".join([
        "$(location //torchgen:gen)",
        "--install_dir=$(RULEDIR)/aten/src/ATen/",
        "--source-path aten/src/ATen",
    ] + (["--static_dispatch_backend CPU"] if rules.is_cpu_static_dispatch_build() else []))

    gen_aten_outs_cuda = (
        GENERATED_H_CUDA + GENERATED_CPP_CUDA +
        aten_ufunc_generated_cuda_sources("aten/src/ATen/{}")
    )

    gen_aten_outs = (
        GENERATED_H + GENERATED_H_CORE +
        GENERATED_CPP + GENERATED_CPP_CORE +
        aten_ufunc_generated_cpu_sources("aten/src/ATen/{}") +
        aten_ufunc_generated_cpu_kernel_sources("aten/src/ATen/{}") + [
            "aten/src/ATen/Declarations.yaml",
        ] + gen_aten_outs_cuda
    )

    rules.genrule(
        name = "gen_aten",
        srcs = gen_aten_srcs,
        tools = ["//torchgen:gen"],
        outs = gen_aten_outs,
        cmd = gen_aten_cmd,
    )

    rules.genrule(
        name = "gen_aten_hip",
        srcs = gen_aten_srcs,
        tools = ["//torchgen:gen"],
        outs = gen_aten_outs_cuda,
        cmd = gen_aten_cmd + " --rocm",
        features = ["-create_bazel_outputs"],
        tags = ["-bazel"],
    )

    rules.genrule(
        name = "generate-code",
        srcs = [
            ":DispatchKeyNativeFunctions.cpp",
            ":DispatchKeyNativeFunctions.h",
            ":LazyIr.h",
            ":LazyNonNativeIr.h",
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

def aten_core_hdrs(rules):
    """The source files that are part of the aten_core_headers target.

    Note that this is a function because globs are not permitted at
    global scope in Starlark.
    """
    return rules.glob([
        "aten/src/ATen/core/*.h",
        "aten/src/ATen/ops/*.h",
        "aten/src/ATen/core/boxing/*.h",
        "aten/src/ATen/core/boxing/impl/*.h",
        "aten/src/ATen/core/dispatch/*.h",
        "aten/src/ATen/core/op_registration/*.h",
    ]) + [
        "aten/src/ATen/CPUGeneratorImpl.h",
        "aten/src/ATen/NumericUtils.h",
        "aten/src/ATen/detail/CUDAHooksInterface.h",
        "aten/src/ATen/detail/ORTHooksInterface.h",
        "aten/src/ATen/detail/HIPHooksInterface.h",
    ]

#
# ATen generated code
# You need to keep this is sync with the files written out
# by gen.py (in the cmake build system, we track generated files
# via generated_cpp.txt and generated_cpp.txt-cuda
#
# Sure would be nice to use gen.py to create this list dynamically
# instead of hardcoding, no? Well, we can't, as discussed in this
# thread:
# https://fb.facebook.com/groups/askbuck/permalink/1924258337622772/

GENERATED_H = [
    "aten/src/ATen/Functions.h",
    "aten/src/ATen/NativeFunctions.h",
    "aten/src/ATen/NativeMetaFunctions.h",
    "aten/src/ATen/FunctionalInverses.h",
    "aten/src/ATen/RedispatchFunctions.h",
    "aten/src/ATen/RegistrationDeclarations.h",
]

GENERATED_H_CORE = [
    "aten/src/ATen/Operators.h",
    # CPUFunctions.h (and likely similar headers) need to be part of core because
    # of the static dispatch build: TensorBody.h directly includes CPUFunctions.h.
    # The disinction looks pretty arbitrary though; maybe will can kill core
    # and merge the two?
    "aten/src/ATen/CPUFunctions.h",
    "aten/src/ATen/CPUFunctions_inl.h",
    "aten/src/ATen/CompositeExplicitAutogradFunctions.h",
    "aten/src/ATen/CompositeExplicitAutogradFunctions_inl.h",
    "aten/src/ATen/CompositeImplicitAutogradFunctions.h",
    "aten/src/ATen/CompositeImplicitAutogradFunctions_inl.h",
    "aten/src/ATen/MetaFunctions.h",
    "aten/src/ATen/MetaFunctions_inl.h",
    "aten/src/ATen/core/TensorBody.h",
    "aten/src/ATen/MethodOperators.h",
    "aten/src/ATen/core/aten_interned_strings.h",
]

GENERATED_H_CUDA = [
    "aten/src/ATen/CUDAFunctions.h",
    "aten/src/ATen/CUDAFunctions_inl.h",
]

GENERATED_CPP_CUDA = [
    "aten/src/ATen/RegisterCUDA.cpp",
    "aten/src/ATen/RegisterNestedTensorCUDA.cpp",
    "aten/src/ATen/RegisterSparseCUDA.cpp",
    "aten/src/ATen/RegisterSparseCsrCUDA.cpp",
    "aten/src/ATen/RegisterQuantizedCUDA.cpp",
]

GENERATED_CPP = [
    "aten/src/ATen/Functions.cpp",
    "aten/src/ATen/RegisterBackendSelect.cpp",
    "aten/src/ATen/RegisterCPU.cpp",
    "aten/src/ATen/RegisterQuantizedCPU.cpp",
    "aten/src/ATen/RegisterNestedTensorCPU.cpp",
    "aten/src/ATen/RegisterSparseCPU.cpp",
    "aten/src/ATen/RegisterSparseCsrCPU.cpp",
    "aten/src/ATen/RegisterMkldnnCPU.cpp",
    "aten/src/ATen/RegisterCompositeImplicitAutograd.cpp",
    "aten/src/ATen/RegisterZeroTensor.cpp",
    "aten/src/ATen/RegisterMeta.cpp",
    "aten/src/ATen/RegisterCompositeExplicitAutograd.cpp",
    "aten/src/ATen/CompositeViewCopyKernels.cpp",
    "aten/src/ATen/RegisterSchema.cpp",
    "aten/src/ATen/RegisterFunctionalization_0.cpp",
    "aten/src/ATen/RegisterFunctionalization_1.cpp",
    "aten/src/ATen/RegisterFunctionalization_2.cpp",
    "aten/src/ATen/RegisterFunctionalization_3.cpp",
]

GENERATED_CPP_CORE = [
    "aten/src/ATen/Operators_0.cpp",
    "aten/src/ATen/Operators_1.cpp",
    "aten/src/ATen/Operators_2.cpp",
    "aten/src/ATen/Operators_3.cpp",
    "aten/src/ATen/Operators_4.cpp",
    "aten/src/ATen/core/ATenOpList.cpp",
    "aten/src/ATen/core/TensorMethods.cpp",
]

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
    "torch/csrc/lazy/generated/LazyNonNativeIr.h",
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

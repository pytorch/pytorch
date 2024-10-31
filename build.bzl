load(
    ":ufunc_defs.bzl",
    "aten_ufunc_generated_cpu_kernel_sources",
    "aten_ufunc_generated_cpu_sources",
    "aten_ufunc_generated_cuda_sources",
)

def define_targets(rules):
    rules.cc_library(
        name = "caffe2_core_macros",
        hdrs = [":caffe2_core_macros_h"],
    )

    rules.cmake_configure_file(
        name = "caffe2_core_macros_h",
        src = "caffe2/core/macros.h.in",
        out = "caffe2/core/macros.h",
        definitions = [
            "CAFFE2_BUILD_SHARED_LIBS",
            "CAFFE2_PERF_WITH_AVX",
            "CAFFE2_PERF_WITH_AVX2",
            "CAFFE2_USE_EXCEPTION_PTR",
            "CAFFE2_USE_CUDNN",
            "USE_MKLDNN",
            "CAFFE2_USE_ITT",
            "USE_ROCM_KERNEL_ASSERT",
            "EIGEN_MPL2_ONLY",
        ],
    )

    rules.cc_library(
        name = "caffe2_serialize",
        srcs = [
            "caffe2/serialize/file_adapter.cc",
            "caffe2/serialize/inline_container.cc",
            "caffe2/serialize/istream_adapter.cc",
            "caffe2/serialize/read_adapter_interface.cc",
        ],
        copts = ["-fexceptions"],
        tags = [
            "-fbcode",
            "supermodule:android/default/pytorch",
            "supermodule:ios/default/public.pytorch",
            "xplat",
        ],
        visibility = ["//visibility:public"],
        deps = [
            ":caffe2_headers",
            "//c10",
            "//third_party/miniz-2.1.0:miniz",
            "@com_github_glog//:glog",
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
        "$(execpath //torchgen:gen)",
        "--install_dir=$(RULEDIR)",
        "--source-path aten/src/ATen",
        "--aoti_install_dir=$(RULEDIR)/torch/csrc/inductor/aoti_torch/generated"
    ] + (["--static_dispatch_backend CPU"] if rules.is_cpu_static_dispatch_build() else []))

    gen_aten_outs_cuda = (
        GENERATED_H_CUDA + GENERATED_CPP_CUDA + GENERATED_AOTI_CUDA_CPP +
        aten_ufunc_generated_cuda_sources()
    )

    gen_aten_outs = (
        GENERATED_H + GENERATED_H_CORE +
        GENERATED_CPP + GENERATED_CPP_CORE +
        GENERATED_AOTI_CPP +
        aten_ufunc_generated_cpu_sources() +
        aten_ufunc_generated_cpu_kernel_sources() + [
            "Declarations.yaml",
        ] + gen_aten_outs_cuda
    )

    rules.genrule(
        name = "gen_aten",
        srcs = gen_aten_srcs,
        outs = gen_aten_outs,
        cmd = gen_aten_cmd,
        tools = ["//torchgen:gen"],
    )

    rules.genrule(
        name = "gen_aten_hip",
        srcs = gen_aten_srcs,
        outs = gen_aten_outs_cuda,
        cmd = gen_aten_cmd + " --rocm",
        features = ["-create_bazel_outputs"],
        tags = ["-bazel"],
        tools = ["//torchgen:gen"],
    )

    rules.genrule(
        name = "generate-code",
        srcs = [
            ":DispatchKeyNativeFunctions.cpp",
            ":DispatchKeyNativeFunctions.h",
            ":LazyIr.h",
            ":LazyNonNativeIr.h",
            ":RegisterDispatchDefinitions.ini",
            ":RegisterDispatchKey.cpp",
            ":native_functions.yaml",
            ":shape_inference.h",
            ":tags.yaml",
            ":ts_native_functions.cpp",
            ":ts_native_functions.yaml",
        ],
        outs = GENERATED_AUTOGRAD_CPP + GENERATED_AUTOGRAD_PYTHON + GENERATED_TESTING_PY,
        cmd = "$(execpath //tools/setup_helpers:generate_code) " +
              "--gen-dir=$(RULEDIR) " +
              "--native-functions-path $(location :native_functions.yaml) " +
              "--tags-path=$(location :tags.yaml) " +
              "--gen_lazy_ts_backend",
        tools = ["//tools/setup_helpers:generate_code"],
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
        cmd = "$(execpath //tools/setup_helpers:gen_version_header) " +
              "--template-path $(location :torch/csrc/api/include/torch/version.h.in) " +
              "--version-path $(location :version.txt) --output-path $@ ",
        tools = ["//tools/setup_helpers:gen_version_header"],
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

GENERATED_H = [
    "Functions.h",
    "NativeFunctions.h",
    "NativeMetaFunctions.h",
    "FunctionalInverses.h",
    "RedispatchFunctions.h",
    "RegistrationDeclarations.h",
    "VmapGeneratedPlumbing.h",
]

GENERATED_H_CORE = [
    "Operators.h",
    # CPUFunctions.h (and likely similar headers) need to be part of core because
    # of the static dispatch build: TensorBody.h directly includes CPUFunctions.h.
    # The disinction looks pretty arbitrary though; maybe will can kill core
    # and merge the two?
    "CPUFunctions.h",
    "CPUFunctions_inl.h",
    "CompositeExplicitAutogradFunctions.h",
    "CompositeExplicitAutogradFunctions_inl.h",
    "CompositeExplicitAutogradNonFunctionalFunctions.h",
    "CompositeExplicitAutogradNonFunctionalFunctions_inl.h",
    "CompositeImplicitAutogradFunctions.h",
    "CompositeImplicitAutogradFunctions_inl.h",
    "CompositeImplicitAutogradNestedTensorFunctions.h",
    "CompositeImplicitAutogradNestedTensorFunctions_inl.h",
    "MetaFunctions.h",
    "MetaFunctions_inl.h",
    "core/TensorBody.h",
    "MethodOperators.h",
    "core/aten_interned_strings.h",
    "core/enum_tag.h",
]

GENERATED_H_CUDA = [
    "CUDAFunctions.h",
    "CUDAFunctions_inl.h",
]

GENERATED_CPP_CUDA = [
    "RegisterCUDA.cpp",
    "RegisterNestedTensorCUDA.cpp",
    "RegisterSparseCUDA.cpp",
    "RegisterSparseCsrCUDA.cpp",
    "RegisterQuantizedCUDA.cpp",
]

GENERATED_CPP = [
    "Functions.cpp",
    "RegisterBackendSelect.cpp",
    "RegisterCPU.cpp",
    "RegisterQuantizedCPU.cpp",
    "RegisterNestedTensorCPU.cpp",
    "RegisterSparseCPU.cpp",
    "RegisterSparseCsrCPU.cpp",
    "RegisterMkldnnCPU.cpp",
    "RegisterCompositeImplicitAutograd.cpp",
    "RegisterCompositeImplicitAutogradNestedTensor.cpp",
    "RegisterZeroTensor.cpp",
    "RegisterMeta.cpp",
    "RegisterQuantizedMeta.cpp",
    "RegisterNestedTensorMeta.cpp",
    "RegisterSparseMeta.cpp",
    "RegisterCompositeExplicitAutograd.cpp",
    "RegisterCompositeExplicitAutogradNonFunctional.cpp",
    "CompositeViewCopyKernels.cpp",
    "RegisterSchema.cpp",
    "RegisterFunctionalization_0.cpp",
    "RegisterFunctionalization_1.cpp",
    "RegisterFunctionalization_2.cpp",
    "RegisterFunctionalization_3.cpp",
]

GENERATED_CPP_CORE = [
    "Operators_0.cpp",
    "Operators_1.cpp",
    "Operators_2.cpp",
    "Operators_3.cpp",
    "Operators_4.cpp",
    "core/ATenOpList.cpp",
    "core/TensorMethods.cpp",
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
    "torch/csrc/autograd/generated/python_return_types.h",
]

_GENERATED_AUTOGRAD_CPP_HEADERS = [
    "torch/csrc/autograd/generated/Functions.h",
    "torch/csrc/autograd/generated/VariableType.h",
    "torch/csrc/autograd/generated/ViewFuncs.h",
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
    "torch/csrc/autograd/generated/python_nested_functions.cpp",
    "torch/csrc/autograd/generated/python_fft_functions.cpp",
    "torch/csrc/autograd/generated/python_linalg_functions.cpp",
    "torch/csrc/autograd/generated/python_return_types.cpp",
    "torch/csrc/autograd/generated/python_enum_tag.cpp",
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
    "torch/csrc/autograd/generated/ViewFuncs.cpp",
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

GENERATED_AOTI_CPP = [
    "torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.cpp",
]

GENERATED_AOTI_CUDA_CPP = [
    "torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.cpp",
]

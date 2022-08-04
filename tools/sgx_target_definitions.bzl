load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")
load("@fbsource//tools/build_defs:buckconfig.bzl", "read_bool")
load(
    "//caffe2:build_variables.bzl",
    "core_sources_common",
    "core_sources_full_mobile",
    "core_trainer_sources",
    "libtorch_extra_sources",
    "libtorch_generated_sources",
)

is_sgx = read_bool("fbcode", "sgx_mode", False)

def libtorch_sgx_sources(gencode_pattern = ":generate-code[{}]"):
    libtorch_core_mobile_sources = sorted(core_sources_common + core_sources_full_mobile + core_trainer_sources)

    sgx_sources_to_exclude = [
        "torch/csrc/jit/tensorexpr/llvm_codegen.cpp",
        "torch/csrc/jit/tensorexpr/llvm_jit.cpp",
        "torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp",
    ]

    return libtorch_generated_sources(gencode_pattern) + [i for i in libtorch_core_mobile_sources if i not in sgx_sources_to_exclude] + [i for i in libtorch_extra_sources if i not in sgx_sources_to_exclude]

def add_sgx_torch_libs():
    # we do not need to define these targets if we are in not SGX mode
    if not is_sgx:
        return

    compiler_flags_cpu = [
        "-DNO_CUDNN_DESTROY_HANDLE",
        "-DPYTORCH_ONNX_CAFFE2_BUNDLE",
        "-DTORCH_ENABLE_LLVM",
        "-Wno-write-strings",
        "-Wno-format",
        "-Wno-strict-aliasing",
        "-Wno-non-virtual-dtor",
        "-Wno-shadow-compatible-local",
        "-Wno-empty-body",
        "-DUSE_XNNPACK",
    ]

    propagated_pp_flags_cpu = [
        "-DSYMBOLICATE_MOBILE_DEBUG_HANDLE",
        "-DC10_MOBILE",
    ]

    include_directories = [
        "..",
        ".",
        "torch/csrc/api/include",
        "torch/csrc",
        "torch/csrc/nn",
        "torch/lib",
    ]

    common_flags = {
        "compiler_specific_flags": {
            "clang": [
                "-Wno-absolute-value",
                "-Wno-expansion-to-defined",
                "-Wno-pessimizing-move",
                "-Wno-return-type-c-linkage",
                "-Wno-unknown-pragmas",
            ],
        },
        "headers": native.glob(["torch/csrc/**/*.h", "torch/csrc/generic/*.cpp", "test/cpp/jit/*.h", "test/cpp/tensorexpr/*.h"]),
    }

    _libtorch_sgx_sources = list(libtorch_sgx_sources())

    cpp_library(
        name = "libtorch-sgx",
        srcs = _libtorch_sgx_sources + [
            "fb/supported_mobile_models/SupportedMobileModels.cpp",
            "torch/csrc/jit/mobile/function.cpp",
            "torch/csrc/jit/mobile/import.cpp",
            "torch/csrc/jit/mobile/interpreter.cpp",
            "torch/csrc/jit/mobile/module.cpp",  # this is only needed to load the model from caffe2/test/cpp/lite_interpreter_runtime/delegate_test.ptl
        ],
        link_whole = True,
        include_directories = include_directories,
        propagated_pp_flags = propagated_pp_flags_cpu,
        exported_deps = [
            ":generated-autograd-headers",
            ":generated-version-header",
            "//caffe2/aten:ATen-sgx-cpu",
            "//caffe2/caffe2:caffe2_sgx_core",
            "//onnx/onnx:onnx_lib",
        ],
        exported_external_deps = [
            ("protobuf", None),
        ],
        compiler_flags = compiler_flags_cpu,
        **common_flags
    )

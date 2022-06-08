load("//tools/build_defs:expect.bzl", "expect")

# NOTE: This file is shared by internal and OSS BUCK build.
# These load paths point to different files in internal and OSS environment
load("//tools/build_defs:fb_native_wrapper.bzl", "fb_native")
load("//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("//tools/build_defs:glob_defs.bzl", "subdir_glob")
load(
    ":build_variables.bzl",
    "jit_core_headers",
)

def read_bool(section, field, default):
    # @lint-ignore BUCKRESTRICTEDSYNTAX
    value = read_config(section, field)
    if value == None:
        return default
    expect(
        value == "0" or value == "1",
        "{}.{} == \"{}\", wanted \"0\" or \"1\".".format(section, field, value),
    )
    return bool(int(value))

def is_oss_build():
    return read_bool("pt", "is_oss", False)

# for targets in caffe2 root path
ROOT_NAME = "//" if is_oss_build() else "//xplat/caffe2"

# for targets in subfolders
ROOT_PATH = "//" if is_oss_build() else "//xplat/caffe2/"

# these targets are shared by internal and OSS BUCK
def define_buck_targets(
        feature = None,
        labels = []):
    fb_xplat_cxx_library(
        name = "th_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            # TH
            ("aten/src", "TH/*.h"),
            ("aten/src", "TH/*.hpp"),
            ("aten/src", "TH/generic/*.h"),
            ("aten/src", "TH/generic/*.hpp"),
            ("aten/src", "TH/generic/simd/*.h"),
            ("aten/src", "TH/vector/*.h"),
            ("aten/src", "TH/generic/*.c"),
            ("aten/src", "TH/generic/*.cpp"),
            ("aten/src/TH", "*.h"),  # for #include <THGenerateFloatTypes.h>
            # THNN
            ("aten/src", "THNN/*.h"),
            ("aten/src", "THNN/generic/*.h"),
            ("aten/src", "THNN/generic/*.c"),
        ]),
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "aten_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            # ATen Core
            ("aten/src", "ATen/core/**/*.h"),
            ("aten/src", "ATen/ops/*.h"),
            # ATen Base
            ("aten/src", "ATen/*.h"),
            ("aten/src", "ATen/cpu/**/*.h"),
            ("aten/src", "ATen/detail/*.h"),
            ("aten/src", "ATen/quantized/*.h"),
            ("aten/src", "ATen/vulkan/*.h"),
            ("aten/src", "ATen/metal/*.h"),
            ("aten/src", "ATen/nnapi/*.h"),
            # ATen Native
            ("aten/src", "ATen/native/*.h"),
            ("aten/src", "ATen/native/ao_sparse/quantized/cpu/*.h"),
            ("aten/src", "ATen/native/cpu/**/*.h"),
            ("aten/src", "ATen/native/sparse/*.h"),
            ("aten/src", "ATen/native/nested/*.h"),
            ("aten/src", "ATen/native/quantized/*.h"),
            ("aten/src", "ATen/native/quantized/cpu/*.h"),
            ("aten/src", "ATen/native/transformers/*.h"),
            ("aten/src", "ATen/native/ufunc/*.h"),
            ("aten/src", "ATen/native/utils/*.h"),
            ("aten/src", "ATen/native/vulkan/ops/*.h"),
            ("aten/src", "ATen/native/xnnpack/*.h"),
            ("aten/src", "ATen/mps/*.h"),
            ("aten/src", "ATen/native/mps/*.h"),
            # Remove the following after modifying codegen for mobile.
            ("aten/src", "ATen/mkl/*.h"),
            ("aten/src", "ATen/native/mkl/*.h"),
            ("aten/src", "ATen/native/mkldnn/*.h"),
        ]),
        visibility = ["PUBLIC"],
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "aten_vulkan_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            ("aten/src", "ATen/native/vulkan/*.h"),
            ("aten/src", "ATen/native/vulkan/api/*.h"),
            ("aten/src", "ATen/native/vulkan/ops/*.h"),
            ("aten/src", "ATen/vulkan/*.h"),
        ]),
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "jit_core_headers",
        header_namespace = "",
        exported_headers = subdir_glob([("", x) for x in jit_core_headers]),
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "torch_headers",
        header_namespace = "",
        exported_headers = subdir_glob(
            [
                ("torch/csrc/api/include", "torch/**/*.h"),
                ("", "torch/csrc/**/*.h"),
                ("", "torch/csrc/generic/*.cpp"),
                ("", "torch/script.h"),
                ("", "torch/library.h"),
                ("", "torch/custom_class.h"),
                ("", "torch/custom_class_detail.h"),
                # Add again due to namespace difference from aten_header.
                ("", "aten/src/ATen/*.h"),
                ("", "aten/src/ATen/quantized/*.h"),
            ],
            exclude = [
                # Don't need on mobile.
                "torch/csrc/Exceptions.h",
                "torch/csrc/python_headers.h",
                "torch/csrc/utils/auto_gil.h",
                "torch/csrc/jit/serialization/mobile_bytecode_generated.h",
            ],
        ),
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
        deps = [
            ":generated-version-header",
        ],
    )

    fb_xplat_cxx_library(
        name = "aten_test_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            ("aten/src", "ATen/test/*.h"),
        ]),
    )

    fb_xplat_cxx_library(
        name = "torch_mobile_headers",
        header_namespace = "",
        exported_headers = subdir_glob(
            [
                ("", "torch/csrc/jit/mobile/*.h"),
            ],
        ),
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "generated_aten_config_header",
        header_namespace = "ATen",
        exported_headers = {
            "Config.h": ":generate_aten_config[Config.h]",
        },
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "generated-autograd-headers",
        header_namespace = "torch/csrc/autograd/generated",
        exported_headers = {
            "Functions.h": ":gen_aten_libtorch[autograd/generated/Functions.h]",
            "VariableType.h": ":gen_aten_libtorch[autograd/generated/VariableType.h]",
            "variable_factories.h": ":gen_aten_libtorch[autograd/generated/variable_factories.h]",
            # Don't build python bindings on mobile.
            #"python_functions.h",
        },
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "generated-version-header",
        header_namespace = "torch",
        exported_headers = {
            "version.h": ":generate-version-header[version.h]",
        },
        feature = feature,
        labels = labels,
    )

    # @lint-ignore BUCKLINT
    fb_native.genrule(
        name = "generate-version-header",
        srcs = [
            "torch/csrc/api/include/torch/version.h.in",
            "version.txt",
        ],
        cmd = "$(exe {}tools/setup_helpers:gen-version-header) ".format(ROOT_PATH) + " ".join([
            "--template-path",
            "torch/csrc/api/include/torch/version.h.in",
            "--version-path",
            "version.txt",
            "--output-path",
            "$OUT/version.h",
        ]),
        outs = {
            "version.h": ["version.h"],
        },
        default_outs = ["."],
    )

    # @lint-ignore BUCKLINT
    fb_native.filegroup(
        name = "aten_src_path",
        srcs = [
            "aten/src/ATen/native/native_functions.yaml",
            "aten/src/ATen/native/tags.yaml",
            # @lint-ignore BUCKRESTRICTEDSYNTAX
        ] + glob(["aten/src/ATen/templates/*"]),
        visibility = [
            "PUBLIC",
        ],
    )

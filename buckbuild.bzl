# NOTE: This file is shared by internal and OSS BUCK build.
# These load paths point to different files in internal and OSS environment

load("//tools/build_defs:fb_native_wrapper.bzl", "fb_native")
load("//tools/build_defs:fb_python_binary.bzl", "fb_python_binary")
load("//tools/build_defs:fb_python_library.bzl", "fb_python_library")
load("//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("//tools/build_defs:fb_xplat_genrule.bzl", "fb_xplat_genrule")
load("//tools/build_defs:fbsource_utils.bzl", "is_arvr_mode")
load("//tools/build_defs:glob_defs.bzl", "subdir_glob")
load("//tools/build_defs:platform_defs.bzl", "APPLETVOS", "IOS", "MACOSX")
load("//tools/build_defs/windows:windows_flag_map.bzl", "windows_convert_gcc_clang_flags")
load(
    ":build_variables.bzl",
    "core_sources_common",
    "core_sources_full_mobile_no_backend_interface",
    "core_trainer_sources",
    "jit_core_headers",
    "libtorch_profiler_sources",
)

def read_bool(section, field, default, required = True):
    # @lint-ignore BUCKRESTRICTEDSYNTAX
    val = read_config(section, field)
    if val != None:
        if val in ["true", "True", "1"]:
            return True
        elif val in ["false", "False", "0"]:
            return False
        else:
            fail(
                "`{}:{}`: must be one of (0, 1, true, false, True, False), but was {}".format(section, field, val),
            )
    elif default != None:
        return default
    elif not required:
        return None
    else:
        fail("`{}:{}`: no value set".format(section, field))

def get_enable_eager_symbolication():
    return read_bool("pt", "enable_eager_symbolication", default = False, required = False)

# @lint-ignore BUCKRESTRICTEDSYNTAX
IS_OSS = read_config("pt", "is_oss", "0") == "1"  # True for OSS BUCK build, and False for internal BUCK build

NOT_OSS = not IS_OSS

# for targets in caffe2 root path
ROOT = "//" if IS_OSS else "//xplat/caffe2"

# for targets in subfolders
ROOT_PATH = "//" if IS_OSS else "//xplat/caffe2/"

C10 = "//c10:c10" if IS_OSS else "//xplat/caffe2/c10:c10"

# a dictionary maps third party library name to fbsource and oss target
THIRD_PARTY_LIBS = {
    "FP16": ["//third-party/FP16:FP16", "//third_party:FP16"],
    "FXdiv": ["//third-party/FXdiv:FXdiv", "//third_party:FXdiv"],
    "XNNPACK": ["//xplat/third-party/XNNPACK:XNNPACK", "//third_party:XNNPACK"],
    "clog": ["//third-party/clog:clog", "//third_party:clog"],
    "cpuinfo": ["//third-party/cpuinfo:cpuinfo", "//third_party:cpuinfo"],
    "flatbuffers-api": ["//third-party/flatbuffers:flatbuffers-api", "//third_party:flatbuffers-api"],
    "flatc": ["//third-party/flatbuffers:flatc", "//third_party:flatc"],
    "fmt": ["//third-party/fmt:fmt", "//third_party:fmt"],
    "glog": ["//third-party/glog:glog", "//third_party:glog"],
    "kineto": ["//xplat/kineto/libkineto:libkineto", "//third_party:libkineto"],
    "psimd": ["//third-party/psimd:psimd", "//third_party:psimd"],
    "pthreadpool": ["//third-party/pthreadpool:pthreadpool", "//third_party:pthreadpool"],
    "pthreadpool_header": ["//third-party/pthreadpool:pthreadpool_header", "//third_party:pthreadpool_header"],
    "pyyaml": ["//third-party/pyyaml:pyyaml", "//third_party:pyyaml"],
    "ruy": ["//third-party/ruy:ruy_xplat_lib", "//third_party:ruy_lib"],
    "typing-extensions": ["//third-party/typing-extensions:typing-extensions", "//third_party:typing-extensions"],
}

def third_party(name):
    if name not in THIRD_PARTY_LIBS:
        fail("Cannot find thrid party library " + name + ", please register it in THIRD_PARTY_LIBS first!")
    return THIRD_PARTY_LIBS[name][1] if IS_OSS else THIRD_PARTY_LIBS[name][0]

def get_pt_compiler_flags():
    return select({
        "DEFAULT": _PT_COMPILER_FLAGS + [
            "-std=gnu++17",  #to accomodate for eigen
        ],
        "ovr_config//compiler:cl": windows_convert_gcc_clang_flags(_PT_COMPILER_FLAGS),
    })

_PT_COMPILER_FLAGS = [
    "-frtti",
    "-Os",
    "-Wno-unknown-pragmas",
    "-Wno-write-strings",
    "-Wno-unused-variable",
    "-Wno-unused-function",
    "-Wno-deprecated-declarations",
    "-Wno-shadow",
    "-Wno-global-constructors",
    "-Wno-missing-prototypes",
]

# these targets are shared by internal and OSS BUCK
def define_buck_targets(
        pt_xplat_cxx_library,
        c2_fbandroid_xplat_compiler_flags = [],
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
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "jit_core_headers",
        header_namespace = "",
        exported_headers = subdir_glob([("", x) for x in jit_core_headers]),
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
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "generated_aten_config_header",
        header_namespace = "ATen",
        exported_headers = {
            "Config.h": ":generate_aten_config[Config.h]",
        },
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
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "generated-version-header",
        header_namespace = "torch",
        exported_headers = {
            "version.h": ":generate-version-header[version.h]",
        },
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

    fb_xplat_cxx_library(
        name = "common_core",
        srcs = [
            "caffe2/core/common.cc",
        ],
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = get_pt_compiler_flags(),
        labels = labels,
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        visibility = ["PUBLIC"],
        windows_preferred_linkage = "static" if is_arvr_mode() else None,
        deps = [
            ":caffe2_headers",
            C10,
        ],
    )

    fb_python_library(
        name = "substitutelib",
        srcs = ["tools/substitute.py"],
        base_module = "",
    )

    fb_python_binary(
        name = "substitute",
        main_module = "tools.substitute",
        visibility = ["PUBLIC"],
        deps = [
            ":substitutelib",
        ],
    )

    # @lint-ignore BUCKLINT
    fb_native.genrule(
        name = "generate_aten_config",
        srcs = [
            "aten/src/ATen/Config.h.in",
        ],
        cmd = "$(exe :substitute) " + " ".join([
            "--install_dir",
            "$OUT",
            "--input-file",
            "aten/src/ATen/Config.h.in",
            "--output-file",
            "Config.h",
            "--replace",
            "@AT_MKLDNN_ENABLED@",
            "ATEN_MKLDNN_ENABLED_FBXPLAT",
            "--replace",
            "@AT_MKL_ENABLED@",
            "ATEN_MKL_ENABLED_FBXPLAT",
            "--replace",
            "@AT_MKL_SEQUENTIAL@",
            "ATEN_MKL_SEQUENTIAL_FBXPLAT",
            "--replace",
            "@AT_FFTW_ENABLED@",
            "0",
            "--replace",
            "@AT_POCKETFFT_ENABLED@",
            "0",
            "--replace",
            "@AT_NNPACK_ENABLED@",
            "ATEN_NNPACK_ENABLED_FBXPLAT",
            "--replace",
            "@CAFFE2_STATIC_LINK_CUDA_INT@",
            "CAFFE2_STATIC_LINK_CUDA_FBXPLAT",
            "--replace",
            "@AT_BUILD_WITH_BLAS@",
            "USE_BLAS_FBXPLAT",
            "--replace",
            "@AT_PARALLEL_OPENMP@",
            "AT_PARALLEL_OPENMP_FBXPLAT",
            "--replace",
            "@AT_PARALLEL_NATIVE@",
            "AT_PARALLEL_NATIVE_FBXPLAT",
            "--replace",
            "@AT_PARALLEL_NATIVE_TBB@",
            "AT_PARALLEL_NATIVE_TBB_FBXPLAT",
            "--replace",
            "@AT_BUILD_WITH_LAPACK@",
            "USE_LAPACK_FBXPLAT",
            "--replace",
            "@AT_BLAS_F2C@",
            "AT_BLAS_F2C_FBXPLAT",
            "--replace",
            "@AT_BLAS_USE_CBLAS_DOT@",
            "AT_BLAS_USE_CBLAS_DOT_FBXPLAT",
        ]),
        outs = {
            "Config.h": ["Config.h"],
        },
        default_outs = ["."],
    )

    fb_python_binary(
        name = "gen_aten_bin",
        main_module = "torchgen.gen",
        visibility = [
            "PUBLIC",
        ],
        deps = [
            ROOT_PATH + "torchgen:torchgen",
        ],
    )

    fb_python_binary(
        name = "gen_unboxing_bin",
        main_module = "tools.jit.gen_unboxing",
        visibility = [
            "PUBLIC",
        ],
        deps = [
            ROOT_PATH + "tools/jit:jit",
        ],
    )

    fb_python_library(
        name = "gen_oplist_lib",
        srcs = subdir_glob([
            ("tools/code_analyzer", "gen_oplist.py"),
            ("tools/code_analyzer", "gen_op_registration_allowlist.py"),
        ]),
        base_module = "",
        tests = [
            ":gen_oplist_test",
        ],
        deps = [
            third_party("pyyaml"),
            ROOT_PATH + "tools/lite_interpreter:gen_selected_mobile_ops_header",
            ROOT_PATH + "torchgen:torchgen",
        ],
    )

    fb_python_library(
        name = "gen_operators_yaml_lib",
        srcs = subdir_glob([
            ("tools/code_analyzer", "gen_operators_yaml.py"),
            ("tools/code_analyzer", "gen_op_registration_allowlist.py"),
        ]),
        base_module = "",
        tests = [
            ":gen_operators_yaml_test",
        ],
        deps = [
            third_party("pyyaml"),
            ROOT_PATH + "torchgen:torchgen",
        ],
    )

    fb_python_binary(
        name = "gen_aten_vulkan_spv_bin",
        main_module = "aten.src.ATen.gen_vulkan_spv",
        visibility = [
            "PUBLIC",
        ],
        deps = [
            ":gen_aten_vulkan_spv_lib",
        ],
    )

    fb_python_library(
        name = "gen_aten_vulkan_spv_lib",
        srcs = [
            "aten/src/ATen/gen_vulkan_spv.py",
        ],
        base_module = "",
        deps = [
            ROOT_PATH + "torchgen:torchgen",
        ],
    )

    fb_python_binary(
        name = "gen_oplist",
        main_module = "gen_oplist",
        visibility = ["PUBLIC"],
        deps = [
            ":gen_oplist_lib",
        ],
    )

    fb_python_binary(
        name = "gen_operators_yaml",
        main_module = "gen_operators_yaml",
        visibility = ["PUBLIC"],
        deps = [
            ":gen_operators_yaml_lib",
        ],
    )

    fb_xplat_cxx_library(
        name = "torch_mobile_observer",
        srcs = [
            "torch/csrc/jit/mobile/observer.cpp",
        ] + ([] if IS_OSS else ["torch/fb/observers/MobileObserverUtil.cpp"]),
        header_namespace = "",
        exported_headers = subdir_glob(
            [
                ("", "torch/csrc/jit/mobile/observer.h"),
            ] + ([] if IS_OSS else [
                ("", "torch/fb/observers/ObserverUtil.h"),
                ("", "torch/fb/observers/MobileObserverUtil.h"),
            ]),
        ),
        fbobjc_compiler_flags = [
            "-Wno-missing-prototypes",
        ],
        labels = labels,
        visibility = ["PUBLIC"],
        deps = [
            C10,
        ],
    )

    # Base library shared by lite-interpreter and full-jit.
    pt_xplat_cxx_library(
        name = "torch_common",
        srcs = core_sources_common,
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        visibility = ["PUBLIC"],
        deps = [
            ":aten_cpu",
            ":generated-autograd-headers",
            ":torch_headers",
            C10,
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_mobile_deserialize_common",
        srcs = [
            "torch/csrc/jit/mobile/parse_bytecode.cpp",
            "torch/csrc/jit/mobile/parse_operators.cpp",
            "torch/csrc/jit/mobile/upgrader_mobile.cpp",
            "torch/csrc/jit/serialization/import_read.cpp",
            "torch/csrc/jit/serialization/unpickler.cpp",
        ],
        header_namespace = "",
        exported_headers = [
            "torch/csrc/jit/serialization/import_read.h",
            "torch/csrc/jit/serialization/unpickler.h",
        ],
        extra_flags = {
            "fbandroid_compiler_flags": ["-frtti"],
        },
        # torch_mobile_deserialize brings in sources neccessary to read a module
        # which depends on mobile module definition
        # link_whole is enable so that all symbols neccessary for mobile module are compiled
        # instead of only symbols used while loading; this prevents symbol
        # found definied in runtime
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = ["-Wl,--no-as-needed"],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":aten_cpu",
            ":caffe2_headers",
            ":caffe2_serialize",
            ":torch_common",
            ":torch_headers",
            ":torch_mobile_headers",
            ":torch_mobile_module",
            ":torch_mobile_observer",
            C10,
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_mobile_module",
        srcs = [
            "torch/csrc/jit/mobile/function.cpp",
            "torch/csrc/jit/mobile/interpreter.cpp",
            "torch/csrc/jit/mobile/module.cpp",
        ],
        header_namespace = "",
        exported_headers = [
        ],
        extra_flags = {
            "fbandroid_compiler_flags": ["-frtti"],
        },
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":aten_cpu",
            ":caffe2_headers",
            ":torch_common",
            ":torch_headers",
            ":torch_mobile_headers",
            ":torch_mobile_observer",
            C10,
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_mobile_debug_symbolication",
        srcs = [
            # included in aten_cpu "torch/csrc/jit/frontend/source_range.cpp",
            "torch/csrc/jit/ir/scope.cpp",
            "torch/csrc/jit/mobile/debug_info.cpp",
            "torch/csrc/jit/serialization/callstack_debug_info_serialization.cpp",
            "torch/csrc/jit/serialization/source_range_serialization.cpp",
            "torch/csrc/jit/serialization/pickle.cpp",
            # pickler.cpp doesn't seem to be needed.
            # "torch/csrc/jit/serialization/pickler.cpp",
            # included in core_sources_common "torch/csrc/jit/serialization/unpickler.cpp",
        ],
        header_namespace = "",
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":torch_mobile_deserialize",
        ],
        exported_deps = [
            ":torch_common",
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_model_tracer",
        srcs = [
            "torch/csrc/jit/mobile/model_tracer/BuildFeatureTracer.cpp",
            "torch/csrc/jit/mobile/model_tracer/CustomClassTracer.cpp",
            "torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.cpp",
            "torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.cpp",
            "torch/csrc/jit/mobile/model_tracer/TracerRunner.cpp",
        ],
        header_namespace = "",
        extra_flags = {
            "exported_preprocessor_flags": ["-DSYMBOLICATE_MOBILE_DEBUG_HANDLE"] if get_enable_eager_symbolication() else [],
        },
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":generated-autograd-headers",
            ":torch_mobile_deserialize",
            ":torch_mobile_headers",
            ":torch_mobile_observer",
        ] + ([] if IS_OSS else ["//xplat/folly:molly"]),
        exported_deps = [
            ":aten_cpu",
            ":torch_common",
        ] + ([] if IS_OSS else [
            "//xplat/caffe2/fb/custom_ops/batch_box_cox:batch_box_cox",
            "//xplat/caffe2/fb/custom_ops/maskrcnn:maskrcnn",
        ]),
    )

    pt_xplat_cxx_library(
        name = "torch_mobile_deserialize",
        srcs = [
            "torch/csrc/jit/mobile/import.cpp",
        ],
        header_namespace = "",
        exported_headers = [
            "torch/csrc/jit/mobile/import.h",
        ],
        # torch_mobile_deserialize brings in sources neccessary to read a module
        # which depends on mobile module definition
        # link_whole is enable so that all symbols neccessary for mobile module are compiled
        # instead of only symbols used while loading; this prevents symbol
        # found definied in runtime
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":aten_cpu",
            ":caffe2_headers",
            ":caffe2_serialize",
            ":torch_common",
            ":torch_headers",
            ":torch_mobile_headers",
            ":torch_mobile_module",
            ":torch_mobile_observer",
            ":torch_mobile_deserialize_common",
            C10,
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_mobile_core",
        srcs = [],
        header_namespace = "",
        exported_headers = [],
        extra_flags = {
            "exported_preprocessor_flags": (["-DSYMBOLICATE_MOBILE_DEBUG_HANDLE"] if get_enable_eager_symbolication() else []),
        },
        # torch_mobile_core brings in sources neccessary to read and run a module
        # link_whole is enabled so that all symbols linked
        # operators, registerations and other few symbols are need in runtime
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":generated-autograd-headers",
            ":torch_mobile_headers",
            ":torch_mobile_observer",
        ],
        exported_deps = [
            ":aten_cpu",
            ":torch_common",
            ":torch_mobile_deserialize",
            ":torch_supported_mobile_models",
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_mobile_core_pickle_and_flatbuffer",
        visibility = ["PUBLIC"],
        exported_deps = [
            ":torch_flatbuffer_all",
            ":torch_mobile_core",
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_core",
        srcs = core_sources_full_mobile_no_backend_interface + [
            "torch/csrc/api/src/jit.cpp",
            "torch/csrc/jit/serialization/export_bytecode.cpp",
            "torch/csrc/jit/serialization/export_module.cpp",
        ],
        visibility = [
            "//xplat/caffe2/android/...",
            "//xplat/caffe2/fb/...",
            "//xplat/caffe2/fb/model_tracer/...",
        ],
        deps = [
            ":aten_cpu",
            ":backend_interface_lib",
            ":generated-autograd-headers",
            ":torch_headers",
            ":torch_mobile_deserialize",
            third_party("glog"),
            C10,
        ] + ([] if IS_OSS else [
            "//xplat/caffe2/fb/custom_ops/batch_box_cox:batch_box_cox",
            "//xplat/caffe2/fb/custom_ops/maskrcnn:maskrcnn",
            "//xplat/third-party/linker_lib:rt",
        ]),
        exported_deps = [
            ":torch_common",
            ":torch_mobile_train",
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_train",
        srcs = [
            "torch/csrc/api/src/data/samplers/random.cpp",
            "torch/csrc/api/src/data/samplers/sequential.cpp",
            "torch/csrc/api/src/optim/optimizer.cpp",
            "torch/csrc/api/src/optim/serialize.cpp",
            "torch/csrc/api/src/optim/sgd.cpp",
            "torch/csrc/api/src/serialize/input-archive.cpp",
            "torch/csrc/api/src/serialize/output-archive.cpp",
            "torch/csrc/jit/api/module_save.cpp",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":aten_cpu",
            ":torch_headers",
            ":torch",
            ":torch_core",
            ":torch_mobile_deserialize",
            ":torch_mobile_train",
            C10,
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_mobile_train",
        srcs = core_trainer_sources + [
            "torch/csrc/autograd/VariableTypeManual.cpp",
            "torch/csrc/autograd/FunctionsManual.cpp",
            "torch/csrc/api/src/data/datasets/mnist.cpp",
            "torch/csrc/jit/mobile/train/export_data.cpp",
            "torch/csrc/jit/mobile/train/optim/sgd.cpp",
            "torch/csrc/jit/mobile/train/random.cpp",
            "torch/csrc/jit/mobile/train/sequential.cpp",
            ":gen_aten_libtorch[autograd/generated/Functions.cpp]",
        ],
        extra_flags = {
            "exported_preprocessor_flags": ["-DUSE_MOBILE_CLASSTYPE"],
        },
        # torch_mobile_train brings in sources neccessary to read and run a mobile
        # and save and load mobile params along with autograd
        # link_whole is enabled so that all symbols linked
        # operators, registerations and autograd related symbols are need in runtime
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        visibility = ["PUBLIC"],
        deps = [
            ":aten_cpu",
            ":generated-autograd-headers",
            ":torch_headers",
            ":torch_mobile_deserialize",
            C10,
        ],
    )

    pt_xplat_cxx_library(
        name = "torch",
        srcs = [
            "torch/csrc/jit/runtime/register_c10_ops.cpp",
            "torch/csrc/jit/runtime/register_prim_ops_fulljit.cpp",
        ],
        # torch brings in all sources neccessary to read and run a mobile module/jit module
        # link_whole is enabled so that all symbols linked
        # operators, registerations and other few symbols are need in runtime
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        visibility = ["PUBLIC"],
        deps = [
            # This is to have autograd profiler available
            # in xplat/caffe2:torch which some builds are using
            # notable xplate/facegen:testsAndroid
            ":torch_headers",
            ":torch_kineto_profiling",
        ],
        exported_deps = [
            ":aten_cpu",
            ":torch_core",
            C10,
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_mobile_train_import_data",
        srcs = [
            "torch/csrc/jit/mobile/import_data.cpp",
        ],
        extra_flags = {
            "exported_preprocessor_flags": ["-DUSE_MOBILE_CLASSTYPE"],
        },
        # torch_mobile_train_import_data brings in sources neccessary to read a mobile module
        # link_whole is enabled so that all symbols linked
        # operators other few symbols are need in runtime
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        visibility = ["PUBLIC"],
        deps = [
            ":torch_headers",
            ":torch_mobile_observer",
            ":torch_mobile_core",
            ":torch_mobile_train",
        ],
    )

    fb_xplat_cxx_library(
        name = "torch_mobile_compatibility",
        srcs = [
            # These .cpp brought in through core_sources_common
            # "torch/csrc/jit/mobile/compatibility/runtime_compatibility.cpp",
            # "torch/csrc/jit/serialization/unpickler.cpp",
            "torch/csrc/jit/mobile/compatibility/model_compatibility.cpp",
            "torch/csrc/jit/serialization/pickle.cpp",
            "torch/csrc/jit/serialization/pickler.cpp",
        ],
        header_namespace = "",
        exported_headers = [
            "torch/csrc/jit/mobile/compatibility/backport.h",
            "torch/csrc/jit/mobile/compatibility/backport_manager.h",
            "torch/csrc/jit/mobile/compatibility/model_compatibility.h",
            "torch/csrc/jit/mobile/compatibility/runtime_compatibility.h",
        ],
        compiler_flags = [
            "-fexceptions",
            "-frtti",
            "-Wno-deprecated-declarations",
            "-Wno-global-constructors",
        ],
        labels = labels,
        visibility = ["PUBLIC"],
        deps = [
            ":torch_mobile_deserialize",
        ],
    )

    pt_xplat_cxx_library(
        name = "jit_module_saving",
        srcs = [
            "torch/csrc/jit/api/module_save.cpp",
            "torch/csrc/jit/serialization/export_bytecode.cpp",
            "torch/csrc/jit/serialization/export_module.cpp",
        ],
        exported_headers = [
            "torch/csrc/jit/serialization/export.h",
            "torch/csrc/jit/serialization/flatbuffer_serializer_jit.h",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":torch",
            ":torch_mobile_core",
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_mobile_model_tracer",
        srcs = [
            "torch/csrc/jit/mobile/model_tracer/MobileModelRunner.cpp",
            "torch/csrc/jit/mobile/model_tracer/TensorUtils.cpp",
        ],
        headers = [
            "torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h",
            "torch/csrc/jit/mobile/model_tracer/TensorUtils.h",
        ],
        header_namespace = "",
        exported_headers = [
            "torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h",
        ],
        extra_flags = {
            "exported_preprocessor_flags": (["-DSYMBOLICATE_MOBILE_DEBUG_HANDLE"] if get_enable_eager_symbolication() else []),
        },
        # torch_mobile_model_tracer brings in sources neccessary to read and run a jit module
        # and trace the ops
        # link_whole is enabled so that all symbols linked
        # operators, registerations and other few symbols are need in runtime
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":caffe2_serialize",
            ":generated-autograd-headers",
            ":torch_mobile_headers",
            ":torch_mobile_observer",
            ":torch_mobile_core",
        ] + ([] if IS_OSS else ["//xplat/folly:molly"]),
        exported_deps = [
            ":aten_cpu",
            ":torch_common",
        ] + ([] if IS_OSS else [
            "//xplat/caffe2/fb/custom_ops/batch_box_cox:batch_box_cox",
            "//xplat/caffe2/fb/custom_ops/maskrcnn:maskrcnn",
            "//xplat/caffe2/fb/custom_ops/sparsenn:sparsenn-all",
        ]),
    )

    pt_xplat_cxx_library(
        name = "torch_mobile_core_flatbuffer",
        srcs = [],
        header_namespace = "",
        exported_headers = [],
        extra_flags = {
            "exported_preprocessor_flags": ["-DSYMBOLICATE_MOBILE_DEBUG_HANDLE"] if get_enable_eager_symbolication() else [],
        },
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":generated-autograd-headers",
            ":torch_mobile_headers",
            ":torch_mobile_observer",
        ],
        exported_deps = [
            ":aten_cpu",
            ":torch_common",
        ] + ([] if IS_OSS else [
            "//xplat/caffe2/fb/runtime:torch_mobile_deserialize_flatbuffer",
        ]),
    )

    fb_xplat_cxx_library(
        name = "backend_interface_lib",
        srcs = [
            "torch/csrc/jit/backends/backend_debug_info.cpp",
            "torch/csrc/jit/backends/backend_interface.cpp",
        ],
        compiler_flags = get_pt_compiler_flags(),
        fbandroid_compiler_flags = c2_fbandroid_xplat_compiler_flags,
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":aten_cpu",
            ":torch_common",
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_kineto_profiling",
        srcs = libtorch_profiler_sources,
        extra_flags = {
            "compiler_flags": ["-Wno-error"],
            "exported_preprocessor_flags": [
                "-DUSE_KINETO",
                "-DUSE_KINETO_UPDATED",
                # Need this otherwise USE_KINETO is undefed
                # for mobile
                "-DEDGE_PROFILER_USE_KINETO",
            ],
        },
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        deps = [
            third_party("glog"),
            third_party("kineto"),
        ],
        exported_deps = [
            ":aten_cpu",
            ":torch_common",
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_edge_profiling",
        srcs = ["torch/csrc/jit/mobile/profiler_edge.cpp"],
        extra_flags = {
            "exported_preprocessor_flags": [
                "-DUSE_KINETO",
                "-DUSE_KINETO_UPDATED",
                "-DEDGE_PROFILER_USE_KINETO",
            ],
        },
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":torch_common",
            ":torch_kineto_profiling",
            ":torch_mobile_core",
        ],
    )

    fb_xplat_genrule(
        name = "mobile_bytecode_header",
        srcs = [
            "torch/csrc/jit/serialization/mobile_bytecode.fbs",
        ],
        outs = {
            "mobile_bytecode_generated.h": ["mobile_bytecode_generated.h"],
        },
        cmd = "$(exe {})".format(third_party("flatc")) +
              " --cpp --gen-mutable --scoped-enums -o ${OUT} ${SRCS}",
        default_outs = ["."],
    )

    fb_xplat_cxx_library(
        name = "mobile_bytecode",
        header_namespace = "",
        exported_headers = {
            "torch/csrc/jit/serialization/mobile_bytecode_generated.h": ":mobile_bytecode_header[mobile_bytecode_generated.h]",
        },
        exported_deps = [
            third_party("flatbuffers-api"),
        ],
    )

    fb_xplat_cxx_library(
        name = "flatbuffer_serializer",
        srcs = ["torch/csrc/jit/serialization/flatbuffer_serializer.cpp"],
        exported_headers = [
            "torch/csrc/jit/serialization/flatbuffer_serializer.h",
        ],
        compiler_flags = [
            "-g0",
            "-O3",
            "-fexceptions",
            "-frtti",
            "-Wno-deprecated-declarations",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":torch_mobile_module",
            C10,
        ],
        exported_deps = [
            ":flatbuffer_loader",
            ":mobile_bytecode",
            ":torch_mobile_train",
            third_party("flatbuffers-api"),
        ],
    )

    pt_xplat_cxx_library(
        name = "flatbuffer_loader",
        srcs = [
            "torch/csrc/jit/mobile/flatbuffer_loader.cpp",
        ],
        exported_headers = [
            "torch/csrc/jit/mobile/flatbuffer_loader.h",
        ],
        extra_flags = {
            "fbandroid_compiler_flags": ["-frtti"],
        },
        # torch_mobile_deserialize brings in sources neccessary to read a module
        # which depends on mobile module definition
        # link_whole is enable so that all symbols neccessary for mobile module are compiled
        # instead of only symbols used while loading; this prevents symbol
        # found definied in runtime
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":mobile_bytecode",
            ":torch_mobile_deserialize",
            third_party("flatbuffers-api"),
            C10,
        ],
    )

    fb_xplat_cxx_library(
        name = "flatbuffer_serializer_jit",
        srcs = ["torch/csrc/jit/serialization/flatbuffer_serializer_jit.cpp"],
        exported_headers = [
            "torch/csrc/jit/serialization/flatbuffer_serializer_jit.h",
        ],
        compiler_flags = [
            "-g0",
            "-O3",
            "-fexceptions",
            "-frtti",
            "-Wno-deprecated-declarations",
        ],
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":flatbuffer_loader",
            ":flatbuffer_serializer",
            ":mobile_bytecode",
            ":torch_core",
            ":torch_mobile_module",
            third_party("flatbuffers-api"),
            C10,
        ],
    )

    fb_xplat_cxx_library(
        name = "torch_flatbuffer_all",
        visibility = ["PUBLIC"],
        exported_deps = [
            ":flatbuffer_loader",
            ":flatbuffer_serializer",
            ":flatbuffer_serializer_jit",
        ],
    )

    pt_xplat_cxx_library(
        name = "torch_supported_mobile_models",
        srcs = [
            "fb/supported_mobile_models/SupportedMobileModels.cpp",
        ] if NOT_OSS else [],
        header_namespace = "",
        exported_headers = ["fb/supported_mobile_models/SupportedMobileModels.h"] if NOT_OSS else [],
        extra_flags = {
            "exported_preprocessor_flags": ["-DSYMBOLICATE_MOBILE_DEBUG_HANDLE"] if get_enable_eager_symbolication() else [],
        },
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        linker_flags = [
            "-Wl,--no-as-needed",
        ],
        visibility = ["PUBLIC"],
        deps = [],
        exported_deps = [
            "//xplat/caffe2/fb/custom_ops/batch_box_cox:batch_box_cox",
            "//xplat/caffe2/fb/custom_ops/maskrcnn:maskrcnn",
        ] if NOT_OSS else [],
    )

    fb_xplat_cxx_library(
        name = "static_runtime",
        srcs = [
            "torch/csrc/jit/runtime/static/fusion.cpp",
            "torch/csrc/jit/runtime/static/generated_ops.cpp",
            "torch/csrc/jit/runtime/static/impl.cpp",
            "torch/csrc/jit/runtime/static/memory_planner.cpp",
            "torch/csrc/jit/runtime/static/native_ops.cpp",
            "torch/csrc/jit/runtime/static/ops.cpp",
            "torch/csrc/jit/runtime/static/passes.cpp",
            "torch/csrc/jit/runtime/static/te_wrapper.cpp",
        ],
        labels = labels,
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        visibility = ["PUBLIC"],
        windows_preferred_linkage = "static" if is_arvr_mode() else None,
        deps = [
            ":aten_cpu",
            ":caffe2_headers",
            ":torch_core",
            C10,
        ],
    )

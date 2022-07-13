# @lint-ignore-every BUCKLINT supress the warning for using native
load("@bazel_skylib//lib:paths.bzl", "paths")
load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")
load("@fbcode_macros//build_defs:cpp_python_extension.bzl", "cpp_python_extension")
load("@fbcode_macros//build_defs:custom_rule.bzl", "custom_rule")
load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")
load("@fbsource//tools/build_defs:glob_defs.bzl", "glob")
load(
    "//caffe2:build_variables.bzl",
    "glob_libtorch_python_sources",
    "libtorch_cuda_sources",
    "libtorch_nvfuser_generated_headers",
    "libtorch_nvfuser_runtime_sources",
    "libtorch_python_cuda_sources",
    "libtorch_sources",
    "torch_cpp_srcs",
)
load(
    "//caffe2:defs_hip.bzl",
    "get_hip_flags",
    "hip_external_deps",
    "hip_pp_flags",
)
load("//caffe2/caffe2/fb:defs_gpu.bzl", "gpu_library_selector", "gpu_library_targets", "is_amd_build")
load("//tools/build/buck:nccl_deps.bzl", "get_nccl_dependency")

def _path_to_filename(fname):
    return paths.split_extension(paths.basename(fname))[0]

def use_kineto():
    return native.host_info().os.is_linux and native.host_info().arch.is_x86_64 and not is_amd_build()

def add_torch_libs():
    r = {}

    torch_cpp_headers = glob(["torch/csrc/api/include/**/*.h"]) + ["torch/script.h"]
    libtorch_python_sources = glob_libtorch_python_sources()

    use_mpi = native.read_config("fbcode", "caffe2_use_mpi", None)
    enable_flatbuffer = bool(native.read_config("fbcode", "caffe2_enable_flatbuffer", None))

    compiler_flags_cpu = [
        "-DUSE_C10D",
        "-DUSE_NUMPY",
        "-DUSE_SCALARS",
        "-DNO_CUDNN_DESTROY_HANDLE",
        "-DBUILD_CAFFE2",
        "-DTORCH_ENABLE_LLVM",
        "-Wno-write-strings",
        "-Wno-format",
        "-Wno-strict-aliasing",
        "-Wno-non-virtual-dtor",
        "-Wno-shadow-compatible-local",
        "-Wno-empty-body",
    ] + ([] if native.host_info().os.is_windows else [
        # XNNPACK depends on an updated version of pthreadpool interface, whose implementation
        # includes <pthread.h> - a header not available on Windows.
        "-DUSE_XNNPACK",
    ])

    # We should really include preprocessor flags here
    # instead of compiler_flags
    propagated_pp_flags_cpu = [
        "-DSYMBOLICATE_MOBILE_DEBUG_HANDLE",
        "-DUSE_DISTRIBUTED",
        "-DUSE_C10D_GLOO",
        "-DUSE_RPC",
        "-DUSE_TENSORPIPE",
    ] + (
        ["-DUSE_C10D_MPI"] if use_mpi else []
    ) + (
        ["-DUSE_KINETO", "-DUSE_KINETO_UPDATED"] if use_kineto() else []
    ) + (
        ["-DENABLE_LIBKINETO_CLIENT"] if native.read_config("kineto", "enable_libkineto_client", "1") == "1" else []
    )

    compiler_flags_cuda = [
        "-DUSE_CUDNN",
        "-DUSE_NCCL",
    ]

    compiler_flags_hip = []

    propagated_pp_flags_cuda = [
        "-DUSE_CUDA",
        "-DUSE_C10D_NCCL",
    ]

    common_headers = glob([
        "torch/csrc/**/*.h",
        # c10d used to be a separate library whose includes ended in .hpp.
        "torch/csrc/distributed/c10d/*.hpp",
        "torch/csrc/generic/*.cpp",
    ]) + [
        "torch/csrc/deploy/Exception.h",
        "torch/csrc/deploy/deploy.h",
        "torch/csrc/deploy/elf_file.h",
        "torch/csrc/deploy/environment.h",
        "torch/csrc/deploy/interpreter/builtin_registry.h",
        "torch/csrc/deploy/interpreter/interpreter_impl.h",
        "torch/csrc/deploy/loader.h",
        "torch/csrc/deploy/mem_file.h",
        "torch/csrc/deploy/noop_environment.h",
        "torch/csrc/deploy/path_environment.h",
        "torch/csrc/deploy/unity/tests/test_unity.h",
        "torch/csrc/deploy/unity/xar_environment.h",
        "torch/csrc/distributed/rpc/metrics/RpcMetricsHandler.h",
        "test/cpp/jit/test_custom_class_registrations.h",
        "test/cpp/jit/test_utils.h",
        "test/cpp/tensorexpr/gtest_assert_float_eq.h",
        "test/cpp/tensorexpr/padded_buffer.h",
        "test/cpp/tensorexpr/test_base.h",
        "test/cpp/tensorexpr/test_utils.h",
    ]
    common_headers.remove("torch/csrc/jit/serialization/mobile_bytecode_generated.h")

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
        "headers": common_headers,
    }

    include_directories = [
        "..",
        ".",
        "torch/csrc/api/include",
        "torch/csrc",
        # c10d used to be a separate library and its includes were c10d/Foo.hpp,
        # hence we now need this hack to keep supporting them.
        "torch/csrc/distributed",
        "torch/csrc/nn",
    ]

    _libtorch_sources = list(libtorch_sources())

    # Add the Gloo and TensorPipe backends specific to Facebook networking.
    _libtorch_sources.append("torch/csrc/distributed/c10d/fb/GlooDeviceFactory.cpp")
    _libtorch_sources.append("torch/csrc/distributed/rpc/fb/tensorpipe_agent.cpp")

    cpp_library(
        name = "libtorch",
        srcs = _libtorch_sources + ([
            "torch/csrc/jit/serialization/flatbuffer_serializer.cpp",
            "torch/csrc/jit/serialization/flatbuffer_serializer_jit.cpp",
            "torch/csrc/jit/mobile/flatbuffer_loader.cpp",
        ] if enable_flatbuffer else []),
        link_whole = True,
        include_directories = include_directories,
        propagated_pp_flags = propagated_pp_flags_cpu,
        exported_deps = (
            [
                ":ATen-cpu",
                ":generated-autograd-headers",
                ":generated-lazy-headers",
                "//caffe2:version_cpp",
                "//caffe2/caffe2:caffe2_cpu",
                "//caffe2/caffe2/quantization/server:dnnlowp_ops",
                "//caffe2/caffe2/serialize:inline_container",
                "//caffe2/torch/lib/libshm:libshm",
                "//gloo:gloo",
                "//gloo/fb/transport/tls:tls",
                "//gloo/transport/tcp:tcp",
                "//tensorpipe:tensorpipe_cpu",
            ] + (["//kineto/libkineto:kineto"] if use_kineto() else []) +
            (["//caffe2:mobile_bytecode"] if enable_flatbuffer else [])
        ),
        exported_external_deps = [
            ("nanopb", None, "protobuf-nanopb"),
            ("protobuf", None),
            ("llvm-fb", None, "LLVMAnalysis"),
            ("llvm-fb", None, "LLVMBPFAsmParser"),
            ("llvm-fb", None, "LLVMBPFCodeGen"),
            ("llvm-fb", None, "LLVMCodeGen"),
            ("llvm-fb", None, "LLVMCore"),
            ("llvm-fb", None, "LLVMExecutionEngine"),
            ("llvm-fb", None, "LLVMIRReader"),
            ("llvm-fb", None, "LLVMInstCombine"),
            ("llvm-fb", None, "LLVMInterpreter"),
            ("llvm-fb", None, "LLVMMC"),
            ("llvm-fb", None, "LLVMNVPTXCodeGen"),
            ("llvm-fb", None, "LLVMOrcJIT"),
            ("llvm-fb", None, "LLVMRISCVAsmParser"),
            ("llvm-fb", None, "LLVMRISCVCodeGen"),
            ("llvm-fb", None, "LLVMScalarOpts"),
            ("llvm-fb", None, "LLVMSupport"),
            ("llvm-fb", None, "LLVMTarget"),
            ("llvm-fb", None, "LLVMTransformUtils"),
            ("llvm-fb", None, "LLVMVectorize"),
            ("llvm-fb", None, "LLVMAArch64AsmParser"),
            ("llvm-fb", None, "LLVMAArch64CodeGen"),
            ("llvm-fb", None, "LLVMAArch64Info"),
            ("llvm-fb", None, "LLVMWebAssemblyAsmParser"),
            ("llvm-fb", None, "LLVMWebAssemblyCodeGen"),
            ("llvm-fb", None, "LLVMWebAssemblyInfo"),
            ("llvm-fb", None, "LLVMX86AsmParser"),
            ("llvm-fb", None, "LLVMX86CodeGen"),
            ("llvm-fb", None, "LLVMipo"),
        ] + ([("openmpi", None, "openmpi")] if use_mpi else []),
        compiler_flags = compiler_flags_cpu,
        **common_flags
    )

    # Below rules are used to stringify NVfuser runtime library into a header files
    python_binary(
        name = "nvfuser-stringify",
        srcs = ["torch/csrc/jit/codegen/cuda/tools/stringify_file.py"],
        base_module = "",
        main_module = "torch.csrc.jit.codegen.cuda.tools.stringify_file",
    )

    # files in libtorch_nvfuser_runtime_sources that are violating package boundaries
    # are mapped to their corresponding export_file rules.
    violation_paths_to_rule = {
        "aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh": ":aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh",
        "aten/src/ATen/cuda/detail/UnpackRaw.cuh": ":aten/src/ATen/cuda/detail/UnpackRaw.cuh",
    }

    for name in libtorch_nvfuser_runtime_sources:
        src_path = violation_paths_to_rule.get(name, name)
        filename = _path_to_filename(src_path)
        native.genrule(
            name = "gen-nvfuser-hdr={}.h".format(filename),
            srcs = {name: src_path},
            bash = "$(exe :nvfuser-stringify) -i $SRCDIR/{} -o $OUT".format(name),
            out = "{}.h".format(filename),
        )
    cpp_library(
        name = "generated-nvfuser-headers",
        headers = [":gen-nvfuser-hdr=" + x for x in libtorch_nvfuser_generated_headers],
        header_namespace = "nvfuser_resources",
    )

    _libtorch_cuda_sources = list(libtorch_cuda_sources)
    cpp_library(
        name = "libtorch_cuda",
        srcs = _libtorch_cuda_sources,
        link_whole = True,
        include_directories = include_directories,
        # TODO: putting USE_CUDA in propagated_pp_flags is error-prone
        propagated_pp_flags = propagated_pp_flags_cuda,
        exported_deps = [
            ":ATen",
            ":generated-aten-headers-cuda",
            ":generated-autograd-headers",
            ":generated-nvfuser-headers",
            ":libtorch",
            "//caffe2/caffe2:caffe2_cpu",
            "//caffe2/caffe2:caffe2_gpu",
            "//caffe2/torch/lib/libshm:libshm",
            "//gloo:gloo_gpu_cuda",
            "//tensorpipe:tensorpipe_cuda",
        ] + get_nccl_dependency(),
        exported_external_deps = [
            ("cudnn", None, "cudnn-lazy"),
            ("cuda", None, "nvToolsExt-lazy"),
            ("cuda", None, "nvrtc-lazy"),
            ("cuda", None, "nvrtc-builtins-lazy"),
        ],
        compiler_flags = compiler_flags_cpu + compiler_flags_cuda,
        **common_flags
    )

    # (original_paths, hipified_paths)
    libtorch_hip_headers_filter = torch_cpp_headers + [h for h in common_headers if any([h.startswith(d) for d in [
        # headers in the following directories are added to libtorch_hip_headers_filter
        # so that they are not hipified.
        "torch/csrc/deploy/",
        "torch/csrc/distributed/rpc/metrics/",
        "torch/csrc/jit/serialization/",
        "torch/cpp/jit/",
        "torch/cpp/tensorexpr/",
    ]])]
    libtorch_hip_sources = (libtorch_cuda_sources, [f.replace(".cu", ".hip") for f in libtorch_cuda_sources])
    libtorch_hip_headers = ([f for f in common_headers if f not in libtorch_hip_headers_filter],) * 2

    custom_rule(
        name = "fb_libtorch_hipify_gen",
        srcs = libtorch_hip_sources[0] + libtorch_hip_headers[0],
        build_args = "--source-dir= --hipify-dir= --copy-dir= --rewrite-cu-ext",
        build_script_dep = "//caffe2:fb_caffe2_hipify",
        output_gen_files = libtorch_hip_sources[1] + libtorch_hip_headers[1],
    )

    cpp_library(
        name = "libtorch_hip_headers",
        headers = [":fb_libtorch_hipify_gen={}".format(f) for f in libtorch_hip_headers[1]],
        header_namespace = "",
    )

    cpp_library(
        name = "libtorch_hip",
        srcs = [":fb_libtorch_hipify_gen={}".format(f) for f in libtorch_hip_sources[1]],
        headers = [f for f in common_headers if f in libtorch_hip_headers_filter],
        link_whole = True,
        propagated_pp_flags = hip_pp_flags,
        exported_deps = [
            ":generated-aten-headers-hip",
            ":generated-autograd-headers",
            ":generated-nvfuser-headers",
            ":libtorch",
            ":libtorch_hip_headers",
            "//caffe2:ATen-hip",
            "//caffe2/caffe2:caffe2_cpu",
            "//caffe2/caffe2:caffe2_gpu_hip",
            "//caffe2/torch/lib/libshm:libshm",
            "//gloo:gloo_gpu_hip",
            "//tensorpipe:tensorpipe_cpu",  # TODO: include a HIP version once it's developed
        ],
        exported_external_deps = hip_external_deps,
        compiler_flags = compiler_flags_cpu + compiler_flags_hip + [
            "-Wno-unused-result",
        ],
        hip_flags = ["-Wno-unused-result"] + get_hip_flags(),
        compiler_specific_flags = common_flags["compiler_specific_flags"],
    )

    gpu_library_targets(
        name = "libtorch_gpu",
        deps_cpu = [
            ":libtorch",
        ],
        deps_cuda = [
            ":libtorch_cuda",
        ],
        deps_hip = [
            ":libtorch_hip",
        ],
        exclude_hip_target = False,
        extra_external_deps = [],
    )

    # torch-cpp is still conditionally compiled based on USE_CUDA. Ideally we'd
    # separate it out as an additive library instead.
    gpu_library_selector(
        name = "torch-cpp",
        deps_cpu = [":torch-cpp-cpu"],
        deps_cuda = [":torch-cpp-cuda"],
        deps_hip = [":torch-cpp-hip"],
        merge_cpu_deps = False,
        exclude_hip_target = False,
    )

    # USE_CUDA flag is propagated through propagated_pp_flags on libtorch
    cpp_library(
        name = "torch-cpp-cuda",
        srcs = torch_cpp_srcs,
        headers = torch_cpp_headers,
        include_directories = [
            ".",
            "torch/csrc/api/include/",
        ],
        exported_deps = [
            ":libtorch_cuda",
            "//caffe2/torch/fb/init:init",
        ],
        exported_external_deps = [
            ("cuda", None, "cuda-lazy"),
            ("cudnn", None, "cudnn-lazy"),
        ],
    )

    cpp_library(
        name = "torch-cpp-hip",
        srcs = torch_cpp_srcs,
        headers = torch_cpp_headers,
        include_directories = [
            ".",
            "torch/csrc/api/include/",
        ],
        exported_deps = [
            ":libtorch_hip",
            "//caffe2/torch/fb/init:init",
        ],
        exported_external_deps = hip_external_deps,
    )

    cpp_library(
        name = "torch-cpp-cpu",
        srcs = torch_cpp_srcs,
        headers = torch_cpp_headers,
        include_directories = [
            ".",
            "torch/csrc/api/include/",
        ],
        exported_deps = [
            ":libtorch",
            "//caffe2/torch/fb/init:init",
        ],
    )

    # _C_impl is still conditionally compiled based on USE_CUDA. Ideally we'd
    # separate it out as an additive library instead.
    # TODO: split it into cpp and cuda parts similarly to libtorch
    gpu_library_selector(
        name = "_C_impl",
        deps_cpu = [":_C_impl_cpu"],
        deps_cuda = [":_C_impl_cuda"],
        deps_hip = [":_C_impl_hip"],
        merge_cpu_deps = False,
        exclude_hip_target = False,
    )

    cpp_library(
        name = "_C_impl_cpu",
        srcs = libtorch_python_sources,
        link_whole = True,
        exported_deps = [
            "fbsource//third-party/fmt:fmt",
            ":torch-cpp-cpu",
            "//caffe2/torch/fb/init:init",
            "//caffe2/torch/lib/libshm:libshm",
        ],
        exported_external_deps = [
            ("numpy", None, "cpp"),
            ("pybind11", None),
            ("python", None),
        ],
        compiler_flags = compiler_flags_cpu,
        compiler_specific_flags = common_flags["compiler_specific_flags"],
    )

    # This target is used to help get headers for compile-time deps for torch::deploy
    # libinterpreter.so build _without_ getting link-time deps, which are supplied
    # separately by the application that dlopens libinterpreter.so.
    #
    # We make use of the buck auto-generated #headers flavor of a target to accomplish this.
    #
    # However, since #headers flavor of target with srcs can't be used in all build modes, we
    # work around this limitation by using this 'pass-through' target, which has a usable
    # #headers flavor in all build modes.
    cpp_library(
        name = "headers_for_torch_python_deps",
        exported_deps = [
            ":_C_impl_cpu",
        ],
    )
    cpp_library(
        name = "headers_for_torch_python_cuda_deps",
        exported_deps = [
            ":_C_impl_cuda",
        ],
    )

    # This target compiles torch_python bindings, but skips the deps on actual
    # torch and python since those will be integrated specially in the wrapper for
    # libinterpreter.so used in torch::deploy
    cpp_library(
        name = "torch_python_without_torch",
        srcs = libtorch_python_sources + torch_cpp_srcs,
        undefined_symbols = True,
        preferred_linkage = "static",
        exported_deps = [
            ":headers_for_torch_python_deps#headers",
        ],
        exported_external_deps = [
            ("pybind11", None),
            ("frozenpython", None, "python-headers"),
        ],
        compiler_flags = compiler_flags_cpu + [
            # some code in the Python bindings compiles differently
            # when you are deploy
            "-DUSE_DEPLOY",
        ],
        compiler_specific_flags = common_flags["compiler_specific_flags"],
    )

    cpp_library(
        name = "torch_python_cuda_without_torch",
        srcs = libtorch_python_sources + torch_cpp_srcs + libtorch_python_cuda_sources,
        undefined_symbols = True,
        preferred_linkage = "static",
        exported_deps = [
            ":headers_for_torch_python_cuda_deps#headers",
        ],
        exported_external_deps = [
            ("pybind11", None),
            ("frozenpython", None, "python-headers"),
        ],
        compiler_flags = compiler_flags_cpu + [
            "-DUSE_CUDA",
            # some code in the Python bindings compiles differently
            # when you are deploy
            "-DUSE_DEPLOY",
        ],
        compiler_specific_flags = common_flags["compiler_specific_flags"],
    )

    cpp_library(
        name = "_C_impl_cuda",
        srcs = libtorch_python_sources + libtorch_python_cuda_sources,
        link_whole = True,
        exported_deps = [
            "fbsource//third-party/fmt:fmt",
            ":torch-cpp-cuda",
            "//caffe2/torch/fb/init:init",
            "//caffe2/torch/lib/libshm:libshm",
        ],
        exported_external_deps = [
            ("numpy", None, "cpp"),
            ("pybind11", None),
            ("python", None),
        ],
        compiler_flags = compiler_flags_cpu + compiler_flags_cuda,
        compiler_specific_flags = common_flags["compiler_specific_flags"],
    )

    # Autogenerated files whose rules contain ":" are not hipified.
    libtorch_python_hip_sources = [f for f in (libtorch_python_sources + libtorch_python_cuda_sources) if ":" in f]
    libtorch_python_hip_sources_hipified = [f for f in (libtorch_python_sources + libtorch_python_cuda_sources) if not ":" in f]

    custom_rule(
        name = "fb_C_impl_hipify_gen",
        srcs = libtorch_python_hip_sources_hipified,
        build_args = "--source-dir= --hipify-dir= --copy-dir=",
        build_script_dep = "//caffe2:fb_caffe2_hipify",
        output_gen_files = libtorch_python_hip_sources_hipified,
    )

    cpp_library(
        name = "_C_impl_hip",
        srcs = [":fb_C_impl_hipify_gen={}".format(f) for f in (libtorch_python_hip_sources_hipified)] + libtorch_python_hip_sources,
        link_whole = True,
        exported_deps = [
            "fbsource//third-party/fmt:fmt",
            ":torch-cpp-hip",
            "//caffe2/torch/fb/init:init",
            "//caffe2/torch/lib/libshm:libshm",
        ],
        exported_external_deps = [
            ("numpy", None, "cpp"),
            ("pybind11", None),
            ("python", None),
        ],
        compiler_flags = compiler_flags_cpu + compiler_flags_hip + ["-Wno-unused-result"],
        compiler_specific_flags = common_flags["compiler_specific_flags"],
    )

    cpp_python_extension(
        name = "_C",
        srcs = [
            "torch/csrc/stub.c",
        ],
        base_module = "torch",
        deps = [
            ":_C_impl",
            "//caffe2:flatbuffer_loader",
        ],
    )

    cpp_python_extension(
        name = "_C_flatbuffer",
        srcs = [
            "torch/csrc/stub_with_flatbuffer.c",
            "torch/csrc/init_flatbuffer_module.cpp",
        ],
        base_module = "torch",
        deps = [
            ":_C_impl",
            "//caffe2:flatbuffer_loader",
            "//caffe2:flatbuffer_serializer",
        ],
    )

    return r

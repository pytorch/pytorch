load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")
load("@fbcode_macros//build_defs:custom_rule.bzl", "custom_rule")
load("//caffe2:build.bzl", "GENERATED_CPP")
load("//caffe2:build_variables.bzl", "jit_core_headers", "jit_core_sources")
load("//caffe2/tools:sgx_target_definitions.bzl", "is_sgx")

default_compiler_flags = [
    "-Wno-error=strict-aliasing",
    "-Wno-unused-local-typedefs",
    "-Wno-shadow-compatible-local",
    "-Wno-maybe-uninitialized",  # aten is built with gcc as part of HHVM
    "-Wno-unknown-pragmas",
    "-Wno-strict-overflow",
    # See https://fb.facebook.com/groups/fbcode/permalink/1813348245368673/
    # These trigger on platform007
    "-Wno-stringop-overflow",
    "-Wno-class-memaccess",
    "-DHAVE_MMAP",
    "-DUSE_GCC_ATOMICS=1",
    "-D_FILE_OFFSET_BITS=64",
    "-DHAVE_SHM_OPEN=1",
    "-DHAVE_SHM_UNLINK=1",
    "-DHAVE_MALLOC_USABLE_SIZE=1",
    "-DTH_HAVE_THREAD",
    "-DCPU_CAPABILITY_DEFAULT",
    "-DTH_INDEX_BASE=0",
    "-DMAGMA_V2",
    "-DNO_CUDNN_DESTROY_HANDLE",
    "-DUSE_QNNPACK",
    "-DUSE_PYTORCH_QNNPACK",
    # The dynamically loaded NVRTC trick doesn't work in fbcode,
    # and it's not necessary anyway, because we have a stub
    # nvrtc library which we load canonically anyway
    "-DUSE_DIRECT_NVRTC",
    "-DUSE_XNNPACK",
    "-Wno-error=uninitialized",
]

compiler_specific_flags = {
    "clang": [
        "-Wno-absolute-value",
        "-Wno-pass-failed",
        "-Wno-braced-scalar-init",
    ],
    "gcc": [
        "-Wno-error=array-bounds",
    ],
}

def add_sgx_aten_libs(ATEN_HEADERS_CPU_MKL, ATEN_SRCS_CPU_MKL, ATEN_CORE_CPP):
    # we do not need to define these targets if we are in not SGX mode
    if not is_sgx:
        return

    x64_compiler_flags = [
        "-DUSE_SSE2",
        "-DUSE_SSE3",
        "-DUSE_SSE4_1",
        "-DUSE_SSE4_2",
        # dont enable AVX2 because we dont have runtime dispatch
        "-DCPU_CAPABILITY_DEFAULT",
        "-DCPU_CAPABILITY=DEFAULT",
        "-DTH_INDEX_BASE=0",
        "-DTH_INDEX_BASE=0",
        "-msse",
        "-msse2",
        "-msse3",
        "-msse4",
        "-msse4.1",
        "-msse4.2",
        "-mavx",
        "-mavx2",
    ]

    cpu_preprocessor_flags = [
        "-DATEN_MKLDNN_ENABLED_FBCODE=0",
        "-DATEN_NNPACK_ENABLED_FBCODE=0",
        "-DATEN_MKL_ENABLED_FBCODE=0",
        "-DAT_BUILD_WITH_BLAS_FBCODE=1",
        "-DAT_BLAS_USE_CBLAS_DOT_FBCODE=1",
        "-DAT_BLAS_F2C_FBCODE=0",
        "-DATEN_CUDNN_ENABLED_FBCODE=1",
        "-DATEN_ROCM_ENABLED_FBCODE=0",
        "-DC10_MOBILE",
        "-DAT_PARALLEL_NATIVE_FBCODE=1",
    ]

    custom_rule(
        name = "generate-sgx-config",
        srcs = [
            "src/ATen/Config.h.in",
        ],
        build_args = " ".join([
            "--input-file",
            "src/ATen/Config.h.in",
            "--output-file",
            "Config.h",
            "--replace",
            "@AT_MKLDNN_ENABLED@",
            "0",
            "--replace",
            "@AT_MKL_ENABLED@",
            "0",
            "--replace",
            "@AT_MKL_SEQUENTIAL@",
            "0",
            "--replace",
            "@AT_FFTW_ENABLED@",
            "0",
            "--replace",
            "@AT_POCKETFFT_ENABLED@",
            "0",
            "--replace",
            "@AT_NNPACK_ENABLED@",
            "ATEN_NNPACK_ENABLED_FBCODE",
            "--replace",
            "@AT_BUILD_WITH_BLAS@",
            "1",
            "--replace",
            "@AT_BUILD_WITH_LAPACK@",
            "0",
            "--replace",
            "@CAFFE2_STATIC_LINK_CUDA_INT@",
            "0",
            "--replace",
            "@AT_BLAS_F2C@",
            "AT_BLAS_F2C_FBCODE",
            "--replace",
            "@AT_BLAS_USE_CBLAS_DOT@",
            "AT_BLAS_USE_CBLAS_DOT_FBCODE",
            "--replace",
            "@AT_PARALLEL_OPENMP@",
            "0",
            "--replace",
            "@AT_PARALLEL_NATIVE@",
            "1",
            "--replace",
            "@AT_PARALLEL_NATIVE_TBB@",
            "0",
        ]),
        build_script_dep = "//caffe2:substitute",
        output_gen_files = ["Config.h"],
    )

    cpp_library(
        name = "generated-sgx-config-header",
        headers = [":generate-sgx-config=Config.h"],
        header_namespace = "ATen",
    )

    ATEN_CORE_H = native.glob([
        "src/ATen/core/*.h",
        "src/ATen/core/boxing/*.h",
        "src/ATen/core/boxing/impl/*.h",
        "src/ATen/core/dispatch/*.h",
        "src/ATen/core/op_registration/*.h",
    ]) + [
        "src/ATen/CPUGeneratorImpl.h",
        "src/ATen/NumericUtils.h",
    ]

    cpp_library(
        name = "ATen-core-sgx-headers",
        headers = ATEN_CORE_H,
        propagated_pp_flags = [
            "-Icaffe2/aten/src",
        ],
        exported_deps = [
            "//caffe2:generated-aten-headers-core",
            "//caffe2/c10:c10",
        ],
    )

    cpp_library(
        name = "ATen-sgx-core",
        # Sorry, this is duped with GENERATED_CPP_CORE.  I was too lazy to refactor
        # the list into a bzl file
        srcs = ATEN_CORE_CPP + [
            ":gen_aten=Operators_0.cpp",
            ":gen_aten=Operators_1.cpp",
            ":gen_aten=Operators_2.cpp",
            ":gen_aten=Operators_3.cpp",
            ":gen_aten=Operators_4.cpp",
            ":gen_aten=core/ATenOpList.cpp",
            ":gen_aten=core/TensorMethods.cpp",
        ],
        headers = native.glob([
            "src/ATen/*.h",
            "src/ATen/ops/*.h",
            "src/ATen/quantized/*.h",
        ]),
        compiler_flags = default_compiler_flags,
        compiler_specific_flags = compiler_specific_flags,
        link_whole = True,
        # Tests that fail in CPU static dispatch mode because they require
        # the dispatcher in order to work can be gated out with `#ifndef
        # ATEN_CPU_STATIC_DISPATCH`.
        propagated_pp_flags = [],
        # Must be linked with caffe2_core
        undefined_symbols = True,
        exported_deps = [
            ":ATen-core-sgx-headers",
            "//caffe2:jit-core-sgx",
        ],
    )

    cpp_library(
        name = "ATen-sgx-cpu",
        srcs = ATEN_SRCS_CPU_MKL + [":gen_aten=" + x for x in GENERATED_CPP],
        headers = ATEN_HEADERS_CPU_MKL,
        arch_compiler_flags = {"x86_64": x64_compiler_flags},
        compiler_flags = default_compiler_flags,
        compiler_specific_flags = compiler_specific_flags,
        include_directories = [
            "src",
            "src/TH",
        ],
        link_whole = True,
        propagated_pp_flags = cpu_preprocessor_flags,
        exported_deps = [
            "fbsource//third-party/cpuinfo_sgx:cpuinfo_coffeelake",
            ":ATen-sgx-core",
            ":aten-headers-cpu",
            ":generated-aten-headers-cpu",
            ":generated-sgx-config-header",
            ":generated-sgx-th-general-header",
            ":generated-sgx-th-general-header-no-prefix",
            "//caffe2/caffe2:caffe2_sgx_core",
            "//caffe2/caffe2/perfkernels:sgx_perfkernels",
            "//xplat/third-party/XNNPACK:XNNPACK",
        ],
        exported_external_deps = [
            ("OpenBLAS", None, "OpenBLAS"),
        ],
        deps = [
            "//caffe2/aten/src/ATen/native/quantized/cpu/qnnpack:pytorch_qnnpack",
        ],
    )

def add_sgx_aten_jit_libs():
    # we do not need to define these targets if we are in not SGX mode
    if not is_sgx:
        return

    cpp_library(
        name = "jit-core-sgx",
        # Sorry, this is duped with GENERATED_CPP_CORE.  I was too lazy to refactor
        # the list into a bzl file
        srcs = jit_core_sources,
        headers = jit_core_headers,
        compiler_flags = default_compiler_flags,
        compiler_specific_flags = compiler_specific_flags,
        include_directories = [""],
        link_whole = True,
        # Must be linked with caffe2_core
        undefined_symbols = True,
        exported_deps = [
            "//caffe2:ATen-core-sgx-headers",
            "//caffe2/c10:c10",
        ],
    )

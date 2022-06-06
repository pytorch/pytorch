# @nolint
load("//arvr/tools/build_defs:genrule_utils.bzl", "gen_cmake_header")
load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library")
load(
    "@fbsource//xplat/caffe2:pt_defs.bzl",
    "gen_aten_files",
    "get_aten_codegen_extra_params",
)

def define_aten_gen():
    backends = [
        "CPU",
        "SparseCPU",
        "SparseCsrCPU",
        # "MkldnnCPU",
        "CUDA",
        "SparseCUDA",
        "SparseCsrCUDA",
        "QuantizedCPU",
        "QuantizedCUDA",
        "Meta",
        "ZeroTensor"
    ]

    gen_aten_files(
        name = "gen_aten_ovrsource",
        extra_flags = get_aten_codegen_extra_params(backends),
        visibility = ["PUBLIC"],
    )

    oxx_static_library(
        name = "ovrsource_aten_generated_cuda_headers",
        header_namespace = "ATen",
        public_generated_headers = {
            "CUDAFunctions.h": ":gen_aten_ovrsource[CUDAFunctions.h]",
            "CUDAFunctions_inl.h": ":gen_aten_ovrsource[CUDAFunctions_inl.h]",
        },
        visibility = ["PUBLIC"],
    )

    oxx_static_library(
        name = "ovrsource_aten_generated_meta_headers",
        header_namespace = "ATen",
        public_generated_headers = {
            "MetaFunctions.h": ":gen_aten_ovrsource[MetaFunctions.h]",
            "MetaFunctions_inl.h": ":gen_aten_ovrsource[MetaFunctions_inl.h]",
        },
        visibility = ["PUBLIC"],
    )

    gen_cmake_header(
        src = "aten/src/ATen/Config.h.in",
        defines = [
            ("@AT_MKLDNN_ENABLED@", "0"),
            ("@AT_MKL_ENABLED@", "0"),
            ("@AT_MKL_SEQUENTIAL@", "0"),
            ("@AT_FFTW_ENABLED@", "0"),
            ("@AT_NNPACK_ENABLED@", "0"),
            ("@AT_PARALLEL_OPENMP@", "0"),
            ("@AT_PARALLEL_NATIVE@", "1"),
            ("@AT_PARALLEL_NATIVE_TBB@", "0"),
            ("@AT_POCKETFFT_ENABLED@", "0"),
            ("@CAFFE2_STATIC_LINK_CUDA_INT@", "1"),
            ("@AT_BUILD_WITH_BLAS@", "1"),
            ("@AT_BUILD_WITH_LAPACK@", "1"),
            ("@AT_BLAS_F2C@", "1"),
            ("@AT_BLAS_USE_CBLAS_DOT@", "0")
        ],
        header = "ATen/Config.h",
        prefix = "ovrsource_aten_",
    )

    gen_cmake_header(
        src = "aten/src/ATen/cuda/CUDAConfig.h.in",
        defines = [
            ("@AT_CUDNN_ENABLED@", "1"),
            ("@AT_ROCM_ENABLED@", "0"),
            ("@NVCC_FLAGS_EXTRA@", " "),
            ("@AT_MAGMA_ENABLED@", "0")
        ],
        header = "ATen/cuda/CUDAConfig.h",
        prefix = "ovrsource_aten_",
    )

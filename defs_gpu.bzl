load("@fbcode_macros//build_defs:native_rules.bzl", "buck_genrule")
load(
    "//caffe2/caffe2:defs_hip.bzl",
    "get_caffe2_hip_headers",
    "get_caffe2_hip_srcs",
)
load(":ufunc_defs.bzl", "aten_ufunc_names")

ATEN_CUDA_H_PATTERN = [
    "aten/src/ATen/cuda/*.h",
    "aten/src/ATen/cuda/detail/*.h",
    "aten/src/ATen/cuda/nvrtc_stub/*.h",
    "aten/src/ATen/cuda/*.cuh",
    "aten/src/ATen/cuda/detail/*.cuh",
]

ATEN_CUDA_CPP_PATTERN = [
    "aten/src/ATen/cuda/*.cpp",
    "aten/src/ATen/cuda/detail/*.cpp",
    "aten/src/ATen/cuda/nvrtc_stub/*.cpp",
]

ATEN_CUDA_CU_PATTERN = [
    "aten/src/ATen/cuda/*.cu",
    "aten/src/ATen/cuda/detail/*.cu",
]

ATEN_CUDNN_H_PATTERN = [
    "aten/src/ATen/cudnn/*.h",
    "aten/src/ATen/cudnn/*.cuh",
]

ATEN_CUDNN_CPP_PATTERN = ["aten/src/ATen/cudnn/*.cpp"]

ATEN_MIOPEN_H_PATTERN = [
    "aten/src/ATen/miopen/*.h",
    "aten/src/ATen/miopen/*.cuh",
]

ATEN_MIOPEN_CPP_PATTERN = ["aten/src/ATen/miopen/*.cpp"]

ATEN_NATIVE_CUDNN_CPP_PATTERN = ["aten/src/ATen/native/cudnn/*.cpp"]

ATEN_NATIVE_MIOPEN_CPP_PATTERN = ["aten/src/ATen/native/miopen/*.cpp"]

ATEN_NATIVE_CUDA_CU_PATTERN = [
    "aten/src/ATen/native/cuda/*.cu",
    "aten/src/ATen/native/nested/cuda/*.cu",
    "aten/src/ATen/native/quantized/cuda/*.cu",
    "aten/src/ATen/native/sparse/cuda/*.cu",
    "aten/src/ATen/native/transformers/**/*.cu",
]

ATEN_NATIVE_CUDA_CPP_PATTERN = [
    "aten/src/ATen/native/cuda/*.cpp",
    "aten/src/ATen/native/cuda/linalg/*.cpp",
    "aten/src/ATen/native/nested/cuda/*.cpp",
    "aten/src/ATen/native/sparse/cuda/*.cpp",
    "aten/src/ATen/native/transformers/cuda/*.cpp",
]

ATEN_NATIVE_CUDA_H_PATTERN = [
    "aten/src/ATen/native/cudnn/**/*.h",
    "aten/src/ATen/native/cuda/**/*.h",
    "aten/src/ATen/native/cuda/**/*.cuh",
    "aten/src/ATen/native/sparse/cuda/*.h",
    "aten/src/ATen/native/sparse/cuda/*.cuh",
    "aten/src/ATen/native/quantized/cuda/*.h",
    "aten/src/ATen/native/transformers/cuda/*.h",
    "aten/src/ATen/native/transformers/**/*.cuh",
]

# T66678203: Clang CUDA rollout
ATEN_CUDA_CLANG_CU_PATTERN = [
    "aten/src/ATen/native/cuda/DistributionBernoulli.cu",
]

### Cuda Files
def get_aten_cuda_headers():
    ATEN_CUDA_H = native.glob(ATEN_CUDA_H_PATTERN)
    ATEN_NATIVE_CUDA_H = native.glob(ATEN_NATIVE_CUDA_H_PATTERN)
    ATEN_CUDNN_H = native.glob(ATEN_CUDNN_H_PATTERN)
    return ATEN_CUDA_H + ATEN_NATIVE_CUDA_H + ATEN_CUDNN_H

def get_aten_cuda_srcs():
    ATEN_CUDA_CU = native.glob(ATEN_CUDA_CU_PATTERN)
    ATEN_NATIVE_CUDA_CU = native.glob(
        ATEN_NATIVE_CUDA_CU_PATTERN,
        exclude = ATEN_CUDA_CLANG_CU_PATTERN,
    )
    return ATEN_CUDA_CU + ATEN_NATIVE_CUDA_CU

def get_aten_cuda_clang_srcs():
    return native.glob(ATEN_CUDA_CLANG_CU_PATTERN)

# CPU+CUDA file
# Note that these sources and headers include the CPU lists too
def get_all_cuda_srcs():
    ATEN_NATIVE_CUDNN_CPP = native.glob(ATEN_NATIVE_CUDNN_CPP_PATTERN)
    ATEN_CUDNN_CPP = native.glob(ATEN_CUDNN_CPP_PATTERN)
    ATEN_NATIVE_MIOPEN_CPP = native.glob(ATEN_NATIVE_MIOPEN_CPP_PATTERN)
    ATEN_CUDA_CPP = native.glob(ATEN_CUDA_CPP_PATTERN)
    ATEN_NATIVE_CUDA_CPP = native.glob(ATEN_NATIVE_CUDA_CPP_PATTERN)

    return ATEN_NATIVE_CUDNN_CPP + ATEN_CUDNN_CPP + ATEN_NATIVE_MIOPEN_CPP + ATEN_CUDA_CPP + ATEN_NATIVE_CUDA_CPP + get_aten_cuda_srcs()

### HIP files
# Files that must be hipified
def get_aten_hip_srcs():
    ## CU -> HIP files
    ATEN_CUDA_CU = native.glob(ATEN_CUDA_CU_PATTERN)

    # HIP does not use clang for ATEN_CUDA_CLANG_CU_PATTERN
    ATEN_NATIVE_CUDA_CU = native.glob(ATEN_NATIVE_CUDA_CU_PATTERN)

    ## CPU files
    ATEN_NATIVE_CUDNN_CPP = native.glob(ATEN_NATIVE_CUDNN_CPP_PATTERN)
    ATEN_CUDNN_CPP = native.glob(ATEN_CUDNN_CPP_PATTERN)
    ATEN_CUDA_CPP = native.glob(ATEN_CUDA_CPP_PATTERN)
    ATEN_NATIVE_CUDA_CPP = native.glob(ATEN_NATIVE_CUDA_CPP_PATTERN)

    # Get hipified file names (before, after)
    srcs = ATEN_CUDA_CU + ATEN_NATIVE_CUDA_CU + ATEN_NATIVE_CUDNN_CPP + ATEN_CUDNN_CPP + ATEN_CUDA_CPP + ATEN_NATIVE_CUDA_CPP
    ret = get_caffe2_hip_srcs(include_patterns = [], include_files = srcs, project_dir = "")
    return (ret[0], [f.replace("aten/src/", "") for f in ret[1]])

def get_aten_hip_headers():
    ATEN_CUDA_H = native.glob(ATEN_CUDA_H_PATTERN)
    ATEN_NATIVE_CUDA_H = native.glob(ATEN_NATIVE_CUDA_H_PATTERN)
    ATEN_CUDNN_H = []  # native.glob(ATEN_CUDNN_H_PATTERN)

    # Get hipified file names (before, after)
    srcs = ATEN_CUDA_H + ATEN_NATIVE_CUDA_H + ATEN_CUDNN_H
    ret = get_caffe2_hip_headers(include_patterns = [], include_files = ATEN_CUDA_H + ATEN_NATIVE_CUDA_H + ATEN_CUDNN_H, project_dir = "")
    return ret[0], [f.replace("aten/src/", "") for f in ret[1]]

# Native HIP-aware files
def get_aten_hip_native_srcs():
    HIP_IMPL_CPP = native.glob(["aten/src/ATen/hip/impl/*.cpp"])
    ATEN_MIOPEN_CPP = native.glob(ATEN_MIOPEN_CPP_PATTERN)
    ATEN_NATIVE_MIOPEN_CPP = native.glob(ATEN_NATIVE_MIOPEN_CPP_PATTERN)
    return HIP_IMPL_CPP + ATEN_MIOPEN_CPP + ATEN_NATIVE_MIOPEN_CPP

def get_aten_hip_native_headers():
    HIP_IMPL_H = native.glob(["aten/src/ATen/hip/impl/*.h"])
    ATEN_MIOPEN_H = native.glob(ATEN_MIOPEN_H_PATTERN)
    return HIP_IMPL_H + ATEN_MIOPEN_H

def get_aten_hip_ufunc_generated_cuda_sources(gencode_pattern = "{}"):
    # Contents of these CUDA files do not need to be hipified at this point,
    # but they must be renamed from ".cu" to ".hip" because, unlike OSS, a compiler
    # is selected based on a file extension.

    renamed_rules = []
    for n in aten_ufunc_names:
        cuda_name = "UfuncCUDA_{}.cu".format(n)
        hip_name = "UfuncCUDA_{}.hip".format(n)
        buck_genrule(
            name = "aten_ufunc_hip_renamed_{}".format(n),
            srcs = [gencode_pattern.format(cuda_name)],
            bash = 'cp "$SRCDIR/{}" "$OUT"'.format(cuda_name),
            out = hip_name,
            default_outs = [],
        )
        renamed_rules.append(":aten_ufunc_hip_renamed_{}".format(n))
    return renamed_rules

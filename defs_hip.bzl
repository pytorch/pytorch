load("@bazel_skylib//lib:paths.bzl", "paths")
load("@fbcode//tools/build/buck:rocm_flags.bzl", "get_rocm_arch_args")

caffe2_includes = [
    "operators/**/*",
    "operators/*",
    "sgd/*",
    "transforms/*",
    # distributed folder is managed by its own TARGETS file
    # "distributed/*",
    "queue/*",
    # "binaries/*",
    "**/*_test*",
    "core/*",
    "db/*",
    "utils/**/*",
]

caffe2_video_image_includes = [
    "image/*",
    "video/*",
]

pytorch_includes = [
    "aten/src/ATen/cuda/*",
    "aten/src/ATen/native/cuda/*",
    "aten/src/ATen/native/cuda/linalg/*",
    "aten/src/ATen/native/cudnn/*",
    "aten/src/ATen/native/nested/cuda/*",
    "aten/src/ATen/native/sparse/cuda/*",
    "aten/src/ATen/native/transformers/cuda/*",
    "aten/src/THC/*",
    "aten/src/ATen/test/*",
    "torch/*",
]

gpu_file_extensions = [".cu", ".c", ".cc", ".cpp"]
gpu_header_extensions = [".cuh", ".h", ".hpp"]

hip_external_deps = [
    ("rocm", None, "amdhip64-lazy"),
    ("rocm", None, "MIOpen-lazy"),
    ("rocm", None, "rccl-lazy"),
    ("rocm", None, "roctracer64-lazy"),
]

hip_pp_flags = [
    # HIP 4.4.21432 -> TORCH_HIP_VERSION=404
    "-DTORCH_HIP_VERSION=(FB_HIP_VERSION/100000)",
    # ROCm 4.5.2 -> ROCM_VERSION=40502
    "-DROCM_VERSION=FB_ROCM_VERSION",
    "-DUSE_ROCM=1",
    "-D__HIP_PLATFORM_HCC__=1",
    "-D__HIP_NO_HALF_OPERATORS__=1",
    "-D__HIP_NO_HALF_CONVERSIONS__=1",
    "-DCUDA_HAS_FP16=1",
    "-DCAFFE2_USE_MIOPEN",
    # The c10/cuda/impl/cuda_cmake_macros.h is not generated for the
    # hip build yet.
    "-DC10_HIP_NO_CMAKE_CONFIGURE_FILE",
    # clang with -fopenmp=libgomp (gcc's OpenMP runtime library) produces
    #      single threaded code and doesn't define -D_OPENMP by default.
    # clang with -fopenmp or -fopenmp=libomp (llvm's OpenMP runtime library)
    #      produces multi-threaded code and defines -D_OPENMP by default.
    #
    # hcc currently don't have llvm openmp runtime project builtin.
    # wrap_hip.py also drops -D_OPENMP if explicitly specified.
    "-U_OPENMP",
]

def get_hip_flags():
    return [
        # Caffe2 cannot be compiled with NDEBUG using ROCm 4.5.2.
        # TODO: The issue should be fixed properly.
        "-UNDEBUG",
        "-Wno-error=absolute-value",
        "-Wno-macro-redefined",
        "-Wno-inconsistent-missing-override",
        "-Wno-exceptions",
        "-Wno-shift-count-negative",
        "-Wno-shift-count-overflow",
        "-Wno-duplicate-decl-specifier",
        "-Wno-implicit-int-float-conversion",
        "-Wno-unused-result",
        "-Wno-pass-failed",
        "-Wno-unknown-pragmas",
        "-Wno-cuda-compat",
    ] + get_rocm_arch_args()

def get_hip_file_path(filepath, is_caffe2 = False):
    """
    this function should be in sync with the hipified script in
    third-party/hipify_torch/hipify/hipify_python.py
    unfortunately because it's a normal python (instead of Starlark)
    we cannot simply import from there

    The general rule of converting file names from cuda to hip is:
       - If there is a directory component named "cuda", replace
         it with "hip", AND

       - If the file name contains "CUDA", replace it with "HIP", AND

    If NONE of the above occurred, then insert "hip" in the file path
    as the direct parent folder of the file

    Furthermore, ALWAYS replace '.cu' with '.hip', because those files
    contain CUDA kernels that needs to be hipified and processed with
    hcc compile
    """
    dirpath = paths.dirname(filepath)
    filename = paths.basename(filepath)
    filename, ext = paths.split_extension(filename)

    if ext == ".cu":
        ext = ".hip"

    orig_dirpath = dirpath

    dirpath = dirpath.replace("cuda", "hip")
    dirpath = dirpath.replace("THC", "THH")

    filename = filename.replace("cuda", "hip")
    filename = filename.replace("CUDA", "HIP")

    # Special case to handle caffe2/core/THCCachingAllocator
    if not (is_caffe2 and dirpath == "core"):
        filename = filename.replace("THC", "THH")

    # if the path doesn't change (e.g., path doesn't include "cuda" so we
    # cannot differentiate), insert "hip" as the direct parent folder
    # special case for utils/cub_namespace, because it is first used and hipified when used
    # from core, it doesn't end up in hip directory
    if dirpath == orig_dirpath and not filename == "cub_namespace":
        dirpath = paths.join(dirpath, "hip")

    return paths.join(dirpath, filename + ext)

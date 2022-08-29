load("@bazel_skylib//lib:paths.bzl", "paths")
load(
    "//caffe2:defs_hip.bzl",
    "caffe2_includes",
    "caffe2_video_image_includes",
    "get_hip_file_path",
)

gpu_file_extensions = [".cu", ".c", ".cc", ".cpp"]
gpu_header_extensions = [".cuh", ".h", ".hpp"]

def is_caffe2_gpu_file(filepath):
    # those files are needed since they define placeholders
    if "/native/cudnn/" in filepath:
        return True

    # files that are already compatible with hip
    if "/hip/" in filepath:
        return False

    # exclude all cudnn and nvrtc implementations except for nvrtc_stub
    if "/nvrtc_stub/" in filepath:
        return True
    if any([keyword in filepath for keyword in ("cudnn", "nvrtc", "NVRTC")]):
        return False

    if "/cuda/" in filepath:
        return True

    filename = paths.basename(filepath)
    _, ext = paths.split_extension(filename)

    if "gpu" in filename or ext in [".cu", ".cuh"]:
        return True

    return False

def get_caffe2_hip_srcs(
        include_patterns = caffe2_includes,
        include_files = [],
        project_dir = "caffe2"):
    gpu_file_pattern = [
        base + suffix
        for base in include_patterns
        for suffix in gpu_file_extensions
    ]
    native_gpu_files = native.glob(gpu_file_pattern) + include_files

    # store the original
    gpu_files = []
    hip_files = []
    for name in native_gpu_files:
        # exclude test files
        if "_test" in paths.basename(name) or not is_caffe2_gpu_file(name):
            continue

        gpu_files.append(name)
        hip_file_name = get_hip_file_path(name, is_caffe2 = True)
        hip_files.append(hip_file_name)

    # there will be some native hip files that needs suffix changed
    native_hip_pattern = [
        base[:-1] + "hip/*.hip"
        for base in include_patterns
    ]
    native_hip_files = native.glob(native_hip_pattern)

    gpu_files += native_hip_files
    hip_files += native_hip_files

    # we run hipify script under the caffe2 folder; therefore we need to
    # prepend caffe2 to the path so that buck can find the hipified file
    real_hip_files = []
    for filename in hip_files:
        real_hip_files.append(paths.join(project_dir, filename))

    # return the src and output_gen files
    return gpu_files, real_hip_files

def get_caffe2_hip_headers(
        include_patterns = caffe2_includes,
        include_files = [],
        project_dir = "caffe2"):
    header_pattern = [
        base + suffix
        for base in include_patterns
        for suffix in gpu_header_extensions
    ]
    native_header_files = native.glob(header_pattern) + include_files

    header_files = []
    hip_headers = []
    for name in native_header_files:
        # exclude test files
        # if the caller directly specifies files via include_files, follow it
        if not name in include_files and ("_test" in paths.basename(name) or not is_caffe2_gpu_file(name)):
            continue

        header_files.append(name)
        hip_header_name = get_hip_file_path(name, is_caffe2 = True)
        hip_headers.append(hip_header_name)

    # we run hipify script under the caffe2 folder; therefore we need to
    # prepend caffe2 to the path so that buck can find the hipified file
    real_hip_headers = []
    for filename in hip_headers:
        real_hip_headers.append(paths.join(project_dir, filename))

    # return the src and output_gen files
    return header_files, real_hip_headers

def get_caffe2_hip_video_image_srcs():
    return get_caffe2_hip_srcs(include_patterns = caffe2_video_image_includes)

def get_caffe2_hip_video_image_headers():
    return get_caffe2_hip_headers(include_patterns = caffe2_video_image_includes)

def get_caffe2_hip_test_files():
    test_includes = [
        "**/*_gpu_test.cc",
    ]

    # let's ignores the mpi test and fb-internal tests for now
    test_ignores = [
        "mpi/mpi_gpu_test.cc",
        # "operators/roi_align_op_gpu_test.cc",
        "**/fb/**/*_gpu_test.cc",
    ]

    native_test_files = native.glob(test_includes, exclude = test_ignores)

    test_files = []
    hip_test_files = []
    for name in native_test_files:
        if not is_caffe2_gpu_file(name):
            continue

        test_files.append(name)
        hip_file_name = get_hip_file_path(name, is_caffe2 = True)
        hip_test_files.append(hip_file_name)

    # we run hipify script under the caffe2 folder; therefore we need to
    # prepend caffe2 to the path so that buck can find the hipified file
    real_hip_test_files = []
    for filename in hip_test_files:
        real_hip_test_files.append(paths.join("caffe2", filename))

    # return the src and output_gen files
    return test_files, real_hip_test_files

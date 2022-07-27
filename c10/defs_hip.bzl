load("@bazel_skylib//lib:paths.bzl", "paths")
load("//caffe2:defs_hip.bzl", "get_hip_file_path")

gpu_file_extensions = [".cu", ".c", ".cc", ".cpp"]
gpu_header_extensions = [".cuh", ".h", ".hpp"]

def is_test_files(filepath):
    if filepath.startswith("test"):
        return True
    else:
        return False

def get_c10_hip_srcs():
    gpu_file_pattern = [
        base + suffix
        for base in c10_includes
        for suffix in gpu_file_extensions
    ]
    native_gpu_files = native.glob(gpu_file_pattern)

    gpu_files = []
    hip_files = []
    for name in native_gpu_files:
        # exclude the test folder
        if is_test_files(name):
            continue

        gpu_files.append(name)
        hip_file_name = get_hip_file_path(paths.join("cuda/", name))
        hip_files.append(hip_file_name)

    # there will be some native hip files that needs suffix changed
    native_hip_pattern = [
        "hip/**/*.hip",
    ]
    native_hip_files = native.glob(native_hip_pattern)

    gpu_files += native_hip_files
    hip_files += native_hip_files

    # we run hipify script under the caffe2 folder; therefore we need to
    # prepend c10 to the path so that buck can find the hipified file
    real_hip_files = []
    for filename in hip_files:
        real_hip_files.append(paths.join("c10", filename))

    # return the src and output_gen files
    return gpu_files, real_hip_files

def get_c10_hip_headers():
    gpu_file_pattern = [
        base + suffix
        for base in c10_includes
        for suffix in gpu_header_extensions
    ]
    native_gpu_files = native.glob(gpu_file_pattern)

    # store the original
    gpu_files = []
    hip_files = []
    for name in native_gpu_files:
        if is_test_files(name):
            continue

        gpu_files.append(name)
        hip_file_name = get_hip_file_path(paths.join("cuda/", name))
        hip_files.append(hip_file_name)

    # there will be some native hip files that needs suffix changed
    native_hip_pattern = [
        "hip/**/*" + suffix
        for suffix in gpu_header_extensions
    ]
    native_hip_files = native.glob(native_hip_pattern)

    gpu_files += native_hip_files
    hip_files += native_hip_files

    # we run hipify script under the caffe2 folder; therefore we need to
    # prepend c10 to the path so that buck can find the hipified file
    real_hip_files = []
    for filename in hip_files:
        real_hip_files.append(paths.join("c10", filename))

    # return the src and output_gen files
    return gpu_files, real_hip_files

def get_c10_hip_test_files():
    gpu_file_pattern = [
        base + suffix
        for base in c10_includes
        for suffix in gpu_file_extensions
    ]
    native_gpu_files = native.glob(gpu_file_pattern)

    # store the original
    gpu_files = []
    hip_files = []
    for name in native_gpu_files:
        if not is_test_files(name):
            continue

        gpu_files.append(name)
        hip_file_name = get_hip_file_path(paths.join("cuda/", name))
        hip_files.append(hip_file_name)

    # there will be some native hip files that needs suffix changed
    native_hip_pattern = [
        "hip/test/**/*" + suffix
        for suffix in gpu_header_extensions
    ]
    native_hip_files = native.glob(native_hip_pattern)

    gpu_files += native_hip_files
    hip_files += native_hip_files

    # we run hipify script under the caffe2 folder; therefore we need to
    # prepend c10 to the path so that buck can find the hipified file
    real_hip_files = []
    for filename in hip_files:
        real_hip_files.append(paths.join("c10", filename))

    # return the src and output_gen files
    return gpu_files, real_hip_files

c10_includes = ["**/*"]

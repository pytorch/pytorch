load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_cuda//cuda:defs.bzl", "requires_cuda_enabled")
load("//c10/macros:cmake_configure_file.bzl", "cmake_configure_file")

# Rules implementation for the Bazel build system. Since the common
# build structure aims to replicate Bazel as much as possible, most of
# the rules simply forward to the Bazel definitions.
rules = struct(
    cc_binary = cc_binary,
    cc_library = cc_library,
    cc_test = cc_test,
    cmake_configure_file = cmake_configure_file,
    filegroup = native.filegroup,
    glob = native.glob,
    requires_cuda_enabled = requires_cuda_enabled,
    select = select,
    test_suite = native.test_suite,
)

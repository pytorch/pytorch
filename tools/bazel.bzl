load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_cuda//cuda:defs.bzl", "cuda_library", "requires_cuda_enabled")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")
load("@pytorch//c10/macros:cmake_configure_file.bzl", "cmake_configure_file")
load("@pytorch//tools/config:defs.bzl", "if_cuda")

def _genrule(**kwds):
    if _enabled(**kwds):
        native.genrule(**kwds)

def _is_cpu_static_dispatch_build():
    return False

# Rules implementation for the Bazel build system. Since the common
# build structure aims to replicate Bazel as much as possible, most of
# the rules simply forward to the Bazel definitions.
rules = struct(
    cc_binary = cc_binary,
    cc_library = cc_library,
    cc_test = cc_test,
    cmake_configure_file = cmake_configure_file,
    cuda_library = cuda_library,
    filegroup = native.filegroup,
    genrule = _genrule,
    glob = native.glob,
    if_cuda = if_cuda,
    is_cpu_static_dispatch_build = _is_cpu_static_dispatch_build,
    py_binary = py_binary,
    py_library = py_library,
    requirement = requirement,
    requires_cuda_enabled = requires_cuda_enabled,
    select = select,
    test_suite = native.test_suite,
)

def _enabled(tags = [], **_kwds):
    """Determines if the target is enabled."""
    return "-bazel" not in tags

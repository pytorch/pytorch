load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_cuda//cuda:defs.bzl", "cuda_library", "requires_cuda_enabled")
load("//c10/macros:cmake_configure_file.bzl", "cmake_configure_file")
load("//tools/config:defs.bzl", "if_cuda")

def _genrule(**kwds):
    if _enabled(**kwds):
        native.genrule(**kwds)

def _is_cpu_static_dispatch_build():
    return False

def _py_library(name, **kwds):
    deps = [dep for dep in kwds.pop("deps", []) if dep != None]
    native.py_library(name = name, deps = deps, **kwds)

def _requirement(_pypi_project):
    return None

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
    py_binary = native.py_binary,
    py_library = _py_library,
    requirement = _requirement,
    requires_cuda_enabled = requires_cuda_enabled,
    select = select,
    test_suite = native.test_suite,
)

def _enabled(tags = [], **_kwds):
    """Determines if the target is enabled."""
    return "-bazel" not in tags

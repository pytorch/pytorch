"""
 Macros for selecting with / without various GPU libraries.  Most of these are meant to be used
 directly by tensorflow in place of their build's own configure.py + bazel-gen system.
"""

load("@bazel_skylib//lib:selects.bzl", "selects")

def if_cuda(if_true, if_false = []):
    """Helper for selecting based on the whether CUDA is configured. """
    return selects.with_or({
        "@//tools/config:cuda_enabled_and_capable": if_true,
        "//conditions:default": if_false,
    })

def if_tensorrt(if_true, if_false = []):
    """Helper for selecting based on the whether TensorRT is configured. """
    return select({
        "//conditions:default": if_false,
    })

def if_rocm(if_true, if_false = []):
    """Helper for selecting based on the whether ROCM is configured. """
    return select({
        "//conditions:default": if_false,
    })

def if_sycl(if_true, if_false = []):
    """Helper for selecting based on the whether SYCL/ComputeCPP is configured."""

    # NOTE: Tensorflow expects some stange behavior (see their if_sycl) if we
    # actually plan on supporting this at some point.
    return select({
        "//conditions:default": if_false,
    })

def if_ccpp(if_true, if_false = []):
    """Helper for selecting based on the whether ComputeCPP is configured. """
    return select({
        "//conditions:default": if_false,
    })

def cuda_default_copts():
    return if_cuda(["-DGOOGLE_CUDA=1"])

def cuda_default_features():
    return if_cuda(["-per_object_debug_info", "-use_header_modules", "cuda_clang"])

def rocm_default_copts():
    return if_rocm(["-x", "rocm"])

def rocm_copts(opts = []):
    return rocm_default_copts() + if_rocm(opts)

def cuda_is_configured():
    # FIXME(dcollins): currently only used by tensorflow's xla stuff, which we aren't building.  However bazel
    # query hits it so this needs to be defined.  Because bazel doesn't actually resolve config at macro expansion
    # time, `select` can't be used here (since xla expects lists of strings and not lists of select objects).
    # Instead, the xla build rules must be rewritten to use `if_cuda_is_configured`
    return False

def if_cuda_is_configured(x):
    return if_cuda(x, [])

def if_rocm_is_configured(x):
    return if_rocm(x, [])

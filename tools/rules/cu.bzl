load("@rules_cuda//cuda:defs.bzl", "cuda_library")

NVCC_COPTS = [
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--compiler-options=-Werror=all",
    # The following warnings come from -Wall. We downgrade them from
    # error to warnings here.
    #
    # sign-compare has a tremendous amount of violations in the
    # codebase. It will be a lot of work to fix them, just disable it
    # for now.
    "--compiler-options=-Wno-sign-compare",
    # We intentionally use #pragma unroll, which is compiler specific.
    "--compiler-options=-Wno-error=unknown-pragmas",
    "--compiler-options=-Werror=extra",
    # The following warnings come from -Wextra. We downgrade them from
    # error to warnings here.
    #
    # unused-parameter-compare has a tremendous amount of violations
    # in the codebase. It will be a lot of work to fix them, just
    # disable it for now.
    "--compiler-options=-Wno-unused-parameter",
    # missing-field-parameters has both a large number of violations
    # in the codebase, but it also is used pervasively in the Python C
    # API. There are a couple of catches though:
    # * we use multiple versions of the Python API and hence have
    #   potentially multiple different versions of each relevant
    #   struct. They may have different numbers of fields. It will be
    #   unwieldy to support multiple versions in the same source file.
    # * Python itself for many of these structs recommends only
    #   initializing a subset of the fields. We should respect the API
    #   usage conventions of our dependencies.
    #
    # Hence, we just disable this warning altogether. We may want to
    # clean up some of the clear-cut cases that could be risky, but we
    # still likely want to have this disabled for the most part.
    "-Wno-missing-field-initializers",
]

def cu_library(name, srcs, copts = [], **kwargs):
    cuda_library(name, srcs = srcs, copts = NVCC_COPTS + copts, **kwargs)

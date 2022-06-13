load("@rules_cuda//cuda:defs.bzl", "cuda_library")

NVCC_COPTS = [
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--compiler-options=-Werror=all",
    "--compiler-options=-Werror=type-limits",
    "--compiler-options=-Werror=unused-but-set-variable",

    # The following warnings come from -Wall. We downgrade them from
    # error to warnings here.
    #
    # sign-compare has a tremendous amount of violations in the
    # codebase. It will be a lot of work to fix them, just disable it
    # for now.
    "--compiler-options=-Wno-sign-compare",
    # We intentionally use #pragma unroll, which is compiler specific.
    "--compiler-options=-Wno-error=unknown-pragmas",
]

def cu_library(name, srcs, copts = [], **kwargs):
    cuda_library(name, srcs = srcs, copts = NVCC_COPTS + copts, **kwargs)

load("@rules_cuda//cuda:defs.bzl", "cuda_library")

NVCC_COPTS = [
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--compiler-options=-Werror=type-limits",
    "--compiler-options=-Werror=unused-but-set-variable",
    "--compiler-options=-Werror=unused-function",
    "--compiler-options=-Werror=unused-variable",
]

def cu_library(name, srcs, copts = [], **kwargs):
    cuda_library(name, srcs = srcs, copts = NVCC_COPTS + copts, **kwargs)

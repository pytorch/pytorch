load("@bazel_skylib//lib:paths.bzl", "paths")
load(":build_variables.bzl", "aten_ufunc_headers")

aten_ufunc_names = [
    paths.split_extension(paths.basename(h))[0]
    for h in aten_ufunc_headers
]

def aten_ufunc_generated_cpu_sources(gencode_pattern = "{}"):
    return [gencode_pattern.format(name) for name in [
        "UfuncCPU_{}.cpp".format(n)
        for n in aten_ufunc_names
    ]]

def aten_ufunc_generated_cpu_kernel_sources(gencode_pattern = "{}"):
    return [gencode_pattern.format(name) for name in [
        "UfuncCPUKernel_{}.cpp".format(n)
        for n in aten_ufunc_names
    ]]

def aten_ufunc_generated_cuda_sources(gencode_pattern = "{}"):
    return [gencode_pattern.format(name) for name in [
        "UfuncCUDA_{}.cu".format(n)
        for n in aten_ufunc_names
    ]]

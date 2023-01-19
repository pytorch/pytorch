"""Test code for CUDA rules."""

load("//cuda:defs.bzl", "CudaTargetsInfo")
load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts")

def _cuda_targets_test_impl(ctx):
    env = analysistest.begin(ctx)
    target_under_test = analysistest.target_under_test(env)
    asserts.equals(
        env,
        ["sm_70", "sm_75"],
        target_under_test[CudaTargetsInfo].cuda_targets,
    )
    return analysistest.end(env)

cuda_targets_test = analysistest.make(
    _cuda_targets_test_impl,
    config_settings = {"//cuda:cuda_targets": ["sm_70", "sm_75"]},
)

def _cuda_runtime_test_impl(ctx):
    env = analysistest.begin(ctx)
    target_under_test = analysistest.target_under_test(env)
    asserts.equals(
        env,
        "<merged target //test:cuda_test_runtime>",
        str(target_under_test),
    )
    return analysistest.end(env)

cuda_runtime_test = analysistest.make(
    _cuda_runtime_test_impl,
    config_settings = {"//cuda:cuda_runtime": "//test:cuda_test_runtime"},
)

def _cuda_library_test_impl(ctx):
    env = analysistest.begin(ctx)
    target_under_test = analysistest.target_under_test(env)
    headers = target_under_test[CcInfo].compilation_context.headers.to_list()
    asserts.true(env, "test/cuda.h" in [h.short_path for h in headers])
    return analysistest.end(env)

cuda_library_test = analysistest.make(
    _cuda_library_test_impl,
    config_settings = {"//cuda:cuda_runtime": "//test:cuda_test_runtime"},
)

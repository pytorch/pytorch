#!/usr/bin/env python3

import os
import re
import sys

sys.path.insert(1, os.path.join(sys.path[0], "..", "..", ".."))

from tools.testing.discover_tests import TESTS

skip_list = [
    # these tests fail due to various reasons
    "dynamo/test_functions",
    "dynamo/test_misc",
    "inductor/test_aot_inductor",
    "inductor/test_cpu_repro",
    "inductor/test_cudacodecache",
    "inductor/test_custom_post_grad_passes",
    "inductor/test_inductor_utils",
    "inductor/test_inplacing_pass",
    "inductor/test_kernel_benchmark",
    "inductor/test_layout_optim",
    "inductor/test_max_autotune",
    "inductor/test_mkldnn_pattern_matcher",
    "inductor/test_move_constructors_to_cuda",
    "inductor/test_multi_kernel",
    "inductor/test_pattern_matcher",
    "inductor/test_perf",
    "inductor/test_select_algorithm",
    "inductor/test_snode_runtime",
    "inductor/test_triton_wrapper",
    "lazy/test_meta_kernel",
    "onnx/dynamo/test_dynamo_with_onnxruntime_backend",
    "onnx/test_autograd_funs",
    "onnx/test_custom_ops",
    "onnx/test_models_onnxruntime",
    "onnx/test_models_quantized_onnxruntime",
    "onnx/test_onnxscript_runtime",
    "onnx/test_op_consistency",
    "onnx/test_pytorch_jit_onnx",
    "onnx/test_pytorch_onnx_no_runtime",
    "onnx/test_pytorch_onnx_onnxruntime",
    "onnx/test_utility_funs",
    "onnx/test_verification",
    "profiler/test_profiler",
    "test_ao_sparsity",
    "test_cpp_extensions_open_device_registration",
    "test_jit",
    "test_metal",
    "test_mps",
    # these tests run long and fail in addition to that
    "dynamo/test_dynamic_shapes",
    "test_dataloader",
    "test_linalg",
    "test_nn",
    "test_ops_fwd_gradients",
    "test_ops_gradients",
    "test_proxy_tensor",
    "test_quantization",
    "inductor/test_torchinductor",
    "inductor/test_torchinductor_dynamic_shapes",
    "inductor/test_torchinductor_opinfo",
    "test_binary_ufuncs",
    "test_unary_ufuncs",
    # empty tests:
    "dynamo/test_torchrec",
    "inductor/test_aot_inductor_utils",
    "inductor/test_coordinate_descent_tuner",
    "test_jiterator",
    # https://github.com/pytorch/pytorch/issues/102078
    "test_decomp",
    # https://github.com/pytorch/pytorch/issues/146698
    "test_model_exports_to_core_aten",
    # slow test, skip for now
    "test_fx",
]

skip_list_regex = [
    # distributed tests fail randomly
    "distributed/.*",
]

all_testfiles = sorted(TESTS)

filtered_testfiles = []

for filename in all_testfiles:
    if filename in skip_list:
        continue

    regex_filtered = False

    for regex_string in skip_list_regex:
        if re.fullmatch(regex_string, filename):
            regex_filtered = True
            break

    if regex_filtered:
        continue

    filtered_testfiles.append(filename)

for filename in filtered_testfiles:
    print('    "' + filename + '",')

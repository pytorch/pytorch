#!/usr/bin/env python3

import os
import re
import sys


sys.path.insert(1, os.path.join(sys.path[0], "..", "..", ".."))

from tools.testing.discover_tests import TESTS


skip_list = [
    # these tests fail due to various reasons
    "dynamo/test_misc",
    "inductor/test_aot_inductor",
    "inductor/test_cpu_repro",
    "inductor/test_cpu_select_algorithm",
    "inductor/test_aot_inductor_arrayref",
    "inductor/test_torchinductor_codegen_dynamic_shapes",
    "lazy/test_meta_kernel",
    "onnx/test_utility_funs",
    "profiler/test_profiler",
    "test_ao_sparsity",
    "test_cpp_extensions_open_device_registration",
    "test_jit",
    "test_metal",
    "test_mps",
    "dynamo/test_torchrec",
    "inductor/test_aot_inductor_utils",
    "inductor/test_coordinate_descent_tuner",
    "test_jiterator",
    # these tests run long and fail in addition to that
    "dynamo/test_dynamic_shapes",
    "test_quantization",
    "inductor/test_torchinductor",
    "inductor/test_torchinductor_dynamic_shapes",
    "inductor/test_torchinductor_opinfo",
    "test_binary_ufuncs",
    "test_unary_ufuncs",
    # these tests fail when cuda is not available
    "inductor/test_cudacodecache",
    "inductor/test_inductor_utils",
    "inductor/test_inplacing_pass",
    "inductor/test_kernel_benchmark",
    "inductor/test_max_autotune",
    "inductor/test_move_constructors_to_cuda",
    "inductor/test_multi_kernel",
    "inductor/test_pattern_matcher",
    "inductor/test_perf",
    "inductor/test_select_algorithm",
    "inductor/test_snode_runtime",
    "inductor/test_triton_wrapper",
    # these tests fail when mkldnn is not available
    "inductor/test_custom_post_grad_passes",
    "inductor/test_mkldnn_pattern_matcher",
    # lacks quantization support
    "onnx/test_models_quantized_onnxruntime",
    "onnx/test_pytorch_onnx_onnxruntime",
    # https://github.com/pytorch/pytorch/issues/102078
    "test_decomp",
    # https://github.com/pytorch/pytorch/issues/146698
    "test_model_exports_to_core_aten",
    # runs very long, skip for now
    "inductor/test_layout_optim",
    "test_fx",
    # some false errors
    "doctests",
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

#!/bin/bash

TESTS="
test_import_time
test_public_bindings
test_ao_sparsity
test_autograd
benchmark_utils/test_benchmark_utils
test_binary_ufuncs
test_buffer_protocol
test_bundled_inputs
test_complex
test_cpp_api_parity
test_cpp_extensions_aot_no_ninja
test_cpp_extensions_aot_ninja
test_cpp_extensions_jit
test_cuda
test_jit_cuda_fuser
test_cuda_primary_ctx
test_dataloader
test_datapipe
distributions/test_constraints
distributions/test_distributions
test_dispatch
test_foreach
test_indexing
test_jit
test_linalg
test_logging
test_mkldnn
test_model_dump
test_module_init
test_modules
test_multiprocessing_spawn
test_native_functions
test_numba_integration
test_nn
test_ops
test_optim
test_functional_optim
test_pytree
test_mobile_optimizer
test_set_default_mobile_cpu_allocator
test_xnnpack_integration
test_vulkan
test_sparse
test_sparse_csr
test_quantization
test_pruning_op
test_spectral_ops
test_serialization
test_shape_ops
test_show_pickle
test_sort_and_select
test_tensor_creation_ops
test_testing
test_torch
test_type_info
test_unary_ufuncs
test_utils
test_view_ops
test_vmap
test_namedtuple_return_api
test_numpy_interop
test_jit_profiling
test_jit_fuser_legacy
test_tensorboard
test_namedtensor
test_reductions
test_type_promotion
test_jit_disabled
test_function_schema
test_overrides
test_jit_fuser_te
test_tensorexpr
test_tensorexpr_pybind
test_profiler
test_futures
test_fx
test_fx_experimental
test_functional_autograd_benchmark
test_package
test_license
"

export PYTORCH_TEST_WITH_ROCM=1

for test in $TESTS
do
   sanitized=`echo $test | tr / _`
   echo $test 
   echo $sanitized
   pushd /tmp/pytorch/test
   python run_test.py --verbose -i $test |& tee ../../${sanitized}.log
   popd
done

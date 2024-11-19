TODO:
1. Talked to James about logging. Need to think about how we'll handle subproc
   logging. Maybe just log it and refer to the subproc log in the main process.

2. Do we need to pass back the kernel files?

3. PyREClientV2?

TO TEST:

TORCHINDUCTOR_FX_GRAPH_ASYNC_COMPILE=1 python test/inductor/test_codecache.py TestFxGraphCache.test_remote_cache_load_function_device_cpu_bfloat16_dynamic_False

CUDA problems:
  python test/inductor/test_triton_kernels.py -k test_triton_kernel_native_grad_True_dynamic_True_backend_inductor
(py39) $ /data/users/aorenste/miniconda3/envs/py39/lib/python3.9/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 5 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '

TORCHINDUCTOR_FX_GRAPH_ASYNC_COMPILE=1 pytest -n 10 test/inductor/test_torchinductor.py

Serialized:
Individual test sequence:
TORCHINDUCTOR_FX_GRAPH_ASYNC_COMPILE=1 python test/inductor/test_torchinductor.py CpuTests.test_bool_cpu
TORCHINDUCTOR_FX_GRAPH_ASYNC_COMPILE=1 python test/inductor/test_torchinductor.py GPUTests.test_config_option_dont_assume_alignment_cuda
TORCHINDUCTOR_FX_GRAPH_ASYNC_COMPILE=1 python test/inductor/test_torchinductor.py CpuTests.test_dtype_sympy_expr_cpu

==================================================================================== 743 failed, 841 passed, 32 skipped in 459.67s (0:07:39) ====================================================================================
==================================================================================== 631 failed, 953 passed, 32 skipped in 459.19s (0:07:39) ====================================================================================
=================================================================================== 519 failed, 1060 passed, 37 skipped in 450.78s (0:07:30) ====================================================================================
=================================================================================== 484 failed, 1094 passed, 38 skipped in 444.73s (0:07:24) ====================================================================================
=================================================================================== 442 failed, 1132 passed, 42 skipped in 443.09s (0:07:23) ====================================================================================
============================================================================== 21 failed, 1545 passed, 50 skipped in 603.78s (0:10:03) ==============================================================================

Subprocess:
Individual test sequence:
TORCHINDUCTOR_FX_GRAPH_ASYNC_COMPILE=0 python test/inductor/test_torchinductor.py CpuTests.test_index_propagation_floordiv_cpu

============================================================================================= 687 failed, 879 passed, 50 skipped in 1008.62s (0:16:48) =============================================================================================
============================================================================================= 661 failed, 905 passed, 50 skipped in 1009.78s (0:16:49) =============================================================================================
============================================================================================= 705 failed, 861 passed, 50 skipped in 988.51s (0:16:28) ==============================================================================================
============================================================================================= 727 failed, 839 passed, 50 skipped in 944.10s (0:15:44) ==============================================================================================
============================================================================================= 630 (132) failed, 936 (1426) passed, 50 skipped in 950.94s (0:15:50) ==============================================================================================
============================================================================================= 793 (86) failed, 724 passed, 105 skipped in 943.38s (0:15:43) =============================================================================================

NORMAL FAILURES:
==================================================================================================== short test summary info ====================================================================================================
FAILED [3.7005s] test/inductor/test_torchinductor.py::CpuTests::test_custom_op_fixed_layout_sequential_cpu - AssertionError: Tensor-likes are not close!
FAILED [1.8661s] test/inductor/test_torchinductor.py::CpuTests::test_inductor_layout_optimization_input_mutations_cpu - AssertionError: Tensor-likes are not close!
FAILED [4.7936s] test/inductor/test_torchinductor.py::CpuTests::test_mutable_custom_op_fixed_layout2_cpu - AssertionError: Tensor-likes are not close!
FAILED [2.5034s] test/inductor/test_torchinductor.py::GPUTests::test_custom_op_fixed_layout_sequential_cuda - AssertionError: Tensor-likes are not close!
FAILED [0.8294s] test/inductor/test_torchinductor.py::GPUTests::test_inductor_layout_optimization_input_mutations_cuda - AssertionError: Tensor-likes are not close!
FAILED [1.4187s] test/inductor/test_torchinductor.py::GPUTests::test_mutable_custom_op_fixed_layout2_cuda - AssertionError: Tensor-likes are not close!
FAILED [0.0736s] test/inductor/test_torchinductor.py::TritonCodeGenTests::test_dtype_aware_codegen_load_upcast_to_fp32_False_float16 - torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
FAILED [0.0577s] test/inductor/test_torchinductor.py::TritonCodeGenTests::test_dtype_aware_codegen_load_upcast_to_fp32_False_bfloat16 - torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
==================================================================================== 8 failed, 1558 passed, 50 skipped in 604.38s (0:10:04) =====================================================================================

============================================================================================================= short test summary info ==============================================================================================================
FAILED [13.4172s] test/inductor/test_torchinductor.py::CpuTests::test_resize_as_cpu - AssertionError: False is not true
FAILED [14.0829s] test/inductor/test_torchinductor.py::CpuTests::test_resize_cpu - AssertionError: False is not true
FAILED [0.1482s] test/inductor/test_torchinductor.py::CpuTests::test_split_with_sizes_with_unbacked_symints_cpu - torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
FAILED [10.1332s] test/inductor/test_torchinductor.py::GPUTests::test_no_mega_fusion_during_lowering_cuda - AssertionError: False is not true
FAILED [7.3759s] test/inductor/test_torchinductor.py::GPUTests::test_resize_as_cuda - AssertionError: False is not true
FAILED [7.6221s] test/inductor/test_torchinductor.py::GPUTests::test_resize_cuda - AssertionError: False is not true
FAILED [0.1237s] test/inductor/test_torchinductor.py::GPUTests::test_split_with_sizes_with_unbacked_symints_cuda - torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
FAILED [7.8272s] test/inductor/test_torchinductor.py::GPUTests::test_transpose_add_cuda - AssertionError: TODO: Bad async compile (disable this test)
FAILED [0.3358s] test/inductor/test_torchinductor.py::TritonCodeGenTests::test_numpy_autograd - torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
====================================================================================================== 9 failed, 2 passed in 86.85s (0:01:26) ======================================================================================================

CI FAILURES:
test/distributed/test_c10d_functional_native.py::CompileTest::test_inductor_all_gather_into_tensor_coalesced
test/distributed/test_c10d_functional_native.py::CompileTest::test_inductor_all_gather_into_tensor_coalesced
test/distributed/test_dynamo_distributed.py::TestMultiProc::test_asymmetric_compilation_with_fx_cache
test/dynamo/test_dynamic_shapes.py::DynamicShapesFunctionTests::test_filter_fallback_dynamic_shapes
test/dynamo/test_dynamic_shapes.py::DynamicShapesFunctionTests::test_filter_fallback_dynamic_shapes
test/functorch/test_eager_transforms.py::TestCompileTransformsCPU::test_compile_vmap_hessian_cpu
test/inductor/test_benchmark_fusion.py::BenchmarkMultiTemplateFusionCudaTest::test_equivalent_template_code
test/inductor/test_cpu_repro.py::CPUReproTests::test_channel_shuffle_cl_output
test/inductor/test_cpu_repro.py::CPUReproTests::test_channel_shuffle_cl_output
test/inductor/test_cpu_repro.py::CPUReproTests::test_channel_shuffle_cl_output
test/inductor/test_cpu_repro.py::CPUReproTests::test_channel_shuffle_cl_output
test/inductor/test_cpu_repro.py::CPUReproTests::test_embedding_vec_bf16
test/inductor/test_cpu_repro.py::CPUReproTests::test_expr_vec_non_contiguous
test/inductor/test_cpu_repro.py::CPUReproTests::test_group_norm_vec
test/inductor/test_cpu_repro.py::CPUReproTests::test_ir_node_str
test/inductor/test_cpu_repro.py::CPUReproTests::test_local_buffer_in_outer_loop_fusion
test/inductor/test_cpu_repro.py::CPUReproTests::test_local_buffer_in_outer_loop_fusion
test/inductor/test_cpu_repro.py::CPUReproTests::test_local_buffer_in_outer_loop_fusion
test/inductor/test_cpu_repro.py::CPUReproTests::test_lstm_packed_unbatched_False_input_size_1_hidden_size_2_num_layers_1_bidirectional_False_bias_False_empty_state_False_batch_first_False_batch_size_2_seq_len_1
test/inductor/test_cpu_repro.py::CPUReproTests::test_lstm_packed_unbatched_False_input_size_1_hidden_size_2_num_layers_1_bidirectional_False_bias_False_empty_state_False_batch_first_True_batch_size_2_seq_len_1
test/inductor/test_max_autotune.py::TestMaxAutotune::test_cat_max_autotune_triton
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test__unsafe_masked_index_put_accumulate_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test__unsafe_masked_index_put_accumulate_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test__unsafe_masked_index_put_accumulate_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test__unsafe_masked_index_put_accumulate_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_adaptive_avg_pool1d_argmax_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_add_const_int_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_add_const_int_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_add_const_int_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_add_const_int_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_add_const_int_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_angle_dynamic_shapes_cpu

lennard_jones
mixer_b16_224
speech_transformer
vision_maskrcnn
vision_maskrcnn
vit_base_patch16_224
##[error]Process completed with exit code 1.
Process completed with exit code 255.
detectron2_fcos_r_50_fpn
detectron2_maskrcnn_r_50_fpn

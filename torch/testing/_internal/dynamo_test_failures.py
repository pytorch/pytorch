# tests in this list will run without Dynamo strict mode by default.
FIXME_default_non_strict = {
    "dynamo/test_logging",
    "export/test_db",
    "export/test_experimental",
    "export/test_export",
    "export/test_export_nonstrict",
    "export/test_functionalized_assertions",
    "export/test_pass_infra",
    "export/test_passes",
    "export/test_retraceability",
    "export/test_serdes",
    "export/test_serialize",
    "export/test_unflatten",
    "export/test_upgrade",
    "export/test_verifier",
    "functorch/test_aotdispatch",
    "functorch/test_ops",
    "functorch/test_vmap",
    "functorch/test_vmap_registrations",
    "inductor/test_aot_inductor",
    "inductor/test_aot_inductor_utils",
    "inductor/test_benchmark_fusion",
    "inductor/test_binary_folding",
    "inductor/test_codecache",
    "inductor/test_codegen_triton",
    "inductor/test_compiled_autograd",
    "inductor/test_compiled_optimizers",
    "inductor/test_config",
    "inductor/test_coordinate_descent_tuner",
    "inductor/test_cpu_cpp_wrapper",
    "inductor/test_cpu_repro",
    "inductor/test_cuda_cpp_wrapper",
    "inductor/test_cuda_repro",
    "inductor/test_cudacodecache",
    "inductor/test_cudagraph_trees",
    "inductor/test_custom_lowering",
    "inductor/test_custom_post_grad_passes",
    "inductor/test_debug_trace",
    "inductor/test_dependencies",
    "inductor/test_efficient_conv_bn_eval",
    "inductor/test_extension_backend",
    "inductor/test_foreach",
    "inductor/test_fp8",
    "inductor/test_fused_attention",
    "inductor/test_fx_fusion",
    "inductor/test_group_batch_fusion",
    "inductor/test_indexing",
    "inductor/test_inductor_freezing",
    "inductor/test_inductor_utils",
    "inductor/test_inplacing_pass",
    "inductor/test_kernel_benchmark",
    "inductor/test_layout_optim",
    "inductor/test_max_autotune",
    "inductor/test_memory_planning",
    "inductor/test_minifier",
    "inductor/test_minifier_isolate",
    "inductor/test_mkldnn_pattern_matcher",
    "inductor/test_mmdecomp",
    "inductor/test_move_constructors_to_cuda",
    "inductor/test_pattern_matcher",
    "inductor/test_perf",
    "inductor/test_profiler",
    "inductor/test_select_algorithm",
    "inductor/test_smoke",
    "inductor/test_snode_runtime",
    "inductor/test_split_cat_fx_passes",
    "inductor/test_standalone_compile",
    "inductor/test_torchinductor",
    "inductor/test_torchinductor_codegen_dynamic_shapes",
    "inductor/test_torchinductor_dynamic_shapes",
    "inductor/test_torchinductor_opinfo",
    "inductor/test_triton_heuristics",
    "inductor/test_triton_wrapper",
    "inductor/test_unbacked_symints",
    "lazy/test_ts_opinfo",
    "profiler/test_memory_profiler",
    "profiler/test_profiler",
    "profiler/test_profiler_tree",
    "test_schema_check",  # nb: times out
    "test_ao_sparsity",
    "test_autograd",
    "test_content_store",
    "test_custom_ops",
    "test_dataloader",
    "test_fx",
    "test_jit",
    "test_jit_fuser_te",
    "test_jit_llga_fuser",
    "test_modules",
    "test_namedtensor",
    "test_ops",
    "test_ops_fwd_gradients",
    "test_ops_gradients",
    "test_ops_jit",
    "test_legacy_vmap",
    "test_package",
    "test_public_bindings",
    "test_python_dispatch",
    "test_quantization",
    "test_tensorexpr",
    "test_tensorexpr_pybind",
    "test_torch",
    "test_vulkan",
    "test_xnnpack_integration",
}

# We generate unittest.expectedFailure for all of the following tests
# when run under PYTORCH_TEST_WITH_DYNAMO=1.
#
# This lists exists so we can more easily add large numbers of failing tests,
dynamo_expected_failures = {
    "TestCppExtensionJIT.test_cpp_frontend_module_has_up_to_date_attribute",
    "TestCppExtensionJIT.test_custom_compound_op_autograd",
    "TestCppExtensionJIT.test_cpp_frontend_module_has_up_to_date_attributes",
    "TestCppExtensionOpenRgistration.test_open_device_registration",
    "TestAutogradFallback.test_supports_tensor_lists_mode_nothing",
    "TestAutogradFallback.test_post_autograd_returns_mix_of_requires_grad_tensors_mode_warn",
    "TestAutogradFallback.test_cpu_return_self_mode_warn",
    "TestAutogradFallback.test_base_does_not_require_grad_mode_warn",
    "TestAutogradFallback.test_undefined_grads_mode_nothing",
    "TestAutogradFallback.test_undefined_grads_mode_warn",
    "TestAutogradFallback.test_autograd_function_registered_to_cpu_mode_warn",
    "TestAutogradFallback.test_cpu_return_self_mode_nothing",
    "TestAutogradFallback.test_composite_registered_to_cpu_mode_nothing",
    "TestAutogradFallback.test_undefined_inputs_outputs_mode_nothing",
    "TestAutogradFallback.test_no_autograd_kernel_inplace_mode_nothing",
    "TestAutogradFallback.test_post_autograd_returns_leaf_mode_nothing",
    "TestAutogradFallback.test_inplace_on_tensor_that_does_not_require_grad_mode_nothing",
    "TestAutogradFallback.test_no_grad_mode_warn",
    "TestAutogradFallback.test_inplace_autograd_function_registered_to_cpu_mode_warn",
    "TestAutogradFallback.test_no_autograd_kernel_mode_warn",
    "TestAutogradFallback.test_base_does_not_require_grad_mode_nothing",
    "TestAutogradFallback.test_composite_registered_to_cpu_mode_warn",
    "TestAutogradFallback.test_post_autograd_returns_mix_of_requires_grad_tensors_mode_nothing",
    "TestAutogradFallback.test_no_autograd_kernel_inplace_mode_warn",
    "TestAutogradFallback.test_no_grad_mode_nothing",
    "TestAutogradFallback.test_no_autograd_kernel_mode_nothing",
    "TestAutogradFallback.test_supports_tensor_lists_mode_warn",
    "TestAutogradFallback.test_post_autograd_returns_leaf_mode_warn",
    "TestAutogradFallback.test_undefined_inputs_outputs_mode_warn",
    "TestAutogradFallback.test_inplace_on_tensor_that_does_not_require_grad_mode_warn",
    "TestAutogradFallback.test_inplace_autograd_function_registered_to_cpu_mode_nothing",
    "TestAutogradFallback.test_autograd_function_registered_to_cpu_mode_nothing",
    "TestFunctionalOptimParity.test_functional_optim_parity_sgd",
    "TestIndexingCPU.test_invalid_index_cpu",
    "NumpyTestsCPU.test_boolean_shape_mismatch_cpu",
    "TestIndexingCPU.test_empty_ndim_index_bool_cpu",
    "TestIndexingCPU.test_out_of_bound_index_cpu",
    "NumpyTestsCPU.test_index_no_floats_cpu",
    "TestIndexingCPU.test_zero_dim_index_cpu",
    "NumpyTestsCPU.test_empty_fancy_index_cpu",
    "TestIndexingCPU.test_index_cpu",
    "TestIndexingCPU.test_index_limits_cpu",
    "NumpyTestsCPU.test_boolean_indexing_weirdness_cpu",
    "TestLinalgCPU.test_inverse_cpu_float32",
    "TestLinalgCPU.test_matrix_rank_cpu_complex64",
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_float32",
    "TestLinalgCPU.test_inverse_cpu_complex128",
    "TestLinalgCPU.test_norm_dtype_cpu_complex128",
    "TestLinalgCPU.test_householder_product_cpu_float64",
    "TestLinalgCPU.test_linalg_lu_family_cpu_float32",
    "TestLinalgCPU.test_linalg_lu_family_cpu_float64",
    "TestLinalgCPU.test_addr_integral_cpu_int64",
    "TestLinalgCPU.test_norm_vector_cpu_float32",
    "TestLinalgCPU.test_solve_cpu_complex128",
    "TestLinalgCPU.test_lobpcg_torchscript_cpu_float64",
    "TestLinalgCPU.test_einsum_sublist_format_cpu_float64",
    "TestLinalgCPU.test_solve_cpu_float32",
    "TestLinalgCPU.test_addr_integral_cpu_int16",
    "TestLinalgCPU.test_norm_vector_cpu_float64",
    "TestLinalgCPU.test_einsum_random_cpu_complex128",
    "TestLinalgCPU.test_addmm_sizes_cpu_float64",
    "TestLinalgCPU.test_norm_dtype_cpu_float64",
    "TestLinalgCPU.test_addr_integral_cpu_int8",
    "TestLinalgCPU.test_einsum_random_cpu_float64",
    "TestLinalgCPU.test_matmul_small_brute_force_3d_Nd_cpu_complex64",
    "TestLinalgCPU.test_matrix_rank_cpu_float32",
    "TestLinalgCPU.test_pinv_cpu_float32",
    "TestLinalgCPU.test_addr_integral_cpu_uint8",
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_complex128",
    "TestLinalgCPU.test_addr_integral_cpu_int32",
    "TestLinalgCPU.test_matmul_small_brute_force_3d_Nd_cpu_int64",
    "TestLinalgCPU.test_solve_cpu_complex64",
    "TestLinalgCPU.test_solve_cpu_float64",
    "TestLinalgCPU.test_addmm_sizes_cpu_float32",
    "TestLinalgCPU.test_norm_bfloat16_and_half_cpu_float16",
    "TestLinalgCPU.test_householder_product_cpu_complex64",
    "TestLinalgCPU.test_linalg_lu_family_cpu_complex128",
    "TestLinalgCPU.test_inverse_cpu_float64",
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_complex64",
    "TestLinalgCPU.test_pinv_cpu_complex64",
    "TestLinalgCPU.test_matmul_small_brute_force_3d_Nd_cpu_float32",
    "TestLinalgCPU.test_geqrf_cpu_complex128",
    "TestLinalgCPU.test_matrix_rank_cpu_complex128",
    "TestLinalgCPU.test_einsum_sublist_format_cpu_complex128",
    "TestLinalgCPU.test_geqrf_cpu_complex64",
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_float64",
    "TestLinalgCPU.test_linalg_lu_family_cpu_complex64",
    "TestLinalgCPU.test_matrix_rank_cpu_float64",
    "TestLinalgCPU.test_geqrf_cpu_float64",
    "TestLinalgCPU.test_householder_product_cpu_complex128",
    "TestLinalgCPU.test_geqrf_cpu_float32",
    "TestLinalgCPU.test_pinv_cpu_complex128",
    "TestLinalgCPU.test_pinv_cpu_float64",
    "TestLinalgCPU.test_householder_product_cpu_float32",
    "TestLinalgCPU.test_norm_bfloat16_and_half_cpu_bfloat16",
    "TestLinalgCPU.test_inverse_cpu_complex64",
    "TestModuleInitCPU.test_nn_FractionalMaxPool3d_cpu_float64",
    "TestModuleInitCPU.test_nn_PReLU_cpu_float64",
    "TestModuleInitCPU.test_nn_MultiLabelSoftMarginLoss_cpu_float64",
    "TestModuleInitCPU.test_nn_TransformerEncoder_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyLinear_cpu_float32",
    "TestModuleInitCPU.test_nn_BatchNorm3d_cpu_float32",
    "TestModuleInitCPU.test_nn_BCEWithLogitsLoss_cpu_float64",
    "TestModuleInitCPU.test_nn_BatchNorm1d_cpu_float32",
    "TestModuleInitCPU.test_quantizable_LSTMCell_cpu_float32",
    "TestModuleInitCPU.test_nn_InstanceNorm2d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConvTranspose1d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyLinear_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConv2d_cpu_float64",
    "TestModuleInitCPU.test_nn_PReLU_cpu_float32",
    "TestModuleInitCPU.test_nn_InstanceNorm1d_cpu_float64",
    "TestModuleInitCPU.test_nn_InstanceNorm2d_cpu_float32",
    "TestModuleInitCPU.test_nn_ConvTranspose1d_cpu_float32",
    "TestModuleInitCPU.test_quantized_InstanceNorm1d_cpu_float64",
    "TestModuleInitCPU.test_nn_TransformerEncoderLayer_cpu_float64",
    "TestModuleInitCPU.test_qat_Conv3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyConvTranspose3d_cpu_float32",
    "TestModuleInitCPU.test_quantized_LeakyReLU_cpu_float32",
    "TestModuleInitCPU.test_quantized_GroupNorm_cpu_float64",
    "TestModuleInitCPU.test_nn_RNNBase_cpu_float32",
    "TestModuleInitCPU.test_nn_FractionalMaxPool2d_cpu_float64",
    "TestModuleInitCPU.test_nn_LSTMCell_cpu_float64",
    "TestModuleInitCPU.test_nn_Embedding_cpu_float32",
    "TestModuleInitCPU.test_quantized_BatchNorm2d_cpu_float64",
    "TestModuleInitCPU.test_nn_RNNCellBase_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConvTranspose3d_cpu_float64",
    "TestModuleInitCPU.test_quantized_GroupNorm_cpu_float32",
    "TestModuleInitCPU.test_nn_MultiLabelSoftMarginLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_GroupNorm_cpu_float32",
    "TestModuleInitCPU.test_nn_RNNCell_cpu_float64",
    "TestModuleInitCPU.test_nn_TransformerEncoder_cpu_float32",
    "TestModuleInitCPU.test_nn_InstanceNorm3d_cpu_float64",
    "TestModuleInitCPU.test_quantized_InstanceNorm2d_cpu_float32",
    "TestModuleInitCPU.test_nn_Conv3d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConv2d_cpu_float32",
    "TestModuleInitCPU.test_nn_RNNCellBase_cpu_float32",
    "TestModuleInitCPU.test_quantized_Quantize_cpu_float32",
    "TestModuleInitCPU.test_nn_MultiheadAttention_cpu_float32",
    "TestModuleInitCPU.test_nn_TransformerEncoderLayer_cpu_float32",
    "TestModuleInitCPU.test_quantized_BatchNorm3d_cpu_float64",
    "TestModuleInitCPU.test_nn_ConvTranspose3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm1d_cpu_float32",
    "TestModuleInitCPU.test_nn_RNNBase_cpu_float64",
    "TestModuleInitCPU.test_nn_ConvTranspose2d_cpu_float64",
    "TestModuleInitCPU.test_nn_AdaptiveLogSoftmaxWithLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_Transformer_cpu_float64",
    "TestModuleInitCPU.test_quantizable_LSTM_cpu_float64",
    "TestModuleInitCPU.test_nn_BCEWithLogitsLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyConv1d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConv3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyBatchNorm2d_cpu_float64",
    "TestModuleInitCPU.test_nn_Embedding_cpu_float64",
    "TestModuleInitCPU.test_nn_FractionalMaxPool3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyBatchNorm3d_cpu_float32",
    "TestModuleInitCPU.test_nn_GroupNorm_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConv3d_cpu_float64",
    "TestModuleInitCPU.test_nn_GRU_cpu_float32",
    "TestModuleInitCPU.test_qat_Conv3d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm1d_cpu_float64",
    "TestModuleInitCPU.test_nn_TransformerDecoder_cpu_float64",
    "TestModuleInitCPU.test_nn_Conv3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyBatchNorm2d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm2d_cpu_float32",
    "TestModuleInitCPU.test_qat_Embedding_cpu_float32",
    "TestModuleInitCPU.test_nn_GRU_cpu_float64",
    "TestModuleInitCPU.test_quantized_LayerNorm_cpu_float32",
    "TestModuleInitCPU.test_quantizable_MultiheadAttention_cpu_float64",
    "TestModuleInitCPU.test_qat_Embedding_cpu_float64",
    "TestModuleInitCPU.test_nn_SyncBatchNorm_cpu_float32",
    "TestModuleInitCPU.test_nn_Transformer_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyBatchNorm3d_cpu_float64",
    "TestModuleInitCPU.test_nn_FractionalMaxPool2d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm2d_cpu_float64",
    "TestModuleInitCPU.test_qat_Conv2d_cpu_float32",
    "TestModuleInitCPU.test_nn_BatchNorm2d_cpu_float32",
    "TestModuleInitCPU.test_nn_BatchNorm1d_cpu_float64",
    "TestModuleInitCPU.test_nn_Bilinear_cpu_float32",
    "TestModuleInitCPU.test_nn_Conv2d_cpu_float64",
    "TestModuleInitCPU.test_qat_EmbeddingBag_cpu_float32",
    "TestModuleInitCPU.test_quantized_InstanceNorm1d_cpu_float32",
    "TestModuleInitCPU.test_quantizable_LSTMCell_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyBatchNorm1d_cpu_float64",
    "TestModuleInitCPU.test_nn_NLLLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyConv1d_cpu_float32",
    "TestModuleInitCPU.test_quantizable_MultiheadAttention_cpu_float32",
    "TestModuleInitCPU.test_nn_BCELoss_cpu_float64",
    "TestModuleInitCPU.test_nn_TransformerDecoderLayer_cpu_float32",
    "TestModuleInitCPU.test_nn_LayerNorm_cpu_float32",
    "TestModuleInitCPU.test_nn_AdaptiveLogSoftmaxWithLoss_cpu_float64",
    "TestModuleInitCPU.test_nn_CrossEntropyLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_LayerNorm_cpu_float64",
    "TestModuleInitCPU.test_nn_RNNCell_cpu_float32",
    "TestModuleInitCPU.test_nn_ConvTranspose1d_cpu_float64",
    "TestModuleInitCPU.test_nn_GRUCell_cpu_float64",
    "TestModuleInitCPU.test_nn_LSTMCell_cpu_float32",
    "TestModuleInitCPU.test_qat_Linear_cpu_float32",
    "TestModuleInitCPU.test_nn_Conv2d_cpu_float32",
    "TestModuleInitCPU.test_nn_InstanceNorm1d_cpu_float32",
    "TestModuleInitCPU.test_nn_TransformerDecoderLayer_cpu_float64",
    "TestModuleInitCPU.test_quantized_InstanceNorm3d_cpu_float64",
    "TestModuleInitCPU.test_nn_SyncBatchNorm_cpu_float64",
    "TestModuleInitCPU.test_nn_RNN_cpu_float32",
    "TestModuleInitCPU.test_nn_RNN_cpu_float64",
    "TestModuleInitCPU.test_quantizable_LSTM_cpu_float32",
    "TestModuleInitCPU.test_quantized_InstanceNorm3d_cpu_float32",
    "TestModuleInitCPU.test_quantized_Hardswish_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyBatchNorm1d_cpu_float32",
    "TestModuleInitCPU.test_quantized_InstanceNorm2d_cpu_float64",
    "TestModuleInitCPU.test_qat_EmbeddingBag_cpu_float64",
    "TestModuleInitCPU.test_quantized_BatchNorm2d_cpu_float32",
    "TestModuleInitCPU.test_nn_CrossEntropyLoss_cpu_float64",
    "TestModuleInitCPU.test_nn_ConvTranspose3d_cpu_float64",
    "TestModuleInitCPU.test_quantized_Quantize_cpu_float64",
    "TestModuleInitCPU.test_nn_BCELoss_cpu_float32",
    "TestModuleInitCPU.test_nn_EmbeddingBag_cpu_float32",
    "TestModuleInitCPU.test_nn_LSTM_cpu_float64",
    "TestModuleInitCPU.test_nn_Linear_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm3d_cpu_float64",
    "TestModuleInitCPU.test_nn_EmbeddingBag_cpu_float64",
    "TestModuleInitCPU.test_nn_ConvTranspose2d_cpu_float32",
    "TestModuleInitCPU.test_nn_BatchNorm2d_cpu_float64",
    "TestModuleInitCPU.test_nn_BatchNorm3d_cpu_float64",
    "TestModuleInitCPU.test_nn_MultiMarginLoss_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyInstanceNorm3d_cpu_float32",
    "TestModuleInitCPU.test_nn_MultiMarginLoss_cpu_float64",
    "TestModuleInitCPU.test_quantized_LayerNorm_cpu_float64",
    "TestModuleInitCPU.test_nn_InstanceNorm3d_cpu_float32",
    "TestModuleInitCPU.test_nn_Bilinear_cpu_float64",
    "TestModuleInitCPU.test_qat_Conv1d_cpu_float64",
    "TestModuleInitCPU.test_nn_Conv1d_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConvTranspose2d_cpu_float32",
    "TestModuleInitCPU.test_nn_LazyConvTranspose2d_cpu_float64",
    "TestModuleInitCPU.test_nn_MultiheadAttention_cpu_float64",
    "TestModuleInitCPU.test_nn_GRUCell_cpu_float32",
    "TestModuleInitCPU.test_quantized_LeakyReLU_cpu_float64",
    "TestModuleInitCPU.test_qat_Conv2d_cpu_float64",
    "TestModuleInitCPU.test_nn_NLLLoss_cpu_float64",
    "TestModuleInitCPU.test_quantized_Hardswish_cpu_float32",
    "TestModuleInitCPU.test_nn_Linear_cpu_float64",
    "TestModuleInitCPU.test_nn_LazyConvTranspose1d_cpu_float32",
    "TestModuleInitCPU.test_nn_Conv1d_cpu_float32",
    "TestModuleInitCPU.test_nn_TransformerDecoder_cpu_float32",
    "TestModuleInitCPU.test_qat_Linear_cpu_float64",
    "TestModuleInitCPU.test_quantized_BatchNorm3d_cpu_float32",
    "TestModuleInitCPU.test_nn_LSTM_cpu_float32",
    "TestModuleInitCPU.test_qat_Conv1d_cpu_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc10_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc9_out_dtype_float64",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_same_kind_ufunc0_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc8_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc1_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc12_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc4_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc16_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc6_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc11_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc1_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc16",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc14_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc8_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc1_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc5",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc14_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc2_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc15_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc12_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc11_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc16_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc4_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc15_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc3_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc1_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc12_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc12_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc12_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc0_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc16_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc13_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc4_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc9_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc12_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc11_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc15_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc8_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc10_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc13_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc8_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc9_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc7_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc8_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc0_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc6_out_dtype_float32",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_unsafe_ufunc0_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc7",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_equiv_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc9_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc0_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc4",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc14_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc13_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc9_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc7_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc6_out_dtype_float64",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_safe_ufunc0_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc5_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc4_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc9_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc8_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc7_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc14_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc10_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc9_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc4_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc8_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc8_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc0",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc7_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc6_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc15_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc12_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc11_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc1_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc0_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc2_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc15",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc16_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc12_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc16_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc1_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc2_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc3_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc6_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc3",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc5_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc3_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc7_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc6_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc5_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc7_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc2_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc10_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc10",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc16_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc5_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc2_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc5_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc3_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc7_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc15_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc5_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc15_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc11_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc14_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc2_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc2_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc11_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc12_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc8_out_dtype_float64",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_same_kind_ufunc0_out_dtype_float64",
    "TestUfuncDtypeKwd.test_binary_ufunc_dtype_and_out",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_same_kind_ufunc0_out_dtype_complex128",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_unsafe_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc1_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc12_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc13_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc5_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc14_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc9_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc14_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc13_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc10_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc3_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc11",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc7_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc13_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc6_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc1",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc0_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc9_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc16_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc16_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc5_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc12",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc3_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc6_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc14_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc15_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc13_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc7_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc14_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc4_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc6_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc4_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc11_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc3_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc9",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc10_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc3_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc11_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc15_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc14",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc15_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc6",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc16_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc5_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc9_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc16_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc4_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc5_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc6_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc11_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc13_out_dtype_float64",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_unsafe_ufunc0_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc14_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc15_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc15_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc4_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc14_out_dtype_complex128",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_no_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc10_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc13_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc12_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc2_out_dtype_float64",
    "TestUnaryUfuncs.test_x_and_out_broadcast_ufunc0",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc16_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc13_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc5_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc8_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc3_out_dtype_complex128",
    "TestUnaryUfuncs.test_x_and_out_casting_casting_safe_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc15_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc12_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc6_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc8_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc11_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc8_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc14_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc10_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc7_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc11_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc9_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc10_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc16_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc9_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc1_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc3_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc10_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc7_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc7_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc0_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc8",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc11_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_no_ufunc13_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc13_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc2_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc1_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc10_out_dtype_float64",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc5_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_unsafe_ufunc4_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_equiv_ufunc6_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc13",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc2_out_dtype_float32",
    "TestBinaryUfuncs.test_xy_and_out_broadcast_ufunc2",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_safe_ufunc10_out_dtype_complex128",
    "TestBinaryUfuncs.test_xy_and_out_casting_casting_same_kind_ufunc1_out_dtype_float64",
    "TestIsScalar.test_is_not_scalar_value6",
    "TestGenericReductions.test_bad_axis_func0",
    "TestGenericReductions.test_bad_axis_func11",
    "TestGenericReductions.test_bad_axis_func7",
    "TestGenericReductions.test_bad_axis_func6",
    "TestGenericReductions.test_bad_axis_func2",
    "TestGenericCumSumProd.test_bad_axis_func1",
    "TestGenericReductions.test_bad_axis_func3",
    "TestGenericReductions.test_bad_axis_func4",
    "TestGenericReductions.test_bad_axis_func10",
    "TestGenericReductions.test_bad_axis_func5",
    "TestGenericReductions.test_bad_axis_func8",
    "TestGenericReductions.test_bad_axis_func1",
    "TestGenericCumSumProd.test_bad_axis_func0",
    "TestGenericReductions.test_bad_axis_func9",
    "TestShuffle.test_1d_use_numpy_True",
    "TestShuffle.test_1d_use_numpy_False",
    "TestShuffle.test_2d_use_numpy_True",
    "TestShuffle.test_2d_use_numpy_False",
    "TestWritebackIfCopy.test_take_mode_raise",
    "TestArange.test_infinite",
    "TestArrayConstruction.test_array_empty",
    "TestAttributes.test_fill_readonly",
    "TestArrayAttributeDeletion.test_multiarray_writable_attributes_deletion",
    "TestMatmul.test_out_contiguous",
    "TestMinMax.test_scalar",
    "TestFromBuffer.test_basic_little_dtype2",
    "TestArrayCreationCopyArgument.test_striding_not_ok",
    "TestArange.test_require_range",
    "TestArange.test_nan_step",
    "TestWritebackIfCopy.test_argmin_with_out",
    "TestArrayAttributeDeletion.test_multiarray_not_writable_attributes_deletion",
    "TestLexsort.test_datetime",
    "TestMinMax.test_axis",
    "TestLexsort.test_mixed",
    "TestWritebackIfCopy.test_dot_out",
    "TestAttributes.test_fill_struct_array",
    "TestFromBuffer.test_empty",
    "TestAssignment.test_assignment_broadcasting",
    "TestMatmul.test_out_arg",
    "TestAttributes.test_set_stridesattr",
    "TestStats.test_out",
    "TestScalarIndexing.test_invalid_subscript",
    "TestWhere.test_error",
    "TestWritebackIfCopy.test_argmax_with_out",
    "TestBool.test_sum_2",
    "TestScalarIndexing.test_invalid_newaxis",
    "TestTake.test_out_overlap",
    "TestScalarIndexing.test_invalid_subscript_assignment",
    "TestFromBuffer.test_basic_little_dtype1",
    "TestWritebackIfCopy.test_choose_mod_raise",
    "TestAttributes.test_fill_max_uint64",
    "TestPutmask.test_byteorder_dtype_<i4",
    "TestPutmask.test_byteorder_dtype_>i4",
    "TestAttributes.test_stridesattr",
    "TestArange.test_zero_step",
    "TestStats.test_dtype_from_dtype",
    "TestArrayCreationCopyArgument.test_scalars",
    "TestConversion.test_to_int_scalar",
    "TestPutmask.test_record_array",
    "TestTake.test_raise",
    "TestFromBuffer.test_basic_little_dtype0",
    "TestMatmul.test_exceptions",
    "TestFlag.test_writeable_from_readonly",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_positional_arr_method_argmax_np_method0",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_1_method_argmin",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_arr_method_argmax_np_method0",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_arr_method_argmin_np_method1",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_0_method_argmax",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_positional_arr_method_argmin_np_method1",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_1_method_argmax",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_0_method_argmin",
    "TestConvertDType.test_convert_np_dtypes_'int64'",
    "TestConvertDType.test_convert_np_dtypes_'uint8'",
    "TestConvertDType.test_convert_np_dtypes_bool",
    "TestConvertDType.test_convert_np_dtypes_'complex128'",
    "TestConvertDType.test_convert_np_dtypes_'float16'",
    "TestConvertDType.test_convert_np_dtypes_'int16'",
    "TestConvertDType.test_convert_np_dtypes_'int32'",
    "TestConvertDType.test_convert_np_dtypes_'int8'",
    "TestConvertDType.test_convert_np_dtypes_'float64'",
    "TestConvertDType.test_convert_np_dtypes_'float32'",
    "TestConvertDType.test_convert_np_dtypes_'complex64'",
    "TestConvertDType.test_convert_np_dtypes_'bool_'",
    "TestOneArr.test_asarray_list_func55",
    "TestOneArr.test_asarray_tensor_func65",
    "TestOneArr.test_asarray_tensor_func44",
    "TestOneArr.test_asarray_array_func59",
    "TestOneArr.test_asarray_array_func45",
    "TestOneArr.test_asarray_list_func70",
    "TestOneArrAndAxis.test_andaxis_list_func7_axis_0",
    "TestSequenceOfArrays.test_single_array_func1",
    "TestOneArrAndAxis.test_andaxis_array_func1_axis_1",
    "TestSequenceOfArraysToSingle.test_several_func6",
    "TestOneArr.test_asarray_list_func0",
    "TestOneArrAndAxis.test_andaxis_list_func9_axis_1",
    "TestOneArrAndAxis.test_andaxis_list_func8_axis_0",
    "TestOneArr.test_asarray_list_func36",
    "TestOneArrAndAxis.test_andaxis_array_func3_axis_0",
    "TestOneArr.test_asarray_tensor_func15",
    "TestOneArr.test_asarray_array_func51",
    "TestOneArr.test_asarray_list_func16",
    "TestOneArrAndAxis.test_andaxis_tensor_func5_axis_0",
    "TestOneArrAndAxis.test_andaxis_tensor_func1_axis_1",
    "TestOneArr.test_asarray_tensor_func1",
    "TestOneArrAndAxesTuple.test_andtuple_list_func0_axes2",
    "TestOneArrAndAxis.test_andaxis_list_func6_axis_1",
    "TestOneArrAndAxis.test_andaxis_tensor_func10_axis_-1",
    "TestSequenceOfArraysToSingle.test_several_func2",
    "TestOneArrAndAxis.test_andaxis_array_func5_axis_1",
    "TestOneArrAndAxis.test_andaxis_list_func10_axis_1",
    "TestOneArr.test_asarray_array_func72",
    "TestOneArrAndShape.test_andshape_list_func0",
    "TestCtorNested.test_arrays_in_lists",
    "TestOneArr.test_asarray_tensor_func51",
    "TestOneArr.test_asarray_array_func0",
    "TestOneArr.test_asarray_array_func10",
    "TestOneArr.test_asarray_array_func43",
    "TestOneArrToScalar.test_toscalar_array_func2_np_func2",
    "TestOneArr.test_asarray_list_func3",
    "TestOneArr.test_asarray_array_func56",
    "TestArrayToSequence.test_asarray_array_func1",
    "TestOneArrAndShape.test_andshape_tensor_func4",
    "TestOneArr.test_asarray_list_func60",
    "TestDivmod.test_divmod_out",
    "TestOneArrAndAxis.test_andaxis_list_func7_axis3",
    "TestOneArrAndAxis.test_andaxis_array_func6_axis_0",
    "TestOneArrAndAxis.test_andaxis_list_func5_axis_0",
    "TestOneArr.test_asarray_tensor_func53",
    "TestOneArrAndAxis.test_andaxis_array_func6_axis3",
    "TestOneArr.test_asarray_tensor_func73",
    "TestDivmod.test_divmod_no_out",
    "TestOneArrAndAxis.test_andaxis_array_func9_axis_1",
    "TestOneArr.test_asarray_list_func58",
    "TestOneArrAndAxis.test_andaxis_tensor_func8_axis_0",
    "TestOneArr.test_asarray_array_func49",
    "TestOneArr.test_asarray_array_func60",
    "TestOneArr.test_asarray_tensor_func62",
    "TestOneArrAndAxesTuple.test_andtuple_tensor_func0_axes0",
    "TestOneArr.test_asarray_array_func22",
    "TestOneArr.test_asarray_list_func24",
    "TestOneArr.test_asarray_list_func15",
    "TestSequenceOfArrays.test_several_func2",
    "TestOneArr.test_asarray_tensor_func66",
    "TestOneArrAndAxis.test_andaxis_tensor_func7_axis3",
    "TestOneArrAndAxis.test_andaxis_tensor_func10_axis_0",
    "TestOneArrAndAxis.test_andaxis_list_func1_axis_-1",
    "TestOneArr.test_asarray_list_func32",
    "TestOneArr.test_asarray_list_func48",
    "TestOneArrToScalar.test_toscalar_array_func1_np_func1",
    "TestOneArr.test_asarray_list_func23",
    "TestOneArr.test_asarray_list_func65",
    "TestOneArr.test_asarray_tensor_func34",
    "TestOneArr.test_asarray_array_func57",
    "TestOneArr.test_asarray_list_func31",
    "TestOneArrAndAxis.test_andaxis_array_func9_axis_0",
    "TestOneArr.test_asarray_array_func63",
    "TestOneArrAndAxis.test_andaxis_tensor_func9_axis_1",
    "TestOneArr.test_asarray_tensor_func0",
    "TestOneArr.test_asarray_list_func43",
    "TestOneArr.test_asarray_list_func62",
    "TestOneArrAndShape.test_andshape_array_func0",
    "TestSequenceOfArrays.test_several_func0",
    "TestOneArrAndAxis.test_andaxis_array_func8_axis_-1",
    "TestOneArr.test_asarray_tensor_func29",
    "TestArrayToSequence.test_asarray_array_func0",
    "TestOneArrAndAxis.test_andaxis_array_func5_axis3",
    "TestOneArr.test_asarray_array_func16",
    "TestOneArr.test_asarray_array_func68",
    "TestOneArr.test_asarray_list_func21",
    "TestOneArrAndAxis.test_andaxis_list_func7_axis_1",
    "TestOneArr.test_asarray_array_func33",
    "TestOneArr.test_asarray_list_func13",
    "TestOneArr.test_asarray_list_func40",
    "TestOneArrAndAxis.test_andaxis_array_func1_axis_0",
    "TestOneArrAndAxesTuple.test_andtuple_list_func0_axes0",
    "TestOneArr.test_asarray_list_func52",
    "TestOneArr.test_asarray_array_func42",
    "TestOneArr.test_asarray_list_func73",
    "TestOneArr.test_asarray_array_func24",
    "TestOneArr.test_asarray_list_func45",
    "TestOneArr.test_asarray_array_func38",
    "TestOneArr.test_asarray_array_func20",
    "TestOneArr.test_asarray_tensor_func45",
    "TestOneArr.test_asarray_array_func66",
    "TestOneArrAndAxis.test_andaxis_list_func2_axis_0",
    "TestOneArr.test_asarray_array_func11",
    "TestOneArrAndAxis.test_andaxis_array_func9_axis3",
    "TestOneArrAndAxis.test_andaxis_list_func5_axis_-1",
    "TestOneArrAndShape.test_andshape_list_func1",
    "TestPythonArgsToArray.test_argstoarray_simple_func4_args4",
    "TestOneArr.test_asarray_tensor_func14",
    "TestOneArr.test_asarray_array_func48",
    "TestOneArr.test_asarray_list_func53",
    "TestOneArr.test_asarray_tensor_func24",
    "TestOneArr.test_asarray_list_func54",
    "TestOneArr.test_asarray_tensor_func33",
    "TestPythonArgsToArray.test_argstoarray_simple_func7_args7",
    "TestOneArrAndAxesTuple.test_andtuple_array_func0_axes1",
    "TestOneArrAndAxis.test_andaxis_list_func2_axis_1",
    "TestSequenceOfArrays.test_single_array_func0",
    "TestOneArr.test_asarray_tensor_func69",
    "TestSequenceOfArraysToSingle.test_several_func3",
    "TestOneArr.test_asarray_array_func36",
    "TestOneArr.test_asarray_list_func11",
    "TestCopyTo.test_copyto_typecast",
    "TestOneArrAndShape.test_andshape_tensor_func1",
    "TestOneArr.test_asarray_array_func71",
    "TestOneArrAndAxis.test_andaxis_list_func6_axis_0",
    "TestOneArrAndAxis.test_andaxis_tensor_func9_axis_0",
    "TestOneArrAndAxis.test_andaxis_array_func2_axis_0",
    "TestOneArr.test_asarray_list_func72",
    "TestSequenceOfArraysToSingle.test_several_func4",
    "TestOneArrAndAxis.test_andaxis_tensor_func2_axis_0",
    "TestOneArrAndAxis.test_andaxis_list_func2_axis_-1",
    "TestOneArr.test_asarray_array_func34",
    "TestOneArr.test_asarray_array_func23",
    "TestOneArr.test_asarray_list_func20",
    "TestOneArrAndAxis.test_andaxis_array_func6_axis_1",
    "TestOneArr.test_asarray_array_func41",
    "TestOneArr.test_asarray_list_func38",
    "TestOneArrAndAxis.test_andaxis_list_func5_axis_1",
    "TestOneArrAndAxis.test_andaxis_array_func3_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func3_axis3",
    "TestOneArrToScalar.test_toscalar_array_func0_np_func0",
    "TestOneArr.test_asarray_tensor_func37",
    "TestOneArr.test_asarray_tensor_func20",
    "TestOneArr.test_asarray_tensor_func42",
    "TestOneArr.test_asarray_list_func67",
    "TestOneArr.test_asarray_list_func30",
    "TestOneArrAndAxis.test_andaxis_list_func4_axis_1",
    "TestSequenceOfArrays.test_several_func3",
    "TestOneArr.test_asarray_array_func54",
    "TestOneArrAndShape.test_andshape_list_func4",
    "TestOneArr.test_asarray_tensor_func2",
    "TestOneArr.test_asarray_tensor_func57",
    "TestOneArrAndAxis.test_andaxis_list_func9_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func0_axis_1",
    "TestOneArrAndAxis.test_andaxis_list_func4_axis_-1",
    "TestOneArr.test_asarray_array_func55",
    "TestOneArrAndAxis.test_andaxis_tensor_func3_axis_-1",
    "TestOneArrAndAxis.test_andaxis_tensor_func5_axis_-1",
    "TestOneArr.test_asarray_list_func14",
    "TestOneArr.test_asarray_list_func29",
    "TestOneArrAndAxis.test_andaxis_array_func7_axis_1",
    "TestOneArrAndShape.test_andshape_list_func3",
    "TestOneArr.test_asarray_tensor_func5",
    "TestOneArr.test_asarray_list_func68",
    "TestOneArr.test_asarray_tensor_func61",
    "TestSequenceOfArrays.test_single_list_func3",
    "TestOneArr.test_asarray_array_func21",
    "TestOneArr.test_asarray_list_func61",
    "TestOneArr.test_asarray_tensor_func55",
    "TestOneArr.test_asarray_tensor_func18",
    "TestOneArr.test_asarray_list_func50",
    "TestOneArrAndAxis.test_andaxis_array_func7_axis_0",
    "TestOneArr.test_asarray_array_func62",
    "TestOneArr.test_asarray_tensor_func50",
    "TestOneArr.test_asarray_array_func6",
    "TestOneArr.test_asarray_list_func66",
    "TestOneArr.test_asarray_list_func59",
    "TestOneArr.test_asarray_tensor_func28",
    "TestShapeLikeToArray.test_shape_func3",
    "TestOneArr.test_asarray_array_func9",
    "TestOneArrAndAxis.test_andaxis_array_func0_axis_0",
    "TestOneArrAndShape.test_andshape_array_func2",
    "TestPythonArgsToArray.test_argstoarray_simple_func2_args2",
    "TestOneArrAndShape.test_andshape_tensor_func0",
    "TestPythonArgsToArray.test_argstoarray_simple_func0_args0",
    "TestOneArr.test_asarray_array_func19",
    "TestOneArr.test_asarray_tensor_func39",
    "TestOneArr.test_asarray_array_func65",
    "TestSequenceOfArrays.test_single_list_func2",
    "TestOneArr.test_asarray_array_func31",
    "TestOneArrAndAxis.test_andaxis_list_func10_axis_0",
    "TestOneArr.test_asarray_list_func2",
    "TestOneArrAndAxis.test_andaxis_array_func10_axis3",
    "TestOneArrAndAxis.test_andaxis_array_func4_axis_1",
    "TestDivmod.test_divmod_out_list",
    "TestOneArr.test_asarray_list_func19",
    "TestOneArrAndAxesTuple.test_andtuple_array_func0_axes2",
    "TestOneArr.test_asarray_array_func1",
    "TestOneArrAndAxis.test_andaxis_tensor_func4_axis_1",
    "TestOneArr.test_asarray_tensor_func43",
    "TestOneArrAndAxis.test_andaxis_array_func5_axis_0",
    "TestOneArrAndAxesTuple.test_andtuple_tensor_func0_axes2",
    "TestOneArr.test_asarray_list_func10",
    "TestSequenceOfArrays.test_single_array_func3",
    "TestOneArr.test_asarray_tensor_func40",
    "TestSequenceOfArraysToSingle.test_several_func0",
    "TestOneArrAndAxis.test_andaxis_array_func7_axis3",
    "TestOneArrAndAxis.test_andaxis_array_func6_axis_-1",
    "TestOneArr.test_asarray_tensor_func35",
    "TestOneArr.test_asarray_tensor_func72",
    "TestOneArr.test_asarray_list_func18",
    "TestOneArr.test_asarray_tensor_func60",
    "TestOneArrAndAxis.test_andaxis_list_func3_axis_0",
    "TestOneArr.test_asarray_array_func37",
    "TestOneArr.test_asarray_array_func74",
    "TestNormalizations.test_unknown_args",
    "TestOneArr.test_asarray_array_func4",
    "TestOneArr.test_asarray_array_func58",
    "TestOneArrAndAxis.test_andaxis_list_func9_axis_0",
    "TestOneArr.test_asarray_tensor_func22",
    "TestOneArr.test_asarray_list_func56",
    "TestOneArrAndAxis.test_andaxis_list_func3_axis_1",
    "TestOneArrAndAxis.test_andaxis_array_func0_axis_-1",
    "TestOneArr.test_asarray_tensor_func4",
    "TestPythonArgsToArray.test_argstoarray_simple_func6_args6",
    "TestOneArrAndAxis.test_andaxis_tensor_func9_axis_-1",
    "TestOneArr.test_asarray_tensor_func68",
    "TestOneArr.test_asarray_list_func27",
    "TestOneArrAndAxis.test_andaxis_array_func4_axis_-1",
    "TestOneArr.test_asarray_array_func13",
    "TestOneArr.test_asarray_list_func6",
    "TestOneArr.test_asarray_array_func39",
    "TestOneArr.test_asarray_array_func73",
    "TestOneArr.test_asarray_tensor_func12",
    "TestOneArrAndAxis.test_andaxis_array_func7_axis_-1",
    "TestOneArr.test_asarray_list_func17",
    "TestShapeLikeToArray.test_shape_func2",
    "TestOneArrAndAxis.test_andaxis_list_func4_axis_0",
    "TestOneArrAndAxis.test_andaxis_array_func3_axis_1",
    "TestOneArrAndAxis.test_andaxis_tensor_func10_axis_1",
    "TestOneArrAndAxis.test_andaxis_list_func8_axis_1",
    "TestOneArr.test_asarray_list_func33",
    "TestOneArrAndAxis.test_andaxis_tensor_func1_axis_-1",
    "TestOneArr.test_asarray_array_func18",
    "TestOneArr.test_asarray_tensor_func3",
    "TestOneArrAndShape.test_andshape_tensor_func2",
    "TestOneArr.test_asarray_list_func35",
    "TestOneArrAndAxis.test_andaxis_tensor_func3_axis_0",
    "TestOneArr.test_asarray_array_func70",
    "TestOneArrAndAxesTuple.test_andtuple_list_func0_axes1",
    "TestOneArrAndAxis.test_andaxis_list_func8_axis_-1",
    "TestOneArr.test_asarray_tensor_func59",
    "TestOneArr.test_asarray_array_func15",
    "TestOneArrAndAxis.test_andaxis_tensor_func6_axis_1",
    "TestOneArr.test_asarray_tensor_func38",
    "TestPythonArgsToArray.test_argstoarray_simple_func8_args8",
    "TestPythonArgsToArray.test_argstoarray_simple_func3_args3",
    "TestOneArr.test_asarray_array_func14",
    "TestPythonArgsToArray.test_argstoarray_simple_func5_args5",
    "TestOneArr.test_asarray_list_func26",
    "TestOneArr.test_asarray_list_func34",
    "TestOneArr.test_asarray_list_func4",
    "TestOneArr.test_asarray_tensor_func67",
    "TestOneArr.test_asarray_array_func3",
    "TestOneArr.test_asarray_array_func5",
    "TestOneArr.test_asarray_array_func52",
    "TestOneArr.test_asarray_tensor_func58",
    "TestOneArr.test_asarray_tensor_func48",
    "TestOneArr.test_asarray_array_func50",
    "TestOneArr.test_asarray_tensor_func47",
    "TestOneArrAndAxis.test_andaxis_array_func4_axis3",
    "TestOneArrAndAxis.test_andaxis_tensor_func2_axis_1",
    "TestOneArrAndAxis.test_andaxis_array_func0_axis3",
    "TestShapeLikeToArray.test_shape_func1",
    "TestOneArrAndAxis.test_andaxis_tensor_func4_axis_-1",
    "TestOneArrAndAxis.test_andaxis_tensor_func8_axis_-1",
    "TestDefaultDtype.test_defaultdtype_defaults",
    "TestOneArr.test_asarray_list_func63",
    "TestOneArrAndShape.test_andshape_list_func2",
    "TestOneArr.test_asarray_array_func27",
    "TestOneArrAndAxis.test_andaxis_array_func4_axis_0",
    "TestOneArr.test_asarray_list_func41",
    "TestSequenceOfArrays.test_single_tensor_func2",
    "TestOneArr.test_asarray_list_func39",
    "TestOneArr.test_asarray_tensor_func6",
    "TestOneArr.test_asarray_tensor_func25",
    "TestOneArr.test_asarray_array_func2",
    "TestOneArrAndAxis.test_andaxis_array_func8_axis_1",
    "TestOneArr.test_asarray_tensor_func56",
    "TestOneArr.test_asarray_array_func69",
    "TestOneArr.test_asarray_list_func28",
    "TestOneArr.test_asarray_tensor_func26",
    "TestArrayToSequence.test_asarray_tensor_func1",
    "TestOneArr.test_asarray_array_func28",
    "TestPythonArgsToArray.test_argstoarray_simple_func1_args1",
    "TestOneArrAndAxis.test_andaxis_list_func10_axis3",
    "TestOneArr.test_asarray_list_func44",
    "TestOneArr.test_asarray_array_func46",
    "TestOneArrAndAxis.test_andaxis_array_func10_axis_1",
    "TestOneArr.test_asarray_tensor_func30",
    "TestOneArr.test_asarray_tensor_func16",
    "TestOneArrAndAxis.test_andaxis_array_func1_axis3",
    "TestOneArr.test_asarray_tensor_func46",
    "TestOneArr.test_asarray_tensor_func10",
    "TestOneArrAndAxis.test_andaxis_array_func2_axis_-1",
    "TestOneArr.test_asarray_list_func47",
    "TestSequenceOfArrays.test_single_tensor_func0",
    "TestOneArrAndAxesTuple.test_andtuple_array_func0_axes0",
    "TestOneArr.test_asarray_list_func12",
    "TestOneArrAndAxis.test_andaxis_array_func8_axis3",
    "TestShapeLikeToArray.test_shape_func0",
    "TestOneArr.test_asarray_array_func61",
    "TestOneArrAndAxis.test_andaxis_tensor_func7_axis_-1",
    "TestOneArrAndAxis.test_andaxis_list_func0_axis_0",
    "TestOneArr.test_asarray_tensor_func31",
    "TestOneArr.test_asarray_array_func67",
    "TestOneArr.test_asarray_list_func64",
    "TestOneArrAndAxis.test_andaxis_array_func5_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func2_axis_1",
    "TestOneArr.test_asarray_array_func32",
    "TestOneArr.test_asarray_array_func8",
    "TestOneArr.test_asarray_list_func5",
    "TestOneArr.test_asarray_array_func17",
    "TestOneArrAndAxis.test_andaxis_list_func7_axis_-1",
    "TestOneArrAndAxis.test_andaxis_tensor_func5_axis_1",
    "TestOneArrAndAxis.test_andaxis_list_func0_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func8_axis_0",
    "TestOneArr.test_asarray_array_func64",
    "TestArrayToSequence.test_asarray_tensor_func0",
    "TestSequenceOfArrays.test_single_array_func2",
    "TestOneArrAndAxis.test_andaxis_list_func10_axis_-1",
    "TestOneArr.test_asarray_list_func71",
    "TestOneArrAndAxesTuple.test_andtuple_tensor_func0_axes1",
    "TestOneArrAndAxis.test_andaxis_tensor_func1_axis_0",
    "TestOneArr.test_asarray_array_func44",
    "TestCopyTo.test_copyto_basic",
    "TestSequenceOfArrays.test_single_tensor_func1",
    "TestOneArr.test_asarray_tensor_func11",
    "TestSequenceOfArrays.test_several_func1",
    "TestOneArr.test_asarray_tensor_func74",
    "TestOneArr.test_asarray_tensor_func36",
    "TestOneArr.test_asarray_array_func53",
    "TestOneArr.test_asarray_tensor_func63",
    "TestOneArrAndShape.test_andshape_array_func3",
    "TestOneArr.test_asarray_list_func74",
    "TestOneArr.test_asarray_tensor_func49",
    "TestOneArrAndAxis.test_andaxis_tensor_func3_axis_1",
    "TestOneArr.test_asarray_tensor_func32",
    "TestOneArrAndAxis.test_andaxis_list_func1_axis_1",
    "TestOneArrAndAxis.test_andaxis_tensor_func4_axis_0",
    "TestOneArrAndShape.test_andshape_tensor_func3",
    "TestOneArr.test_asarray_tensor_func27",
    "TestOneArr.test_asarray_list_func22",
    "TestOneArr.test_asarray_list_func69",
    "TestOneArr.test_asarray_array_func26",
    "TestOneArrAndAxis.test_andaxis_array_func9_axis_-1",
    "TestOneArrAndAxis.test_andaxis_tensor_func6_axis_-1",
    "TestSequenceOfArrays.test_single_tensor_func3",
    "TestOneArrAndShape.test_andshape_array_func1",
    "TestOneArr.test_asarray_array_func25",
    "TestOneArrAndAxis.test_andaxis_tensor_func2_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func2_axis3",
    "TestOneArr.test_asarray_tensor_func41",
    "TestOneArrAndAxis.test_andaxis_tensor_func0_axis_1",
    "TestOneArr.test_asarray_list_func49",
    "TestOneArr.test_asarray_list_func57",
    "TestOneArrAndAxis.test_andaxis_tensor_func8_axis_1",
    "TestOneArr.test_asarray_tensor_func71",
    "TestSequenceOfArrays.test_single_list_func1",
    "TestPythonArgsToArray.test_argstoarray_simple_func9_args9",
    "TestOneArr.test_asarray_list_func37",
    "TestOneArrAndAxis.test_andaxis_tensor_func0_axis_0",
    "TestOneArr.test_asarray_array_func30",
    "TestOneArr.test_asarray_tensor_func21",
    "TestOneArr.test_asarray_array_func35",
    "TestOneArr.test_asarray_tensor_func64",
    "TestOneArr.test_asarray_list_func51",
    "TestOneArr.test_asarray_array_func47",
    "TestOneArrAndAxis.test_andaxis_tensor_func7_axis_1",
    "TestOneArr.test_asarray_array_func29",
    "TestOneArrAndAxis.test_andaxis_array_func1_axis_-1",
    "TestOneArr.test_asarray_tensor_func19",
    "TestOneArrAndAxis.test_andaxis_list_func1_axis_0",
    "TestOneArr.test_asarray_tensor_func17",
    "TestOneArrAndAxis.test_andaxis_list_func0_axis_1",
    "TestOneArr.test_asarray_tensor_func70",
    "TestOneArr.test_asarray_tensor_func54",
    "TestOneArr.test_asarray_tensor_func23",
    "TestOneArr.test_asarray_array_func7",
    "TestOneArr.test_asarray_array_func12",
    "TestOneArrAndAxis.test_andaxis_list_func3_axis_-1",
    "TestOneArrAndAxis.test_andaxis_array_func10_axis_0",
    "TestOneArr.test_asarray_tensor_func13",
    "TestOneArrAndAxis.test_andaxis_tensor_func6_axis_0",
    "TestOneArrAndShape.test_andshape_array_func4",
    "TestOneArrAndAxis.test_andaxis_tensor_func10_axis3",
    "TestOneArr.test_asarray_array_func40",
    "TestOneArrAndAxis.test_andaxis_tensor_func7_axis_0",
    "TestOneArr.test_asarray_list_func42",
    "TestOneArrAndAxis.test_andaxis_tensor_func0_axis_-1",
    "TestOneArr.test_asarray_list_func25",
    "TestOneArr.test_asarray_tensor_func52",
    "TestOneArrAndAxis.test_andaxis_list_func6_axis_-1",
    "TestSequenceOfArraysToSingle.test_several_func1",
    "TestCopyTo.test_copytobcast",
    "TestOneArrAndAxis.test_andaxis_array_func10_axis_-1",
    "TestSequenceOfArraysToSingle.test_several_func5",
    "TestOneArr.test_asarray_list_func1",
    "TestOneArr.test_asarray_list_func46",
    "TestSequenceOfArrays.test_single_list_func0",
    "TestCond.test_sq_cases",
    "TestNormInt64.test_bad_args",
    "TestQR.test_qr_empty_m_0_n_3",
    "TestMultiDot.test_dynamic_programming_optimization_and_out",
    "TestNormDouble.test_bad_args",
    "TestCond.test_empty_sq_cases",
    "TestQR.test_qr_empty_m_0_n_0",
    "TestQR.test_mode_raw",
    "TestMultiDot.test_two_arguments_and_out",
    "TestMultiDot.test_three_arguments_and_out",
    "TestNormDouble.test_axis",
    "TestMisc.test_generalized_raise_multiloop",
    "TestEigvalsh.test_invalid",
    "TestNormDouble.test_matrix_2x2",
    "TestMisc.test_byteorder_check",
    "TestNormInt64.test_axis",
    "TestQR.test_qr_empty_m_3_n_0",
    "TestEigh.test_invalid",
    "TestNormSingle.test_bad_args",
    "TestNormSingle.test_matrix_2x2",
    "TestNormSingle.test_axis",
    "TestMultiDot.test_too_few_input_arrays",
    "TestNormInt64.test_matrix_2x2",
    "TestFliplr.test_basic",
    "TestHistogram2d.test_binparameter_combination",
    "TestHistogram2d.test_all_outliers",
    "TestTriuIndicesFrom.test_exceptions",
    "TestTrilIndicesFrom.test_exceptions",
    "TestHistogram2d.test_asym",
    "TestDiag.test_failure",
    "TestVsplit.test_non_iterable",
    "TestVsplit.test_1D_array",
    "TestApplyAlongAxis.test_scalar_array",
    "TestDstack.test_non_iterable",
    "TestSplit.test_unequal_split",
    "TestPutAlongAxis.test_broadcast",
    "TestArraySplit.test_integer_0_split",
    "TestDsplit.test_2D_array",
    "TestTakeAlongAxis.test_invalid",
    "TestHsplit.test_0D_array",
    "TestDsplit.test_1D_array",
    "TestDsplit.test_non_iterable",
    "TestDsplit.test_0D_array",
    "TestHsplit.test_non_iterable",
    "TestColumnStack.test_non_iterable",
    "TestApplyAlongAxis.test_axis_insertion",
    "TestVsplit.test_0D_array",
    "TestExpandDims.test_repeated_axis",
    "TestExpandDims.test_axis_out_of_range",
    "TestApplyAlongAxis.test_0d_array",
    "TestHistogramdd.test_bins_errors",
    "TestHistogramdd.test_equal_edges",
    "TestHistogram.test_precision",
    "TestHistogramdd.test_finite_range",
    "TestHistogramdd.test_weights",
    "TestHistogram.test_error_binnum_type",
    "TestHistogram.test_finite_range",
    "TestHistogramdd.test_inf_edges",
    "TestHistogramdd.test_bins_error_2",
    "TestHistogramdd.test_simple",
    "TestHistogram.test_one_bin",
    "TestHistogram.test_unsigned_monotonicity_check",
    "TestQuantile.test_quantile_monotonic_method_weibull",
    "TestGradient.test_badargs",
    "TestRot90.test_basic",
    "TestDiff.test_axis",
    "TestQuantile.test_quantile_monotonic_method_median_unbiased",
    "TestGradient.test_values",
    "TestCov.test_aweights",
    "TestQuantile.test_quantile_monotonic_method_interpolated_inverted_cdf",
    "TestQuantile.test_quantile_monotonic_method_inverted_cdf",
    "TestPercentile.test_keepdims_out_q1_axis_1",
    "TestSortComplex.test_sort_real_type_in_g_type_out_G",
    "TestMedian.test_keepdims_out_axis2",
    "TestMeshgrid.test_invalid_arguments",
    "TestGradient.test_specific_axes",
    "TestPercentile.test_keepdims_out_q_7_axis4",
    "TestPercentile.test_keepdims_out_q1_axis4",
    "TestDelete.test_slices",
    "TestPercentile.test_extended_axis_invalid",
    "TestGradient.test_second_order_accurate",
    "TestMedian.test_keepdims_out_axis0",
    "TestDiff.test_prepend",
    "TestMedian.test_keepdims_out_axis_1",
    "TestPercentile.test_keepdims_out_q1_axis0",
    "TestQuantile.test_quantile_monotonic_method_averaged_inverted_cdf",
    "TestMedian.test_keepdims_out_axis4",
    "TestBincount.test_with_incorrect_minlength",
    "TestSortComplex.test_sort_real_type_in_H_type_out_F",
    "TestDiff.test_n",
    "TestMeshgrid.test_indexing",
    "TestQuantile.test_quantile_monotonic_method_closest_observation",
    "TestFlip.test_axes",
    "TestPercentile.test_keepdims_out_q1_axis3",
    "TestPercentile.test_keepdims_out_q_7_axis0",
    "TestMedian.test_keepdims_out_axis3",
    "TestCov.test_fweights",
    "TestDiff.test_append",
    "TestPercentile.test_scalar_q",
    "TestMedian.test_extended_axis_invalid",
    "TestMedian.test_out",
    "TestPercentile.test_keepdims_out_q_7_axis2",
    "TestPercentile.test_keepdims_out_q1_axis2",
    "TestQuantile.test_quantile_monotonic_method_hazen",
    "TestPercentile.test_keepdims_out_q_7_axis3",
    "TestPercentile.test_keepdims_out_q_7_axis_1",
    "TestPercentile.test_api",
    "TestQuantile.test_quantile_monotonic_method_normal_unbiased",
    "TestSetOps.test_in1d_table_timedelta_fails",
    "TestUnique.test_unique_axis_errors",
    "TestSetOps.test_setdiff1d",
    "TestSetOps.test_in1d_timedelta_kind_sort",
    "TestSetOps.test_in1d_timedelta_kind0",
    "TestUnique.test_unique_axis",
    "TestConstant.test_check_constant_float3",
    "TestConstant.test_check_constant_pad_2d",
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f8_casting_safe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_c8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_f8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f8_casting_no",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_exceptions",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_c8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_i8_casting_equiv",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_large_concatenate_axis_None",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_concatenate",  # torch_np/numpy_tests/core/test_shape_base
    "TestVstack.test_empty_input",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_i8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_i8_casting_no",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f8_casting_no",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f4_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_f8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestVstack.test_non_iterable",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_i8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f4_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_f8_casting_safe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f4_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_i8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_c8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f8_casting_safe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_c8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_f4_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f4_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_c8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_i8_casting_safe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_bad_out_shape",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_f8_casting_equiv",  # torch_np/numpy_tests/core/test_shape_base
    "TestHstack.test_non_iterable",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis0_out_dtype_c8_casting_same_kind",  # torch_np/numpy_tests/core/test_shape_base
    "TestHstack.test_empty_input",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_f4_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_out_and_dtype_axis_0_out_dtype_f8_casting_equiv",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack_out_and_dtype_axis_0_out_dtype_i8_casting_unsafe",  # torch_np/numpy_tests/core/test_shape_base
    "TestNegative.test_exceptions",  # torch_np/numpy_tests/core/test_scalarmath
    "TestPower.test_modular_power",  # torch_np/numpy_tests/core/test_scalarmath
    "TestBaseMath.test_lower_align",  # torch_np/numpy_tests/core/test_scalarmath
    "TestArrayFromScalar.test_integers_np_longlong_t26",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_np_intc_np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_t15_np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_np_longlong_np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_np_byte_np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_np_short_np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestArrayFromScalar.test_integers_np_int__np_longlong",  # torch_np/numpy_tests/core/test_scalar_ctors
    "TestScalarTypeNames.test_names_reflect_attributes_t4",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t1",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t7",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t5",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t9",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t6",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t2",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t8",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t0",  # torch_np/numpy_tests/core/test_numerictypes
    "TestScalarTypeNames.test_names_reflect_attributes_t3",  # torch_np/numpy_tests/core/test_numerictypes
    "TestClip.test_clip_inplace_array",  # torch_np/numpy_tests/core/test_numeric
    "TestRequire.test_require_each",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_with_out_simple_int32",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_inplace_01",  # torch_np/numpy_tests/core/test_numeric
    "TestStdVar.test_out_scalar",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_int32_inout_casting_unsafe",  # torch_np/numpy_tests/core/test_numeric
    "TestMoveaxis.test_errors",  # torch_np/numpy_tests/core/test_numeric
    "TestNonzeroAndCountNonzero.test_count_nonzero_axis",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_with_out_memory_overlap",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_func_takes_out",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_noncontig_inplace",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_type_cast_12",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_int64_out",  # torch_np/numpy_tests/core/test_numeric
    "TestRollaxis.test_exceptions",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_inplace_02",  # torch_np/numpy_tests/core/test_numeric
    "TestRequire.test_C_and_F_simul",  # torch_np/numpy_tests/core/test_numeric
    "TestNonarrayArgs.test_dunder_round_edgecases_val_2147483647_ndigits_-1",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_complex",  # torch_np/numpy_tests/core/test_numeric
    "TestBoolArray.test_logical_not_abs",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_out",  # torch_np/numpy_tests/core/test_numeric
    "TestBroadcast.test_broadcast_single_arg",  # torch_np/numpy_tests/core/test_numeric
    "TestRequire.test_unknown_requirement",  # torch_np/numpy_tests/core/test_numeric
    "TestBoolArray.test_logical_and_or_xor",  # torch_np/numpy_tests/core/test_numeric
    "TestBroadcast.test_broadcast_error_kwargs",  # torch_np/numpy_tests/core/test_numeric
    "TestNonarrayArgs.test_dunder_round_edgecases_val_2147483647_ndigits_-9",  # torch_np/numpy_tests/core/test_numeric
    "TestNonarrayArgs.test_dunder_round_edgecases_val_2147483647_ndigits_-10",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_type_cast_10",  # torch_np/numpy_tests/core/test_numeric
    "TestOuterMisc.test_outer_out_param",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_inplace_simple",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_with_out_transposed",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_with_out_simple",  # torch_np/numpy_tests/core/test_numeric
    "TestCross.test_broadcasting_shapes",  # torch_np/numpy_tests/core/test_numeric
    "TestIndexing.test_index_no_floats",  # torch_np/numpy_tests/core/test_indexing
    "TestBooleanIndexing.test_boolean_indexing_weirdness",  # torch_np/numpy_tests/core/test_indexing
    "TestBooleanIndexing.test_bool_as_int_argument_errors",  # torch_np/numpy_tests/core/test_indexing
    "TestBroadcastedAssignments.test_simple_broadcasting_errors",  # torch_np/numpy_tests/core/test_indexing
    "TestFloatNonIntegerArgument.test_non_integer_argument_errors",  # torch_np/numpy_tests/core/test_indexing
    "TestIndexing.test_slicing_no_floats",  # torch_np/numpy_tests/core/test_indexing
    "TestBroadcastedAssignments.test_prepend_not_one",  # torch_np/numpy_tests/core/test_indexing
    "TestFloatNonIntegerArgument.test_reduce_axis_float_index",  # torch_np/numpy_tests/core/test_indexing
    "TestEinsum.test_different_paths_dtype_f",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_D",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_e",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_fixed_collapsingbug",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_combined_views_mapping",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_B",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_cfloat64",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_broadcast",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_int32",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_b",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_fixedstridebug",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_out_is_res",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_subscript_range",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_float64",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_float32",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_cfloat128",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_small_boolean_arrays",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_i",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_d",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_l",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_h",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_misc",  # torch_np/numpy_tests/core/test_einsum
    "TestMisc.test_f16_on_cuda",
    "TestMisc.test_overlap",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_sums_int64",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_einsum_failed_on_p9_and_s390x",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_F",  # torch_np/numpy_tests/core/test_einsum
    "TestDLPack.test_dtype_passthrough_dtype4",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_23",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_12",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_27",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_32",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_from_dlpack_refcount",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype2",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_2",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_ndim0",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_1",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_17",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_13",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_14",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype7",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype9",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_29",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dunder_dlpack_refcount",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_15",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_non_contiguous",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype3",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_30",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_6",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_7",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype6",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype5",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_4",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_31",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_from_torch",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_24",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_21",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype8",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_28",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_3",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_10",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_0",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_16",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_18",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_20",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_11",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_25",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_5",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_22",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dlpack_device",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_9",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype0",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_dtype_passthrough_dtype1",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_19",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_26",  # torch_np/numpy_tests/core/test_dlpack
    "TestDLPack.test_higher_dims_ndim_8",  # torch_np/numpy_tests/core/test_dlpack
    "WeakTest.test_make_weak_keyed_dict_from_weak_keyed_dict",  # test_weak
    "TestViewOpsLAZY.test_advanced_indexing_assignment_lazy",  # test_view_ops
    "TestOldViewOpsCPU.test_crow_col_indices_cpu",  # test_view_ops
    "TestViewOpsLAZY.test_advanced_indexing_nonview_lazy",  # test_view_ops
    "TestTypePromotionCPU.test_alpha_mismatch_cpu",  # test_type_promotion
    "TestTypePromotionCPU.test_alternate_result_cpu",  # test_type_promotion
    "TestTypeHints.test_doc_examples",  # test_type_hints
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_float32_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape0_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPAFailureModesCPU.test_invalid_inputs_different_datatypes_kernel2_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_float64_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_float16_cpu_float16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape1_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPAFailureModesCPU.test_invalid_inputs_different_datatypes_kernel1_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_float32",
    "TestAttnMasksCPU.test_is_causal_and_mask_fails_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_attention_math_with_negative_scale_kernel0_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_float32",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape2_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_float64",
    "TestTransformersCPU.test_train_with_is_causal_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_bfloat16_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPAFailureModesCPU.test_invalid_inputs_1_dimensional_inputs_kernel0_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPAFailureModesCPU.test_invalid_inputs_different_datatypes_kernel0_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_float64",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape3_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_float32_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPAFailureModesCPU.test_invalid_inputs_1_dimensional_inputs_kernel2_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPAFailureModesCPU.test_invalid_inputs_1_dimensional_inputs_kernel1_cpu",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_float64_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_1_head_dim_16_causal_True_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_1030_n_head_1_head_dim_8_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_1_head_dim_8_causal_False_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_8_causal_False_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_12_seq_len_1030_n_head_3_head_dim_8_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_False_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_1_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_False_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_float16_cpu_float16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_seq_len_1030_n_head_3_head_dim_8_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_1030_n_head_3_head_dim_16_causal_True_train_False_cpu_float64",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_bfloat16_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_267_n_head_1_head_dim_16_causal_True_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_1_head_dim_8_causal_False_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_2_seq_len_267_n_head_3_head_dim_8_causal_True_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_seq_len_1030_n_head_3_head_dim_16_causal_False_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float64_batch_size_2_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_vs_math_cpu_fused_kernel0_float32_batch_size_12_seq_len_267_n_head_3_head_dim_16_causal_True_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_bfloat16_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_bfloat16",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float32_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float32",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_12_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_1030_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_1179_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_1_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_16_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_2_bool_mask_1_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_0_train_True_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_False_cpu_float64",
    "TestSDPACPU.test_scaled_dot_product_fused_attention_mask_vs_math_cpu_fused_kernel0_float64_batch_size_2_q_seq_len_267_kv_seq_len_514_n_head_3_head_dim_8_mask_dim_4_bool_mask_1_train_True_cpu_float64",
    "TestAssertCloseSparseCOO.test_matching_coalesced",  # test_testing
    "TestImports.test_circular_dependencies",  # test_testing
    "TestAssertCloseSparseCSR.test_mismatching_crow_indices_msg",  # test_testing
    "TestAssertCloseSparseBSC.test_mismatching_row_indices_msg",  # test_testing
    "TestAssertCloseSparseCOO.test_mismatching_values_msg",  # test_testing
    "TestAssertCloseQuantized.test_matching_per_channel",  # test_testing
    "TestTestParametrizationDeviceTypeCPU.test_ops_decorator_applies_op_and_param_specific_decorators_cpu",  # test_testing
    "TestAssertCloseSparseCOO.test_matching_uncoalesced",  # test_testing
    "TestAssertCloseSparseCSR.test_matching",  # test_testing
    "TestAssertCloseSparseBSR.test_mismatching_crow_indices_msg",  # test_testing
    "TestAssertCloseSparseBSR.test_matching",  # test_testing
    "TestAssertCloseQuantized.test_mismatching_is_quantized",  # test_testing
    "TestAssertCloseSparseCOO.test_mismatching_indices_msg",  # test_testing
    "TestAssertCloseSparseBSC.test_mismatching_ccol_indices_msg",  # test_testing
    "TestAssertCloseSparseBSC.test_mismatching_values_msg",  # test_testing
    "TestAssertCloseSparseCSC.test_mismatching_row_indices_msg",  # test_testing
    "TestAssertCloseSparseBSC.test_matching",  # test_testing
    "TestAssertCloseSparseCSC.test_matching",  # test_testing
    "TestAssertCloseSparseCSR.test_mismatching_values_msg",  # test_testing
    "TestAssertCloseSparseBSR.test_mismatching_values_msg",  # test_testing
    "TestAssertCloseSparseCSC.test_mismatching_values_msg",  # test_testing
    "TestAssertCloseSparseBSR.test_mismatching_col_indices_msg",  # test_testing
    "TestAssertCloseSparseCOO.test_mismatching_nnz",  # test_testing
    "TestAssertCloseSparseCSR.test_mismatching_col_indices_msg",  # test_testing
    "TestAssertCloseQuantized.test_mismatching_qscheme",  # test_testing
    "TestAssertCloseQuantized.test_matching_per_tensor",  # test_testing
    "TestAssertCloseSparseCSC.test_mismatching_ccol_indices_msg",  # test_testing
    "TestTensorBoardUtils.test_to_HWC",  # test_tensorboard
    "TestTensorBoardEmbedding.test_embedding",  # test_tensorboard
    "TestTensorProtoSummary.test_float_tensor_proto",  # test_tensorboard
    "TestTensorBoardSummary.test_image_without_channel",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_smoke",  # test_tensorboard
    "TestTensorBoardUtils.test_numpy_vid_uint8",  # test_tensorboard
    "TestTensorProtoSummary.test_complex_tensor_proto",  # test_tensorboard
    "TestTensorBoardSummary.test_image_with_one_channel",  # test_tensorboard
    "TestTensorBoardEmbedding.test_embedding_64",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_domain_discrete",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_wrong_parameter",  # test_tensorboard
    "TestTensorBoardSummary.test_video",  # test_tensorboard
    "TestTensorProtoSummary.test_int_tensor_proto",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_number",  # test_tensorboard
    "TestTensorBoardWriter.test_writer",  # test_tensorboard
    "TestTensorProtoSummary.test_empty_tensor_proto",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_string",  # test_tensorboard
    "TestTensorBoardSummary.test_hparams_bool",  # test_tensorboard
    "TestTensorBoardSummary.test_uint8_image",  # test_tensorboard
    "TestAsArrayCPU.test_copy_list_cpu_float64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_int64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_int32",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_float32",  # test_tensor_creation_ops
    "TestTensorCreationCPU.test_tensor_factory_type_inference_cpu",  # test_tensor_creation_ops
    "TestBufferProtocolCPU.test_byte_to_int_cpu",  # test_tensor_creation_ops
    "TestTensorCreationCPU.test_block_diag_cpu",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_int8",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_float16",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_complex64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_uint8",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_bfloat16",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_bool",  # test_tensor_creation_ops
    "TestTensorCreationCPU.test_constructor_dtypes_cpu",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_complex128",  # test_tensor_creation_ops
    "TestTensorCreationCPU.test_tensor_factory_copy_var_cpu",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_int16",  # test_tensor_creation_ops
    "TestTensorCreationCPU.test_cartesian_prod_cpu",  # test_tensor_creation_ops
    "TestSubclass.test_parametrization_non_wrapper_tensor_leave_parametrized_True",  # test_subclass
    "TestSubclass.test_module_optimization_non_wrapper_tensor",  # test_subclass
    "TestSubclass.test_serialization_non_wrapper_tensor_as_param_True",  # test_subclass
    "TestSubclass.test_module_optimization_sparse_tensor",  # test_subclass
    "TestSubclass.test_param_invariants_non_wrapper_tensor_tensor_requires_grad_False",  # test_subclass
    "TestSubclass.test_param_invariants_sparse_tensor_tensor_requires_grad_True",  # test_subclass
    "TestSubclass.test_param_invariants_diag_tensor_below_tensor_requires_grad_True",  # test_subclass
    "TestSubclass.test_param_invariants_diag_tensor_below_tensor_requires_grad_False",  # test_subclass
    "TestSubclass.test_param_invariants_non_wrapper_tensor_tensor_requires_grad_True",  # test_subclass
    "TestSubclass.test_parametrization_non_wrapper_tensor_leave_parametrized_False",  # test_subclass
    "TestSubclass.test_type_propagation_non_wrapper_tensor_as_param_False",  # test_subclass
    "TestSubclass.test_module_optimization_diag_tensor_below",  # test_subclass
    "TestSubclass.test_parametrization_base_tensor_leave_parametrized_True",  # test_subclass
    "TestSubclass.test_type_propagation_non_wrapper_tensor_as_param_True",  # test_subclass
    "TestSubclass.test_parametrization_base_tensor_leave_parametrized_False",  # test_subclass
    "TestSubclass.test_param_invariants_sparse_tensor_tensor_requires_grad_False",  # test_subclass
    "TestStatelessFunctionalAPI.test_reparametrize_module_fail_reset_to_original_torch_func",  # test_stateless
    "TestStatelessFunctionalAPI.test_reparametrized_module_change_parametrization_original_stateless",  # test_stateless
    "TestStatelessFunctionalAPI.test_reparametrized_module_change_parametrization_original_torch_func",  # test_stateless
    "TestStatelessFunctionalAPI.test_reparametrize_module_fail_reset_to_original_stateless",  # test_stateless
    "TestSortAndSelectCPU.test_isin_cpu_int32",  # test_sort_and_select
    "TestSortAndSelectCPU.test_sort_overflow_cpu_int16",  # test_sort_and_select
    "TestSortAndSelectCPU.test_topk_quantized_scalar_input_cpu",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_float64",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_uint8",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_int8",  # test_sort_and_select
    "TestSortAndSelectCPU.test_topk_arguments_cpu",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_int16",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_int64",  # test_sort_and_select
    "TestSortAndSelectCPU.test_isin_cpu_float32",  # test_sort_and_select
    "TestShapeOpsCPU.test_flip_cpu_float64",  # test_shape_ops
    "TestShapeOpsCPU.test_flip_cpu_float32",  # test_shape_ops
    "TestShapeOpsCPU.test_flip_cpu_complex64",  # test_shape_ops
    "TestShapeOpsCPU.test_flip_cpu_float16",  # test_shape_ops
    "TestShapeOpsCPU.test_flip_cpu_complex128",  # test_shape_ops
    "TestShapeOpsCPU.test_clamp_cpu_int64",  # test_shape_ops
    "TestShapeOpsCPU.test_clamp_propagates_nans_cpu",  # test_shape_ops
    "TestShapeOpsCPU.test_flip_cpu_bfloat16",  # test_shape_ops
    "TestShapeOpsCPU.test_clamp_cpu_float32",  # test_shape_ops
    "TestSubclassSerialization.test_tensor_subclass_deepcopy",  # test_serialization
    "TestOldSerialization.test_save_different_dtype_unallocated",  # test_serialization
    "TestSubclassSerialization.test_tensor_subclass_getstate_overwrite",  # test_serialization
    "TestSerialization.test_save_different_dtype_unallocated",  # test_serialization
    "TestSubclassSerialization.test_tensor_subclass_wrapper_serialization",  # test_serialization
    "TestScatterGatherCPU.test_scatter_reduce_sum_cpu_float32",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_mean_cpu_int16",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_sum_cpu_bfloat16",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_mean_cpu_complex64",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_sum_cpu_float64",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_mean_cpu_bfloat16",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_sum_cpu_complex128",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_mean_cpu_float16",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter__reductions_cpu_float16",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_mean_cpu_int32",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_sum_cpu_int32",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_mean_cpu_float32",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_mean_cpu_int64",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_sum_cpu_float16",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter__reductions_cpu_float32",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_mean_cpu_uint8",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_sum_cpu_uint8",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_sum_cpu_int16",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_mean_cpu_complex128",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_sum_cpu_int8",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_mean_cpu_float64",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_sum_cpu_complex64",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_sum_cpu_int64",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter_reduce_mean_cpu_int8",  # test_scatter_gather_ops
    "TestScatterGatherCPU.test_scatter__reductions_cpu_complex64",  # test_scatter_gather_ops
    "TestCxxPytree.test_pytree_serialize_spec8",  # test_pytree
    "TestGenericPytree.test_flatten_unflatten_namedtuple_py",  # test_pytree
    "TestCxxPytree.test_pytree_serialize_spec9",  # test_pytree
    "TestCxxPytree.test_pytree_serialize_spec3",  # test_pytree
    "TestGenericPytree.test_flatten_unflatten_deque_py",  # test_pytree
    "TestGenericPytree.test_flatten_unflatten_deque_cxx",  # test_pytree
    "TestCxxPytree.test_pytree_serialize_spec2",  # test_pytree
    "TestCxxPytree.test_pytree_serialize_spec5",  # test_pytree
    "TestCxxPytree.test_pytree_serialize_namedtuple",  # test_pytree
    "TestCxxPytree.test_pytree_serialize_spec0",  # test_pytree
    "TestCxxPytree.test_pytree_serialize_spec6",  # test_pytree
    "TestCxxPytree.test_pytree_serialize_spec4",  # test_pytree
    "TestCxxPytree.test_pytree_serialize_spec7",  # test_pytree
    "TestCxxPytree.test_pytree_serialize_spec1",  # test_pytree
    "TestPythonPytree.test_treespec_equality",  # test_pytree
    "TestOutDtypeOp.test_out_dtype_non_op_overload",  # test_out_dtype_op
    "TestOutDtypeOp.test_out_dtype_wrong_output",  # test_out_dtype_op
    "TestNumPyInteropCPU.test_numpy_non_writeable_cpu",  # test_numpy_interop
    "TestNN.test_Sequential_append",  # test_nn
    "TestNNDeviceTypeCPU.test_upsamplingBiMode2d_antialias_True_align_corners_False_mode_bicubic_memory_format0_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_nll_loss_all_ignored_cpu",  # test_nn
    "TestNN.test_ParameterList_replication",  # test_nn
    "TestNNDeviceTypeCPU.test_CTCLoss_no_batch_dim_reduction_none_use_module_form_False_cpu",  # test_nn
    "TestNN.test_interpolate_buffer_overflow",  # test_nn
    "TestNNDeviceTypeCPU.test_CTCLoss_no_batch_dim_reduction_mean_use_module_form_False_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_hardsigmoid_grad_cpu",  # test_nn
    "TestNN.test_batchnorm_raises_error_if_running_var_or_running_mean_have_forward_grad",  # test_nn
    "TestNNDeviceTypeCPU.test_nll_loss_byte_target_matches_long_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_module_to_empty_cpu_float32",  # test_nn
    "TestNNDeviceTypeCPU.test_nll_loss_empty_tensor_reduction_none_cpu",  # test_nn
    "TestNN.test_Sequential_extend",  # test_nn
    "TestNN.test_overwrite_module_params_on_conversion",  # test_nn
    "TestNN.test_ModuleList",  # test_nn
    "TestNNDeviceTypeCPU.test_hardswish_grad_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_threshold_inplace_overlap_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_module_to_empty_cpu_float64",  # test_nn
    "TestNNDeviceTypeCPU.test_upsamplingBiMode2d_antialias_True_align_corners_True_mode_bicubic_memory_format1_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_upsamplingBiMode2d_antialias_True_align_corners_True_mode_bicubic_memory_format0_cpu",  # test_nn
    "TestNN.test_Sequential_imul",  # test_nn
    "TestNN.test_upsampling_bfloat16",  # test_nn
    "TestNNDeviceTypeCPU.test_triplet_margin_with_distance_loss_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_CTCLoss_no_batch_dim_reduction_sum_use_module_form_False_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_nll_loss_empty_tensor_reduction_sum_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_upsamplingTrilinear3d_align_corners_False_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_upsamplingBiMode2d_antialias_True_align_corners_False_mode_bicubic_memory_format1_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_batchnorm_grad_cpu",  # test_nn
    "TestNN.test_interpolate",  # test_nn
    "TestNN.test_register_state_dict_pre_hook",  # test_nn
    "TestNNDeviceTypeCPU.test_upsamplingTrilinear3d_align_corners_True_cpu",  # test_nn
    "TestNN.test_fb_fc_packed",  # test_nn
    "TestFusionEval.test_fuse_module_eval_numerics",  # test_nn
    "TestNNDeviceTypeCPU.test_invalid_reduction_strings_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_nll_loss_total_weight_is_zero_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_nll_loss_empty_tensor_reduction_mean_cpu",  # test_nn
    "TestNN.test_register_state_dict_pre_hook_lazy_module",  # test_nn
    "TestNN.test_ParameterDict_replication",  # test_nn
    "TestNN.test_Sequential_iadd",  # test_nn
    "TestNN.test_upsamplingLinear1d",  # test_nn
    "TestNativeFunctions.test_symintlist_error_with_overload",  # test_native_functions
    "TestNativeFunctions.test_vararg_symintlist_error",  # test_native_functions
    "TestNativeFunctions.test_optional_intlist_invalid",  # test_native_functions
    "TestNativeFunctions.test_symintlist_error",  # test_native_functions
    "TestNativeFunctions.test_optional_floatlist_invalid",  # test_native_functions
    "TestMultiprocessing.test_empty_shared",  # test_multiprocessing
    "TestMultiprocessing.test_inherit_tensor",  # test_multiprocessing
    "TestMultiprocessing.test_is_shared",  # test_multiprocessing
    "TestMultiprocessing.test_fs_is_shared",  # test_multiprocessing
    "TestMkldnnCPU.test_resnext50_32x4d_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_resnet18_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_add_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_linear_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_prelu_bf16_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_prelu_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_batch_norm_2d_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_mul_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_conv1d_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_linear_lowp_cpu_float16",  # test_mkldnn
    "TestMkldnnCPU.test_sigmoid_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_conv3d_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_reshape_blocked_format_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_copy_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_tanh_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_conv2d_cpu",  # test_mkldnn
    "TestMkldnnCPU.test_batch_norm_3d_cpu",  # test_mkldnn
    "TestFunctionalAutogradBenchmark.test_fast_tasks",  # test_functional_autograd_benchmark
    "TestFunctionSchema.test_serialize_and_deserialize",  # test_function_schema
    "FakeTensorOperatorInvariants.test_like_ops",  # test_fake_tensor
    "FakeTensorConverterTest.test_memoized_conversion_from_meta",  # test_fake_tensor
    "FakeTensorOperatorInvariants.test_non_kwarg_only_device",  # test_fake_tensor
    "FakeTensorOperatorInvariants.test_tensor_constructors_all_have_kwarg_device",  # test_fake_tensor
    "TestExpandedWeightModuleCPU.test_Conv1d_reflect_stride2_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv2d_zeros_stride2_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Embedding_discontiguous_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_conv1d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_instance_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightModuleCPU.test_Conv2d_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv2d_reflect_stride2_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv2d_replicate_stride2_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_conv3d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_instance_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightHelperFunctionCPU.test_unpack_expanded_weight_or_tensor_with_custom_function_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv1d_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightHelperFunctionCPU.test_unpack_expanded_weight_or_tensor_cpu",  # test_expanded_weights
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_conv1d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightModuleCPU.test_Conv1d_pad1_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_per_sample_api_compute_batch_size_not_pytreeable_cpu",  # test_expanded_weights
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_linear_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_group_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_group_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_conv3d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightModuleCPU.test_Conv2d_dilated_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_conv2d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightModuleCPU.test_Conv1d_replicate_stride2_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv3d_replicate_stride2_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv1d_pad2size1_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv1d_stride_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_conv2d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_embedding_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightModuleCPU.test_per_sample_api_compute_batch_size_cpu",  # test_expanded_weights
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_layer_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_linear_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_embedding_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_linear_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightModuleCPU.test_Conv2d_padding_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv3d_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv3d_zeros_stride2_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv3d_1x1x1_no_bias_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv3d_no_bias_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightHelperFunctionCPU.test_set_grad_sample_if_exists_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Linear_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_group_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightModuleCPU.test_Linear_no_bias_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Embedding_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_layer_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_conv3d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_embedding_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_instance_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_conv1d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightModuleCPU.test_Conv1d_circular_stride2_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv1d_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv1d_pad1size1_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv2d_strided_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv3d_circular_stride2_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv3d_stride_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv2d_no_bias_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightModuleCPU.test_Conv3d_stride_padding_multiple_inputs_cpu",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_layer_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightModuleCPU.test_Conv1d_zeros_stride2_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_conv2d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightModuleCPU.test_Conv2d_circular_stride2_pad2_multiple_inputs_cpu",  # test_expanded_weights
    "TestTorchDlPackCPU.test_dlpack_export_non_strided_cpu",  # test_dlpack
    "TestIterDataPipeGraphFastForward.test_simple_snapshot_custom_self_next",  # test_datapipe
    "TestIterDataPipeGraphFastForward.test_simple_snapshot_custom_non_generator",  # test_datapipe
    "TestGraph.test_traverse_circular_datapipe",  # test_datapipe
    "TestGraph.test_traverse_unhashable_datapipe",  # test_datapipe
    "TestCppExtensionJIT.test_cpp_frontend_module_has_same_output_as_python",  # test_cpp_extensions_jit
    "TestPoolingNN.test_quantized_max_pool1d_empty_kernel",  # nn/test_pooling
    "TestPoolingNN.test_MaxUnpool2d_output_size",  # nn/test_pooling
    "TestNNParametrization.test_deepcopy_after_parametrization",  # nn/test_parametrization
    "TestNNParametrization.test_new_spectral_norm_dim",  # nn/test_parametrization
    "TestNNParametrization.test_new_spectral_norm_forward",  # nn/test_parametrization
    "TestNNParametrization.test_weight_norm_state_dict_compat",  # nn/test_parametrization
    "TestNNParametrization.test_new_spectral_norm",  # nn/test_parametrization
    "TestNNParametrization.test_weight_norm_deepcopy",  # nn/test_parametrization
    "PackedSequenceTest.test_to",  # nn/test_packed_sequence
    "PackedSequenceTest.test_type_casts",  # nn/test_packed_sequence
    "PackedSequenceTest.test_pack_sequence",  # nn/test_packed_sequence
    "PackedSequenceTest.test_total_length",  # nn/test_packed_sequence
    "TestModuleHooks.test_forward_pre_hooks_named_tuple_True",  # nn/test_module_hooks
    "TestModuleHooks.test_full_backward_pre_hooks_named_tuple_True",  # nn/test_module_hooks
    "TestModuleHookNN.test_hook_submodule_registration",  # nn/test_module_hooks
    "TestModuleHooks.test_forward_hooks_named_tuple_False",  # nn/test_module_hooks
    "TestModuleHooks.test_full_backward_hooks_named_tuple_False",  # nn/test_module_hooks
    "TestModuleHooks.test_forward_hooks_named_tuple_True",  # nn/test_module_hooks
    "TestStateDictHooks.test_pickled_hook",  # nn/test_module_hooks
    "TestModuleHookNN.test_hook_inplace",  # nn/test_module_hooks
    "TestModuleGlobalHooks.test_module_backward_global_hook_writeable",  # nn/test_module_hooks
    "TestModuleHookNN.test_hook_buffer_registration",  # nn/test_module_hooks
    "TestModuleHooks.test_full_backward_hooks_named_tuple_True",  # nn/test_module_hooks
    "TestModuleHookNN.test_hook_no_requires_grad",  # nn/test_module_hooks
    "TestModuleHookNN.test_hook_backward_writeable",  # nn/test_module_hooks
    "TestModuleHooks.test_forward_pre_hooks_named_tuple_False",  # nn/test_module_hooks
    "TestModuleHookNN.test_hook_parameter_registration",  # nn/test_module_hooks
    "TestModuleHooks.test_full_backward_pre_hooks_named_tuple_False",  # nn/test_module_hooks
    "TestModuleHookNN.test_hook_cpp",  # nn/test_module_hooks
    "TestStateDictHooks.test_load_state_dict_pre_hook",  # nn/test_module_hooks
    "TestModuleHookNN.test_hook_invalid_outputs",  # nn/test_module_hooks
    "TestModuleHookNN.test_backward_hooks_interaction",  # nn/test_module_hooks
    "TestModuleHookNN.test_hooks",  # nn/test_module_hooks
    "TestModuleHookNN.test_hook_last_arg_requires_grad",  # nn/test_module_hooks
    "TestModuleGlobalHooks.test_module_global_hook_invalid_outputs",  # nn/test_module_hooks
    "TestLazyModules.test_lazy_module_parameter",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_batchnorm2d_state",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv3d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv_transposed1d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv2d",  # nn/test_lazy_modules
    "TestLazyModules.test_optimizer_pass",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_instancenorm3d_state",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_batchnorm3d_state",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv_transpose1d_pickle",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_instancenorm2d",  # nn/test_lazy_modules
    "TestLazyModules.test_invalid_functions",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_instancenorm2d_state",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv3d_pickle",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_batchnorm2d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_instancenorm1d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_batchnorm1d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_instancenorm1d_state",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv_transpose3d_pickle",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_instancenorm3d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_batchnorm3d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv2d_pickle",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv1d_pickle",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_module_jit_buffer",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv1d",  # nn/test_lazy_modules
    "TestLazyModules.test_linear",  # nn/test_lazy_modules
    "TestLazyModules.test_materialize_dtype",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_module_buffer",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_batchnorm1d_state",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_module_jit_param",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_batchnorm_with_dict_input",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv_transpose2d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv_transpose2d_pickle",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv_transpose3d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_linear_pickle",  # nn/test_lazy_modules
    "TestNNInit.test_kaiming_normal_errors_on_inputs_smaller_than_2d",  # nn/test_init
    "TestNNInit.test_xavier_normal_errors_on_inputs_smaller_than_2d",  # nn/test_init
    "TestNNInit.test_ones_and_zeros",  # nn/test_init
    "TestNNInit.test_eye",  # nn/test_init
    "TestNNInit.test_kaiming_uniform_errors_on_inputs_smaller_than_2d",  # nn/test_init
    "TestNNInit.test_dirac_only_works_on_3_4_5d_inputs",  # nn/test_init
    "TestNNInit.test_sparse_only_works_on_2d_inputs",  # nn/test_init
    "TestNNInit.test_constant",  # nn/test_init
    "TestNNInit.test_xavier_uniform_errors_on_inputs_smaller_than_2d",  # nn/test_init
    "TestNNInit.test_xavier_uniform",  # nn/test_init
    "TestNNInit.test_eye_only_works_on_2d_inputs",  # nn/test_init
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_bfloat16_cpu_int32_int64",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int32_int64_bfloat16",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_EmbeddingBag_per_sample_weights_and_no_offsets_cpu_int32_float32",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int32_int64_float32",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_dimension_errors_cpu",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_EmbeddingBag_per_sample_weights_and_no_offsets_cpu_int64_float64",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_bfloat16_cpu_int64_int64",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int64_int32_bfloat16",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int64_int64_float32",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_half_cpu_int64_int64",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int64_int32_float16",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int64_int64_bfloat16",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int32_int32_float32",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_EmbeddingBag_per_sample_weights_and_no_offsets_cpu_int64_float32",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_EmbeddingBag_per_sample_weights_and_no_offsets_cpu_int32_float64",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int64_int64_float64",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_half_cpu_int32_int32",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int32_int32_float16",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int32_int32_float64",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_half_cpu_int64_int32",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int64_int64_float16",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int64_int32_float64",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int32_int64_float64",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int64_int32_float32",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_half_cpu_int32_int64",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_bfloat16_cpu_int64_int32",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_bfloat16_cpu_int32_int32",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int32_int64_float16",  # nn/test_embedding
    "TestEmbeddingNNDeviceTypeCPU.test_embedding_bag_device_cpu_int32_int32_bfloat16",  # nn/test_embedding
    "TestDropoutNN.test_invalid_dropout_p",  # nn/test_dropout
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel3d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel2d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNN.test_Conv2d_missing_argument",  # nn/test_convolution
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel1d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel2d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel3d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel3d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel2d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel2d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel3d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel2d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel3d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel2d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel3d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel2d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel2d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel1d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch1d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel2d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel3d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel1d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel2d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch3d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch1d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel3d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch1d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel3d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel3d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel1d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch3d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel3d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel2d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel1d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel2d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel3d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch2d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch1d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel2d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel3d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel3d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel1d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel1d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel3d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel1d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel2d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel2d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel2d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel2d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel3d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch2d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel2d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel1d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel3d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel1d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel3d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel1d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel3d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel2d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel3d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel1d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel1d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel1d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel1d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel2d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel3d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch1d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel2d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel1d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch3d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel3d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch3d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch1d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel2d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch2d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel1d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel1d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel2d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel1d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel2d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel2d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel3d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel1d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel2d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch3d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel1d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel3d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel1d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel2d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch3d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch2d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch1d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch3d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch2d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch2d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch2d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel1d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel1d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel1d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel1d_has_bias_False_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel2d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel2d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel3d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel2d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel2d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel3d_has_bias_True_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel3d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel3d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel3d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel3d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel1d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel1d_has_bias_True_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch3d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_batch_channel3d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel1d_has_bias_True_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel2d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel3d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_channel1d_has_bias_False_strided_True_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_mkldnn_empty_channel1d_has_bias_True_strided_False_contiguous_False_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch1d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch_channel1d_has_bias_False_strided_False_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestConvolutionNNDeviceTypeCPU.test_conv_backend_empty_batch2d_has_bias_False_strided_True_contiguous_True_cpu",  # nn/test_convolution  # noqa: B950
    "TestDistributionShapes.test_mixture_same_family_shape",  # distributions/test_distributions
    "TestFunctors.test_cat_transform",  # distributions/test_distributions
    "TestFunctors.test_cat_transform_non_uniform",  # distributions/test_distributions
    "TestRearrange.test_ellipsis_ops",  # functorch/test_rearrange
    "TestRearrange.test_concatenations_and_stacking",  # functorch/test_rearrange
    "TestRearrange.test_rearrange_permutations",  # functorch/test_rearrange
    "TestRearrange.test_collapsed_ellipsis_errors_out",  # functorch/test_rearrange
    "TestRearrange.test_rearrange_consistency",  # functorch/test_rearrange
    "TestRearrange.test_unsqueeze",  # functorch/test_rearrange
    "TestRearrange.test_squeeze",  # functorch/test_rearrange
    "TestMin.test_manual_stuff",  # functorch/test_dims
    "TestMin.test_dim_args",  # functorch/test_dims
    "TestMin.test_dims_with_size",  # functorch/test_dims
    "TestMin.test_functorch",  # functorch/test_dims
    "TestMin.test_eq",  # functorch/test_dims
    "TestMinFunctorchOnly.test_adapt",  # functorch/test_dims
    "TestMinFunctorchOnly.test_monkey",  # functorch/test_dims
    "TestMinFunctorchOnly.test_network",  # functorch/test_dims
    "TestMin.test_doc",  # functorch/test_dims
    "TestMin.test_c",  # functorch/test_dims
    "TestMinFunctorchOnly.test_index_placement",  # functorch/test_dims
    "TestMinFunctorchOnly.test_dims_with_size",  # functorch/test_dims
    "TestMinFunctorchOnly.test_with_dims_split",  # functorch/test_dims
    "TestMin.test_index",  # functorch/test_dims
    "TestMinFunctorchOnly.test_softmax_split",  # functorch/test_dims
    "TestMin.test_mm_fuse",  # functorch/test_dims
    "TestMinFunctorchOnly.test_simple",  # functorch/test_dims
    "TestMin.test_seg",  # functorch/test_dims
    "TestMinFunctorchOnly.test_doc",  # functorch/test_dims
    "TestMin.test_attn",  # functorch/test_dims
    "TestMin.test_mask",  # functorch/test_dims
    "TestMinFunctorchOnly.test_mm",  # functorch/test_dims
    "TestMin.test_index_placement",  # functorch/test_dims
    "TestMin.test_dir",  # functorch/test_dims
    "TestMinFunctorchOnly.test_hello",  # functorch/test_dims
    "TestMin.test_network",  # functorch/test_dims
    "TestMin.test_permute_orig",  # functorch/test_dims
    "TestMinFunctorchOnly.test_parse",  # functorch/test_dims
    "TestMinFunctorchOnly.test_order_keyword",  # functorch/test_dims
    "TestMinFunctorchOnly.test_seg",  # functorch/test_dims
    "TestMin.test_diag",  # functorch/test_dims
    "TestMin.test_monkey",  # functorch/test_dims
    "TestMinFunctorchOnly.test_embed",  # functorch/test_dims
    "TestMinFunctorchOnly.test_stack",  # functorch/test_dims
    "TestMin.test_hello",  # functorch/test_dims
    "TestMin.test_expand",  # functorch/test_dims
    "TestMin.test_time_mm_fuse",  # functorch/test_dims
    "TestMinFunctorchOnly.test_compare_dims",  # functorch/test_dims
    "TestMinFunctorchOnly.test_manual_stuff",  # functorch/test_dims
    "TestMinFunctorchOnly.test_max",  # functorch/test_dims
    "TestMin.test_adapt",  # functorch/test_dims
    "TestMinFunctorchOnly.test_dir",  # functorch/test_dims
    "TestMinFunctorchOnly.test_order",  # functorch/test_dims
    "TestMin.test_mm",  # functorch/test_dims
    "TestMinFunctorchOnly.test_index",  # functorch/test_dims
    "TestMinFunctorchOnly.test_permute_orig",  # functorch/test_dims
    "TestMinFunctorchOnly.test_functorch",  # functorch/test_dims
    "TestMinFunctorchOnly.test_dim_args",  # functorch/test_dims
    "TestMinFunctorchOnly.test_eq",  # functorch/test_dims
    "TestMinFunctorchOnly.test_mask",  # functorch/test_dims
    "TestMin.test_big_split",  # functorch/test_dims
    "TestMinFunctorchOnly.test_attn",  # functorch/test_dims
    "TestMin.test_max",  # functorch/test_dims
    "TestMin.test_compare_dims",  # functorch/test_dims
    "TestMin.test_parse",  # functorch/test_dims
    "TestMinFunctorchOnly.test_big_split",  # functorch/test_dims
    "TestMin.test_simple",  # functorch/test_dims
    "TestMinFunctorchOnly.test_mm_fuse",  # functorch/test_dims
    "TestMin.test_order_keyword",  # functorch/test_dims
    "TestMin.test_inplace",  # functorch/test_dims
    "TestMin.test_with_dims_split",  # functorch/test_dims
    "TestMin.test_softmax_split",  # functorch/test_dims
    "TestMin.test_order",  # functorch/test_dims
    "TestMinFunctorchOnly.test_c",  # functorch/test_dims
    "TestMinFunctorchOnly.test_inplace",  # functorch/test_dims
    "TestMin.test_embed",  # functorch/test_dims
    "TestMinFunctorchOnly.test_diag",  # functorch/test_dims
    "TestMinFunctorchOnly.test_expand",  # functorch/test_dims
    "TestMin.test_stack",  # functorch/test_dims
    "TestControlFlowTraced.test_tracing_map_symbolic_simple",  # functorch/test_control_flow
    "TestControlFlow.test_map_list_in_out",  # functorch/test_control_flow
    "TestControlFlowTraced.test_nested_cond_map_cond_symbolic",  # functorch/test_control_flow
    "TestControlFlowTraced.test_tracing_map_autograd_aot_functionalized",  # functorch/test_control_flow
    "TestControlFlow.test_map_autograd_nested_list",  # functorch/test_control_flow
    "TestControlFlowTraced.test_tracing_map_autograd_symbolic_simple",  # functorch/test_control_flow
    "TestControlFlowTraced.test_tracing_map_real",  # functorch/test_control_flow
    "TestControlFlowTraced.test_map_functionalized_aot_func",  # functorch/test_control_flow
    "TestControlFlowTraced.test_tracing_map_symbolic_list",  # functorch/test_control_flow
    "TestControlFlowTraced.test_tracing_map_symbolic_dict",  # functorch/test_control_flow
    "TestControlFlow.test_map_dict_in_out",  # functorch/test_control_flow
    "TestControlFlowTraced.test_tracing_map_autograd_symbolic_list",  # functorch/test_control_flow
    "TestControlFlowTraced.test_tracing_map_autograd_symbolic_dict",  # functorch/test_control_flow
    "TestControlFlow.test_map_autograd_no_grad_output",  # functorch/test_control_flow
    "TestControlFlowTraced.test_map_functionalized",  # functorch/test_control_flow
    "TestControlFlowTraced.test_nested_map_cond_symbolic",  # functorch/test_control_flow
    "TestControlFlowTraced.test_nested_map_cond_real",  # functorch/test_control_flow
    "TestMetaKernel.test_addmm_invalid_dtype",  # lazy/test_meta_kernel
    "TestVerifyCorrectness.test_incorrect_verify_true",  # dynamo/test_verify_correctness
    "TestVerifyCorrectness.test_torchscript",  # dynamo/test_verify_correctness
    "UnspecTests.test_conv1d_symint_padding",  # dynamo/test_unspec
    "UnspecTests.test_no_recompiles",  # dynamo/test_unspec
    "UnspecTests.test_no_recompilations",  # dynamo/test_unspec
    "UnspecTests.test_builtin_max_min",  # dynamo/test_unspec
    "UnspecTests.test_propagate_dynamic_dim",  # dynamo/test_unspec
    "UnspecTests.test_isinstance_symint",  # dynamo/test_unspec
    "UnspecTests.test_mark_01_dynamic",  # dynamo/test_unspec
    "UnspecTests.test_use_and_specialize",  # dynamo/test_unspec
    "TraceRuleTests.test_skipfiles_inlinelist",  # dynamo/test_trace_rules
    "SubGraphTests.test_dynamic_order_dependence",  # dynamo/test_subgraphs
    "SubGraphTests.test_no_graph_break_on_item",  # dynamo/test_subgraphs
    "SubGraphTests.test_dynamic_zero_inference",  # dynamo/test_subgraphs
    "SubGraphTests.test_enumerate_not_break_graph",  # dynamo/test_subgraphs
    "SubGraphTests.test_dynamic_duck_size",  # dynamo/test_subgraphs
    "SubGraphTests.test_dynamic_getitem",  # dynamo/test_subgraphs
    "SubGraphTests.test_dynamic_kwarg",  # dynamo/test_subgraphs
    "SubclassTests.test_compile_with_functionalization",  # dynamo/test_subclasses
    "TestNestedTensor.test_binary_recompiles",  # dynamo/test_subclasses
    "SubclassTests.test_wrapper_subclass_with_differently_sized_inner_tensor",  # dynamo/test_subclasses
    "SubclassTests.test_compile_higher_order_with_functionalization",  # dynamo/test_subclasses
    "SubclassTests.test_wrapper_subclass_guards_on_inner_tensor",  # dynamo/test_subclasses
    "TestNestedTensor.test_unbind",  # dynamo/test_subclasses
    "SubclassTests.test_torch_function_state_guards",  # dynamo/test_subclasses
    "SubclassTests.test_wrapper_subclass_with_same_sized_inner_tensor",  # dynamo/test_subclasses
    "SkipNonTensorTests.test_do_not_skip_side_effects",  # dynamo/test_skip_non_tensor
    "SkipNonTensorTests.test_add_tensor_dict",  # dynamo/test_skip_non_tensor
    "SkipNonTensorTests.test_add_tensor2",  # dynamo/test_skip_non_tensor
    "SkipNonTensorTests.test_add_tensor1",  # dynamo/test_skip_non_tensor
    "SkipNonTensorTests.test_add_tensor_list",  # dynamo/test_skip_non_tensor
    "SkipNonTensorTests.test_recursive_list",  # dynamo/test_skip_non_tensor
    "ReproTests.test_function_in_skipfiles",  # dynamo/test_repros
    "ReproTests.test_negative_shape_guard",  # dynamo/test_repros
    "ReproTests.test_dynamic_shapes_implicit_guard",  # dynamo/test_repros
    "ReproTests.test_reformer_min_chunk_len",  # dynamo/test_repros
    "ReproTests.test_jit_trace_errors",  # dynamo/test_repros
    "ReproTests.test_chunk_reformer_ff",  # dynamo/test_repros
    "ReproTests.test_module_in_skipfiles",  # dynamo/test_repros
    "ReproTests.test_do_paste_mask",  # dynamo/test_repros
    "ReproTests.test_boxes_len",  # dynamo/test_repros
    "ReproTests.test_convert_boxes_to_pooler_format",  # dynamo/test_repros
    "ReproTests.test_list_self_reference",  # dynamo/test_repros
    "ReproTests.test_validate_model_kwargs",  # dynamo/test_repros
    "ReproTests.test_create_rand_mask_from_inputs",  # dynamo/test_repros
    "ReproTests.test_rewrite_assert_with_non_string_msg",  # dynamo/test_repros
    "ReproTests.test_empty_list_contains_with_jump",  # dynamo/test_repros
    "ReproTests.test_issue175",  # dynamo/test_repros
    "ReproTests.test_restricted_list_subclass1",  # dynamo/test_repros
    "ReproTests.test_add_sub_alpha_out",  # dynamo/test_repros
    "ReproTests.test_dynamic_shapes_float_guard",  # dynamo/test_repros
    "ReproTests.test_list_aliasing",  # dynamo/test_repros
    "ReproTests.test_addr_alpha_beta_out",  # dynamo/test_repros
    "ReproTests.test_merge_criteria_processor_list2",  # dynamo/test_repros
    "ReproTests.test_reformer_eval",  # dynamo/test_repros
    "ReproTests.test_longformer_chunk",  # dynamo/test_repros
    "ReproTests.test_functools_wraps",  # dynamo/test_repros
    "ReproTests.test_tensor_data_kwarg",  # dynamo/test_repros
    "ReproTests.test_recursive_map",  # dynamo/test_repros
    "ReproTests.test_reformer_sorting",  # dynamo/test_repros
    "ReproTests.test_size_typematch",  # dynamo/test_repros
    "ReproTests.test_threading_local",  # dynamo/test_repros
    "ReproTests.test_merge_criteria_processor_list1",  # dynamo/test_repros
    "ReproTests.test_relative_import",  # dynamo/test_repros
    "ReproTests.test_rewrite_assert_noop",  # dynamo/test_repros
    "ReproTests.test_seq_append_list",  # dynamo/test_repros
    "ReproTests.test_relative_import_no_modulename",  # dynamo/test_repros
    "ReproTests.test_multi_import",  # dynamo/test_repros
    "ReproTests.test_rewrite_assert_with_msg",  # dynamo/test_repros
    "ReproTests.test_hf_t5_forward",  # dynamo/test_repros
    "ReproTests.test_hf_xsoftmax_training",  # dynamo/test_repros
    "ReproTests.test_numpy_not_ndarray_recompiles",  # dynamo/test_repros
    "RecompileTests.test_automatic_dynamic_reduce_recompiles",  # dynamo/test_recompiles
    "RecompileTests.test_automatic_dynamic_tensor_scalar_change",  # dynamo/test_recompiles
    "RecompileTests.test_recompiles_true_false_flop",  # dynamo/test_recompiles
    "RecompileTests.test_dynamic_shape_parameter_recompile",  # dynamo/test_recompiles
    "RecompileTests.test_aliasing_guard_failures_with_globals",  # dynamo/test_recompiles
    "RecompileUxTests.test_verbose_tensor_check",  # dynamo/test_recompile_ux
    "RecompileUxTests.test_mismatched_type",  # dynamo/test_recompile_ux
    "TestPythonAutograd.test_backwards2",  # dynamo/test_python_autograd
    "TestPythonAutograd.test_forwards1",  # dynamo/test_python_autograd
    "TestPythonAutograd.test_split",  # dynamo/test_python_autograd
    "TestPythonAutograd.test_forwards2",  # dynamo/test_python_autograd
    "DynamoProfilerTests.test_dynamo_timed_profiling_isolated",  # dynamo/test_profiler
    "DynamoProfilerTests.test_profiler_cache_lookup",  # dynamo/test_profiler
    "DynamoProfilerTests.test_profiler_dynamo_compiled_region",  # dynamo/test_profiler
    "OptimizerTests.test_rmsprop",  # dynamo/test_optimizers
    "OptimizerTests.test_rprop",  # dynamo/test_optimizers
    "OptimizerTests.test_nadam",  # dynamo/test_optimizers
    "OptimizerTests.test_adagrad",  # dynamo/test_optimizers
    "OptimizerTests.test_adamax",  # dynamo/test_optimizers
    "OptimizerTests.test_adam",  # dynamo/test_optimizers
    "OptimizerTests.test_asgd",  # dynamo/test_optimizers
    "End2EndTests.test_init_group",  # dynamo/test_optimizers
    "OptimizerTests.test_adamw",  # dynamo/test_optimizers
    "OptimizerTests.test_sgd",  # dynamo/test_optimizers
    "OptimizedModuleTest.test_module_dict_iter_name",  # dynamo/test_modules
    "NNModuleTests.test_lazy_module1",  # dynamo/test_modules
    "OptimizedModuleTest.test_composition_with_opt_mod",  # dynamo/test_modules
    "OptimizedModuleTest.test_backward_hooks",  # dynamo/test_modules
    "OptimizedModuleTest.test_nn_module",  # dynamo/test_modules
    "OptimizedModuleTest.test_to",  # dynamo/test_modules
    "OptimizedModuleTest.test_hooks_outer",  # dynamo/test_modules
    "NNModuleTests.test_lazy_module5",  # dynamo/test_modules
    "OptimizedModuleTest.test_hooks_inner",  # dynamo/test_modules
    "NNModuleTests.test_lazy_module_no_cls_to_become",  # dynamo/test_modules
    "OptimizedModuleTest.test_cache_size_limit_on_guarded_nn_modules",  # dynamo/test_modules
    "NNModuleTests.test_self_mutating1",  # dynamo/test_modules
    "NNModuleTests.test_unsupportedmodule",  # dynamo/test_modules
    "NNModuleTests.test_unsupportedmethod",  # dynamo/test_modules
    "OptimizedModuleTest.test_hooks_skip_guards",  # dynamo/test_modules
    "NNModuleTests.test_lazy_module2",  # dynamo/test_modules
    "OptimizedModuleTest.test_no_recompile_on_nn_guarded_modules",  # dynamo/test_modules
    "NNModuleTests.test_lazy_module4",  # dynamo/test_modules
    "NNModuleTests.test_lazy_module6",  # dynamo/test_modules
    "MiscTests.test_tensor_dict3",  # dynamo/test_misc
    "MiscTests.test_with_builtin_type",  # dynamo/test_misc
    "MiscTests.test_tolist_scalar",  # dynamo/test_misc
    "MiscTests.test_raise_guard_full_constraint",  # dynamo/test_misc
    "MiscTests.test_setattr_mutation1",  # dynamo/test_misc
    "MiscTests.test_closure_out_of_scope_cell",  # dynamo/test_misc
    "MiscTests.test_size_input",  # dynamo/test_misc
    "MiscTests.test_inplace_view_on_graph_input",  # dynamo/test_misc
    "MiscTests.test_itertools_groupby_pure_python_key_func",  # dynamo/test_misc
    "MiscTests.test_frozenset_torch_func_contains",  # dynamo/test_misc
    "MiscTests.test_nested_closure",  # dynamo/test_misc
    "MiscTests.test_tolist_1d",  # dynamo/test_misc
    "MiscTests.test_any_all_symnode",  # dynamo/test_misc
    "MiscTests.test_iter_set",  # dynamo/test_misc
    "MiscTests.test_dict_order_keys_tensors",  # dynamo/test_misc
    "MiscTests.test_mark_static",  # dynamo/test_misc
    "MiscTests.test_backend_match_guard",  # dynamo/test_misc
    "MiscTests.test_tolist_0d",  # dynamo/test_misc
    "MiscTests.test_str_format_assert2",  # dynamo/test_misc
    "MiscTests.test_typing_typevar",  # dynamo/test_misc
    "MiscTests.test_raise_guard_partial_constraint_no_graph_break",  # dynamo/test_misc
    "MiscTests.test_get_cache_entry",  # dynamo/test_misc
    "MiscTests.test_nested_optimize_decorator",  # dynamo/test_misc
    "MiscTests.test_nested_optimize_run",  # dynamo/test_misc
    "MiscTests.test_release_input_memory",  # dynamo/test_misc
    "MiscTests.test_inplace_param_update",  # dynamo/test_misc
    "MiscTests.test_numpy_recompilation_scalar",  # dynamo/test_misc
    "MiscTests.test_inline_closure_not_loaded_by_parent",  # dynamo/test_misc
    "MiscTests.test_tracing_nested_py_tree_dicts",  # dynamo/test_misc
    "MiscTests.test_tracing_py_tree_tensor_subclass",  # dynamo/test_misc
    "MiscTests.test_user_getattribute",  # dynamo/test_misc
    "MiscTests.test_listcomp",  # dynamo/test_misc
    "MiscTests.test_yield_gen_and_from",  # dynamo/test_misc
    "MiscTests.test_guard_failure_fn",  # dynamo/test_misc
    "MiscTests.test_numpy_non_torch_dtype",  # dynamo/test_misc
    "MiscTests.test_numpy_with_builtin_type",  # dynamo/test_misc
    "MiscTests.test_py_guards_mark_dynamic",  # dynamo/test_misc
    "MiscTests.test_namedtuple2",  # dynamo/test_misc
    "MiscTests.test_itertools_accumulate_tensors_default_sum",  # dynamo/test_misc
    "MiscTests.test_guard_failure_fn2",  # dynamo/test_misc
    "MiscTests.test_cond_nested",  # dynamo/test_misc
    "MiscTests.test_nn_module_getattr",  # dynamo/test_misc
    "MiscTests.test_itertools_repeat",  # dynamo/test_misc
    "MiscTests.test_numpy_int_constant",  # dynamo/test_misc
    "MiscTests.test_torch_seed",  # dynamo/test_misc
    "MiscTests.test_callpacked",  # dynamo/test_misc
    "MiscTests.test_mandelbrot_numpy",  # dynamo/test_misc
    "MiscTests.test_pure_python_accumulate",  # dynamo/test_misc
    "MiscTests.test_release_module_memory",  # dynamo/test_misc
    "MiscTests.test_dunder_new_function_inlining",  # dynamo/test_misc
    "MiscTests.test_numpy_readonly",  # dynamo/test_misc
    "MiscTests.test_tolist_kd",  # dynamo/test_misc
    "MiscTests.test_deque_append_left",  # dynamo/test_misc
    "MiscTests.test_nested_optimize",  # dynamo/test_misc
    "MiscTests.test_tracing_nested_py_tree",  # dynamo/test_misc
    "MiscTests.test_repeat_interleave_graphbreaks",  # dynamo/test_misc
    "MiscTests.test_itertools_infinite_repeat",  # dynamo/test_misc
    "MiscTests.test_is_compiling",  # dynamo/test_misc
    "MiscTests.test_boolarg",  # dynamo/test_misc
    "MiscTests.test_compare_shapes_with_constant",  # dynamo/test_misc
    "MiscTests.test_itertools_groupby_pure_python_default_identify_func",  # dynamo/test_misc
    "MiscTests.test_no_raise_guard_partial_constraint",  # dynamo/test_misc
    "MiscTests.test_tracing_nested_py_tree_tuples",  # dynamo/test_misc
    "MiscTests.test_set_aliasing_recompiles",  # dynamo/test_misc
    "MiscTests.test_guard_failure_fn_tensor_iter",  # dynamo/test_misc
    "MiscTests.test_itertools_infinite_repeat_mutation",  # dynamo/test_misc
    "MiscTests.test_numpy_force",  # dynamo/test_misc
    "MiscTests.test_tracing_nested_py_tree_mixed_all",  # dynamo/test_misc
    "MiscTests.test_inference_mode",  # dynamo/test_misc
    "MiscTests.test_numpy_array_of_arrays",  # dynamo/test_misc
    "MiscTests.test_itertools_infinite_cycle",  # dynamo/test_misc
    "MiscTests.test_tracing_py_tree",  # dynamo/test_misc
    "MiscTests.test_tensor_build_list_unpack",  # dynamo/test_misc
    "MiscTests.test_simple_set_usage",  # dynamo/test_misc
    "MiscTests.test_no_raise_guard_partial_constraint_across_break",  # dynamo/test_misc
    "MiscTests.test_numpy_tolist",  # dynamo/test_misc
    "MiscTests.test_dictcomp",  # dynamo/test_misc
    "MiscTests.test_dict_order_keys",  # dynamo/test_misc
    "MiscTests.test_numpy_size_attr",  # dynamo/test_misc
    "MiscTests.test_add_to_set",  # dynamo/test_misc
    "MiscTests.test_tolist_kd_dynamic",  # dynamo/test_misc
    "MiscTests.test_raise_on_backend_error",  # dynamo/test_misc
    "MiscTests.test_nan",  # dynamo/test_misc
    "MiscTests.test_numpy_iter",  # dynamo/test_misc
    "MiscTests.test_return_nested_function",  # dynamo/test_misc
    "MiscTests.test_tensor_item_capture",  # dynamo/test_misc
    "MiscTests.test_guard_failure_fn_shape_control",  # dynamo/test_misc
    "MiscTests.test_tensor_dict1",  # dynamo/test_misc
    "MiscTests.test_deterministic_algorithms_mutated",  # dynamo/test_misc
    "MiscTests.test_dataclass_fields",  # dynamo/test_misc
    "MiscTests.test_dict_order_keys_modules",  # dynamo/test_misc
    "MiscTests.test_deque_input",  # dynamo/test_misc
    "MiscTests.test_itertools_accumulate_symint_default_sum",  # dynamo/test_misc
    "MiscTests.test_numpy_subdtype",  # dynamo/test_misc
    "MiscTests.test_numpy_torch_operators",  # dynamo/test_misc
    "MiscTests.test_dtypes_no_graphbreaks",  # dynamo/test_misc
    "MiscTests.test_tracing_tree_map_only",  # dynamo/test_misc
    "MiscTests.test_out_variants_with_resizing_on_graph_inputs",  # dynamo/test_misc
    "MiscTests.test_nested_closure_mutation",  # dynamo/test_misc
    "MiscTests.test_tensor_dict2",  # dynamo/test_misc
    "MiscTests.test_cond_side_effects",  # dynamo/test_misc
    "MiscTests.test_namedtuple1",  # dynamo/test_misc
    "MiscTests.test_yield_send_to_subgenerator_graph_break",  # dynamo/test_misc
    "MiscTests.test_recompile_on_global_state_change",  # dynamo/test_misc
    "MiscTests.test_type_copy",  # dynamo/test_misc
    "MiscTests.test_yield_from",  # dynamo/test_misc
    "MiscTests.test_intermediary_tensor_grad_access",  # dynamo/test_misc
    "MiscTests.test_release_scope_memory",  # dynamo/test_misc
    "InteropTests.test_vmap_in_graph",  # dynamo/test_interop
    "TestInputAttrTracking.test_complex_attr_access_without_graph_breaks",  # dynamo/test_input_attr_tracking
    "TestInputAttrTracking.test_tensor_property_on_tensor",  # dynamo/test_input_attr_tracking
    "TestInputAttrTracking.test_set_data_on_input_tensor",  # dynamo/test_input_attr_tracking
    "TestInputAttrTracking.test_tensor_property_assigned_on_tensor",  # dynamo/test_input_attr_tracking
    "TestInputAttrTracking.test_const_property_on_tensor",  # dynamo/test_input_attr_tracking
    "HooksTests.test_tensor_register_multiple_hooks_handles_in_list",  # dynamo/test_hooks
    "HooksTests.test_functools_arg_vary",  # dynamo/test_hooks
    "HooksTests.test_tensor_register_hook_in_graph_local",  # dynamo/test_hooks
    "HooksTests.test_post_acc_grad_hook",  # dynamo/test_hooks
    "HooksTests.test_tensor_register_multiple_hooks",  # dynamo/test_hooks
    "HooksTests.test_tensor_register_global_hook",  # dynamo/test_hooks
    "HigherOrderOpTests.test_wrap_kwarg_recompile",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_grad_non_tensor_input",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_grad_with_side_effect",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_fallback_on_python_primitives_output",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_cond_branches_no_arguments",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_free_const",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_multiple_invocation_in_dims",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_grad",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_illegal_op_graph_break",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_cond_pytree_operands",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_cond_branches_no_arguments_no_closure",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_side_effects",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_nested_tuple_output",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_output_with_dict",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_grad_two_tensor_has_aux",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_wrap_subgraph_name_is_valid",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_grad_freevar_python_scalar",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_free_tensor",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_two_inputs_tuple_in_dims",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_grad_with_graph_break",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_multiple_invocation_out_dims",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_map_pytree_return",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_capture_untracked_global_nested",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_multiple_outputs_diff_dims",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_vmap_source_fn_stack",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_kwargs",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_over_vmap_two_inputs",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_grad_disable_capture",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_grad_fn_with_kwargs",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_multiple_outputs",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_map_lowers_to_graph",  # dynamo/test_higher_order_ops
    "ActivationCheckpointingTests.test_cond_with_kwargs",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_map_source_fn_stack",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_capture_value_created_in_subgraph",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_grad_pytree",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_cond_pytree_operands_with_non_tensor_leaves",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_side_effect_in_body",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_cond_subgraph_name_is_valid",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_modules",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_over_vmap_captured",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_cond_source_fn_stack",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_map_subgraph_name_is_valid",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_pytree_inputs",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_disable_capture",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_multiple_outputs_out_dims_tuple",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_map_multi_return",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_grad_two_tensor_all_grad_has_aux",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_grad_has_aux",  # dynamo/test_higher_order_ops
    "HigherOrderOpTests.test_map_symint_input",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_two_inputs",  # dynamo/test_higher_order_ops
    "DefaultsTests.test_in_set_inplace",  # dynamo/test_functions
    "FunctionTests.test_default_dict_lambda",  # dynamo/test_functions
    "FunctionTests.test_default_dict_closure",  # dynamo/test_functions
    "FunctionTests.test_is_contiguous_frame_counts",  # dynamo/test_functions
    "FunctionTests.test_partials_as_input_partials_lambda",  # dynamo/test_functions
    "FunctionTests.test_default_dict",  # dynamo/test_functions
    "DefaultsTests.test_cast_tensor_single_elem",  # dynamo/test_functions
    "DefaultsTests.test_func_default_torch_args",  # dynamo/test_functions
    "FunctionTests.test_math_radians",  # dynamo/test_functions
    "DefaultsTests.test_compare_constant_and_tensor",  # dynamo/test_functions
    "DefaultsTests.test_in_set_would_fail_broadcast",  # dynamo/test_functions
    "FunctionTests.test_partials_as_input_partials_mod",  # dynamo/test_functions
    "ExportTests.test_cond_raise_user_error_on_non_list_operands",  # dynamo/test_export
    "ExportTests.test_export_dynamic_dim_range_constraint",  # dynamo/test_export
    "ExportTests.test_export_with_constant_free_function_and_class_method_multiarg",  # dynamo/test_export
    "ExportTests.test_export_with_constant_free_function_and_class_method",  # dynamo/test_export
    "ExportTests.test_cond_raise_user_error_on_unsupported_pred",  # dynamo/test_export
    "ExportTests.test_export_no_raise",  # dynamo/test_export
    "ExportTests.test_torch_inference_mode_ctx",  # dynamo/test_export
    "ExportTests.test_export_decomp",  # dynamo/test_export
    "ExportTests.test_export_mark_dynamic_conflict_dynamic_dim",  # dynamo/test_export
    "ExportTests.test_export_with_constant_not_return_const",  # dynamo/test_export
    "ExportTests.test_export_with_constant_none_control_flow",  # dynamo/test_export
    "ExportTests.test_exported_graph_serialization",  # dynamo/test_export
    "ExportTests.test_export_with_constant_free_function",  # dynamo/test_export
    "ExportTests.test_export_preserve_constraints_as_metadata_scalar",  # dynamo/test_export
    "ExportTests.test_export_with_constant_none_control_flow_free_func",  # dynamo/test_export
    "ExportTests.test_cond_raise_user_error_on_non_tensor_operands",  # dynamo/test_export
    "ExportTests.test_export_with_constant_not_none_control_flow_free_func",  # dynamo/test_export
    "ExportTests.test_export_with_constant_not_none_control_flow_pos",  # dynamo/test_export
    "ExportTests.test_export_with_constant_tuple_nonzero",  # dynamo/test_export
    "ExportTests.test_export_with_constant_not_none_control_flow",  # dynamo/test_export
    "ExportTests.test_export_with_constant_dict_values",  # dynamo/test_export
    "ExportTests.test_export_with_constant_list_nonzero",  # dynamo/test_export
    "ExportTests.test_export_dynamic_dim_cleanup",  # dynamo/test_export
    "ExportTests.test_export_with_constant_method_on_module_invoke_twice",  # dynamo/test_export
    "ExportTests.test_map_cond_param_buffer_lifted",  # dynamo/test_export
    "ExportTests.test_export_multi_dynamic_dim_unsafe_relationship",  # dynamo/test_export
    "ExportTests.test_export_with_constant_free_function_and_class_method_multiarg_diff",  # dynamo/test_export
    "ExportTests.test_export_with_constant_method_on_module",  # dynamo/test_export
    "ExportTests.test_export_multi_dynamic_dim_constraint",  # dynamo/test_export
    "ExportTests.test_untracked_inputs_in_constraints",  # dynamo/test_export
    "ExportTests.test_export_raise_on_relationship",  # dynamo/test_export
    "ExportTests.test_export_with_map_cond",  # dynamo/test_export
    "ExportTests.test_export_with_constant_list_nonzero_free_function",  # dynamo/test_export
    "ExcTests.test_graph_break_log",  # dynamo/test_exc
    "ExcTests.test_trigger_bisect_on_error",  # dynamo/test_exc
    "ExcTests.test_backend_suppress_line",  # dynamo/test_exc
    "ExcTests.test_not_implemented_error",  # dynamo/test_exc
    "ExcTests.test_trigger_on_error",  # dynamo/test_exc
    "ExcTests.test_internal_error_no_suppress",  # dynamo/test_exc
    "ExcTests.test_internal_error_suppress_errors",  # dynamo/test_exc
    "DecoratorTests.test_mark_static_address_guarded",  # dynamo/test_decorators
    "DecoratorTests.test_mark_static_address_unguarded",  # dynamo/test_decorators
    "CtxManagerTests.test_disable_saved_tensors_hooks",  # dynamo/test_ctx_manager
    "CtxManagerTests.test_disable_saved_tensors_hooks_prev_disabled",  # dynamo/test_ctx_manager
    "CtxManagerTests.test_autograd_profiler_enabled",  # dynamo/test_ctx_manager
    "CtxManagerTests.test_disable_saved_tensors_hooks_prev_disabled_nested",  # dynamo/test_ctx_manager
    "ComptimeTests.test_graph_break",  # dynamo/test_comptime
    "PublicTorchCompilerTests.test_dynamo_signatures",  # dynamo/test_compile
    "TestCustomBackendAPI.test_aot_autograd_api",  # dynamo/test_backends
    "TestOptimizations.test_example_inputs",  # dynamo/test_backends
    "TestCustomBackendAPI.test_lookup_backend",  # dynamo/test_backends
    "TestOptimizations.test_example_inputs_runtime_use",  # dynamo/test_backends
    "TestCustomBackendAPI.test_register_backend_api",  # dynamo/test_backends
    "AutogradFunctionTests.test_stride_in_bwd",  # dynamo/test_autograd_function
    "AutogradFunctionTests.test_print_in_bwd",  # dynamo/test_autograd_function
    "AutogradFunctionTests.test_graph_break_if_lifted_free_variable",  # dynamo/test_autograd_function
    "AotAutogradFallbackTests.test_aot_sequence_nr",  # dynamo/test_aot_autograd
    "TestTorchFunctionOverride.test_Tensor_log",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_prod",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___iand__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_orgqr",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___isub__",  # test_overrides
    "TestTorchFunctionOverride.test_tensor_subclass_propagation",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_detach_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_gcd_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addmv_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_matmul",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_copysign",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_index_add",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_chalf",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rfloordiv__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sign",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_tolist",  # test_overrides
    "TestRNN.test_rnn",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_retains_grad___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_squeeze_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_aminmax",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___lt__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_isclose",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase__backward_hooks___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_inverse",  # test_overrides
    "TestNamedTuple.test_max",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___ror__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_hypot",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_max",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arccos",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase__cdata___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__update_names",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sinc_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___div__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_q_per_channel_scales",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_tan_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_as_strided_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_lgamma",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_shared",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logdet",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_scatter_reduce",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_erf",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_equal",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sinh",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_split",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addmv",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_kthvalue",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_relu",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_select",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___contains__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_mm",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_det",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_igammac_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_atanh_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_int_repr",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_log_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_nested___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_digamma",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ormqr",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sparse_dim",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___irshift__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___imul__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_nanmean",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_to",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_index_add_",  # test_overrides
    "TestBroadcastAllOverride.test_broadcast_all",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_div",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_slice_inverse",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_slice_scatter",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_remainder_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_baddbmm_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_renorm",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_igamma_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_same_size",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_add_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sort",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_clone",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_crow_indices",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_log10",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_atan2_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_resize_as_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_subtract",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ne_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___index__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sparse_resize_and_clear_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_dot",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___reversed__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_msort",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_to_mkldnn",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addcmul_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ndimension",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_rsqrt_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__dimV",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logical_and_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_share_memory_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_rad2deg",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_greater_equal",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_coalesced",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_clamp_min_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_swapaxes",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cosh_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___array__",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_mkldnn___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_short",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_le_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_var",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_hsplit",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___deepcopy__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_fmin",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_expm1_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_slogdet",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_conj_physical_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logical_or_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_clamp_max",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_erf_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sub_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_nonzero",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_inner",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase__base___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_fliplr",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_unsafe_split",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_nelement",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase__post_accumulate_grad_hooks___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___setstate__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logical_xor_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_allclose",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_log1p",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_pow_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_lerp_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_vdot",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_requires_grad___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_indices",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_lgamma_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_mT___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__values",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_distributed",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_mtia___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_histc",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logical_or",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_svd",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cfloat",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_flipud",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___and__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_index_fill_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_swapdims_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_names___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_resolve_neg",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_output_nr___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cross",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_add",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_isposinf",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_tile",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_nan_to_num",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_trace",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___long__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_set_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arcsin_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_random_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_index_put",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_uniform_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_frac_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sigmoid_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_signbit",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_repeat_interleave",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bernoulli_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_fill_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___lshift__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_map_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_isnan",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_long",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_subtract_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_ndim___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_data_ptr",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_i0",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_qscheme",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logcumsumexp",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ldexp_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_neg",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sign_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_unsqueeze_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arctan2_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_clip_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cumprod_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_data___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_mode",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_layout___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_atan_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_xpu",  # test_overrides
    "TestTorchFunctionMode.test_mode_notimplemented_loop",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_baddbmm",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_index_reduce_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_requires_grad_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_minimum",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_erfinv",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_detach",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_dequantize",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_or_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_acos_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_trunc",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_quantile",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___mod__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__nested_tensor_storage_offsets",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_xlogy_",  # test_overrides
    "TestTorchFunctionOverride.test_mm_semantics",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_multiply",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_expand",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___xor__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cholesky_inverse",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_storage_type",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___idiv__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_dim_order",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rrshift__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_contiguous",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_contiguous",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_isreal",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_min",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_col_indices",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_register_post_accumulate_grad_hook",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_floor",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___invert__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_left_shift_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_flip",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logit_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_div_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_resize_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_view_as",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_gather",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___mul__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_flatten",  # test_overrides
    "TestTorchFunctionMode.test_disable_enable_subclass",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_numpy",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cumsum_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_fmod",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_has_names",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_xpu___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_tanh",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_eq_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ipu",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logsumexp",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_expand_as",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_split_with_sizes",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cumprod",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_fix_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_igamma",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_median",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_nextafter_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_polygamma_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase__version___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_less_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_lt_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_put_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_nanmedian",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_half",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logaddexp2",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logit",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___dlpack__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_nan_to_num_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cumsum",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_hardshrink",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_squeeze",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___add__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bernoulli",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sparse_resize_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_deg2rad_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_corrcoef",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_isfinite",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_fmax",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_lu_solve",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_rename_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_real___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_transpose",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__autocast_to_reduced_precision",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_any",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sinh_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_dsplit",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___setitem__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_amax",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_smm",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_stft",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_asin_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ne",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_rad2deg_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_scatter_reduce_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bincount",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_chunk",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_floor_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_renorm_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_asinh_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sspaddmm",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_reshape",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logical_xor",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_put",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_ort___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sgn_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___bool__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_q_scale",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_right_shift",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_refine_names",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_map2_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_negative_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_element_size",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_index_reduce",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bmm",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_mean",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_leaf___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_size",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_q_zero_point",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_meta___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_tan",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_log2_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_scatter_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__to_dense",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_divide",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_asin",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_quantized___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_exp_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_imag___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_swapaxes_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_std",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___len__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_not_equal",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___ifloordiv__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_fill_diagonal_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rmatmul__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_acosh",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arctanh_",  # test_overrides
    "TestTorchFunctionOverride.test_mean_semantics",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_rot90",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_diag",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_tril_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_softmax",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sparse_mask",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_double",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_erfc",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_tensor_split",  # test_overrides
    "TestGradCheckOverride.test_gradcheck",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_device___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_or",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___repr__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_less",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___cuda_array_interface_____get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_clamp_max_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_nansum",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cummin",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cauchy_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_igammac",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_mvlgamma_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addmm_",  # test_overrides
    "TestTorchFunctionMode.test_modes_return_notimplemented",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_atan",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_acosh_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___dlpack_device__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_q_per_channel_zero_points",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ceil",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_histogram",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_lcm_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_i0_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_atan2",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_narrow",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sinc",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sigmoid",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ldexp",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logical_and",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_mvlgamma",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_diff",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_resize",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sum",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__nested_tensor_strides",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_movedim",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_unsafe_chunk",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_resize_as_sparse_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_expm1",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_gcd",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_swapdims",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rmul__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logical_not",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_lerp",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_zero_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addcdiv",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_sparse___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_round_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_exp2_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_register_hook",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_diag_embed",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___array_wrap__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___radd__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_absolute_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ge_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___ge__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_type",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_T___get__",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_dtype___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_less_equal",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_grad___get__",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_volatile___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_clamp",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_clip",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_index_put_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase__grad_fn___get__",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_shape___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_permute",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_resolve_conj",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sqrt",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_true_divide_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_get_device",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_diagonal_scatter",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sub",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___ior__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_greater",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase__grad___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_apply_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_argsort",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_matrix_power",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cos_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_log_softmax",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_erfinv_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__autocast_to_full_precision",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_untyped_storage",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_where",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_angle",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_coalesce",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rdiv__",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_nbytes___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_masked_fill_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bfloat16",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_hypot_",  # test_overrides
    "TestEinsumOverride.test_wrapper",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sgn",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_char",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_triangular_solve",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_float",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___gt__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_scatter_add_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_exp",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_scatter_add",  # test_overrides
    "TestIterator.test_iterator",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___eq__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_masked_scatter_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_tril",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_nanquantile",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_masked_select",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_mv",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_negative",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_argwhere",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_matrix_exp",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_unique_consecutive",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_unfold",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___format__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_round",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_deg2rad",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_grad_fn___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_align_as",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_complex",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cosh",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_align_to",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_square",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arcsinh_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_unsqueeze",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_not",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_rename",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___reduce_ex__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_t_",  # test_overrides
    "TestTorchFunctionMode.test_modes_handle_first",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___ne__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_item",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___le__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ccol_indices",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_nonzero",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_copy_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_broadcast_to",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_right_shift_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_cpu___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arccosh",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arcsin",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_as_strided",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_nextafter",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_q_per_channel_axis",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_trunc_",  # test_overrides
    "TestTorchFunctionMode.test_nested_modes_with_python_has_torch_function",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arctan",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_t",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_diagonal",  # test_overrides
    "TestTorchFunctionMode.test_subclass_hash",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_isinf",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ger",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__coalesced_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_not_equal_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_multiply_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ge",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_abs",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___int__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_argmin",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_inference",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__is_view",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rmod__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_amin",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_select_scatter",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_index_fill",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_square_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addr",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_asinh",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_heaviside_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_adjoint",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_float_power",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_gt_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_xlogy",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__dimI",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_gt",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_index_copy_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___ixor__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_to_dense",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___complex__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_and_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_neg",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_norm",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___float__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__sparse_mask_projection",  # test_overrides
    "TestPickle.test_pickle",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___iadd__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_divide_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_geqrf",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_index_select",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___ilshift__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_frexp",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_int",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_maximum",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_storage_offset",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_byte",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_unbind",  # test_overrides
    "TestIndexing.test_getitem_subclass",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_floating_point",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_log10_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_xla___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cos",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_positive",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_resize_as",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_storage",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_pow",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_type_as",  # test_overrides
    "TestGradNewOnesOverride.test_newones",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__indices",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_tanh_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_masked_scatter",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_not_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cholesky",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_all",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_heaviside",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_lcm",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_reshape_as",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_greater_equal_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_backward",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_greater_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logaddexp",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_xor",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_transpose_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_polygamma",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_row_indices",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addbmm_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_triu_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_atanh",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rxor__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_normal_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_index_copy",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_logical_not_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_log1p_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_absolute",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_abs_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_count_nonzero",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_qr",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_relu_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_conj",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_fmod_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_unique",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___nonzero__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_geometric_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_topk",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rlshift__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_diagflat",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cummax",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cov",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_istft",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arctanh",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_name___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_nonzero_static",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_pin_memory",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arctan2",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_retain_grad",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_repeat",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_reciprocal_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_lt",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_record_stream",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___truediv__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rand__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_fix",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_multinomial",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___imod__",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_itemsize___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_copysign_",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_vulkan___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_dense_dim",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_prelu",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sin_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_reciprocal",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_sparse_csr___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arccosh_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_dist",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_narrow_copy",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_conj_physical",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arctan_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sin",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cdouble",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_masked_fill",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_log2",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sqrt_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_clamp_min",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addr_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rshift__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_mul",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_roll",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_signed",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addcdiv_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_true_divide",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_pinned",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_cuda___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_xor_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_floor_divide_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_eq",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_neg_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_view",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_as_strided_scatter",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_pinverse",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_vsplit",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___matmul__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_unsafe_split_with_sizes",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_ipu___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_and",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arcsinh",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_float_power_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_values",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__nnz",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_outer",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_log_normal_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___sub__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addbmm",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_mul_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor__nested_tensor_size",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bitwise_left_shift",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_digamma_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_isneginf",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_le",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_erfc_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_scatter",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_exp2",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_sum_to_size",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_take",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_take_along_dim",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rsub__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cholesky_solve",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_remainder",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_dim",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_triu",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_numel",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_acos",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_is_mps___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ravel",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_frac",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_moveaxis",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_lu",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_exponential_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___floordiv__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___getitem__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_rsqrt",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cpu",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_kron",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_cuda",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_less_equal_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_bool",  # test_overrides
    "TestTorchFunctionOverride.test_TensorBase_mH___get__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_arccos_",  # test_overrides
    "TestTorchFunctionOverride.test_precedence_semantics",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___or__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___rpow__",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addcmul",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_addmm",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_conj",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_is_set_to",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_to_sparse",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_argmax",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_clamp_",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_floor_divide",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor_ceil_",  # test_overrides
    "TestNamedTupleAPI.test_namedtuple_return",  # test_namedtuple_return_api
}

dynamo_skips = {
    "TestMatmulOperator.test_matmul_raises",
    "TestMatmulOperator.test_exceptions",
    "TestMatmulOperator.test_matmul_inplace",
    "TestMonitorTensorboard.test_event_handler",  # weird
    "TestMethods.test_diagonal",
    "TestMethods.test_searchsorted_complex",
    "TestMethods.test_round",
    "TestMethods.test_searchsorted_type_specific_2",
    "TestMethods.test_dot",
    "TestMethods.test_dot_out_mem_overlap",
    "TestMethods.test_partition_iterative",
    "TestMethods.test_trace",
    "TestMethods.test_matmul_out",
    "TestMethods.test_transpose",
    "TestMethods.test_conjugate",
    "TestMethods.test_choose_2",
    "TestMethods.test_size_zero_memleak",
    "TestMethods.test_searchsorted_with_invalid_sorter",
    "TestMethods.test_choose",
    "TestMethods.test_conjugate_out",
    "TestMethods.test_compress",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_arr_method_argmax_np_method0",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_arr_method_argmin_np_method1",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_0_method_argmin",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_0_method_argmax",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_1_method_argmax",
    "TestArgmaxArgminCommon.test_ret_is_out_ndim_1_method_argmin",
    "TestIsreal.test_fail",  # known py311 fail
    "TestIscomplexobj.test_basic",  # known py311 fail
    "TestIsrealobj.test_basic",  # known py311 fail
    "TestIsreal.test_pass",  # known py311 fail
    "TestIscomplex.test_pass",  # known py311 fail
    "TestIscomplexobj.test_list",  # known py311 fail
    "TestDiag.test_matrix",  # known py311 fail
    "TestVander.test_dtypes",  # known py311 fail
    "TestDstack.test_generator",  # known py311 fail
    "TestColumnStack.test_generator",  # known py311 fail
    "TestCov.test_complex",  # known py311 fail
    "TestSortComplex.test_sort_complex",  # known py311 fail
    "TestCorrCoef.test_xy",  # known py311 fail
    "TestCov.test_xy",  # known py311 fail
    "TestCorrCoef.test_complex",  # known py311 fail
    "TestUnique.test_simple_complex",  # known py311 fail
    "TestDigitize.test_casting_error",  # known py311 fail
    "TestConstant.test_check_constant",  # known py311 fail
    "TestFFTShift.test_fft_n",  # known py311 fail
    "TestHstack.test_generator",  # known py311 fail
    "TestVstack.test_generator",  # known py311 fail
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_I_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_I_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_L_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_Q_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_Q_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_P_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_P_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_H_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_H_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_L_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestCorrelate.test_complex",  # known py311 fail
    "TestStdVarComplex.test_basic",  # known py311 fail
    "TestEinsum.test_broadcasting_dot_cases",  # known py311 fail
    "WeakTest.test_make_weak_keyed_dict_from_dict",  # known py311 fail
    "TestViewOpsCPU.test_as_strided_gradients_cpu",  # known py311 fail
    "TestViewOpsLAZY.test_as_strided_gradients_lazy",  # known py311 fail
    "LoggingTest.testApiUsage",  # flaky?
    "TestPruningNN.test_global_pruning_importance_scores",  # flaky
    "TestOpenMP_ParallelFor.test_one_thread",  # test_openmp
    "TestTorchrun.test_multi_threads",  # backends/xeon/test_launch
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape3_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape0_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape1_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape2_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape3_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape0_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape2_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape1_cpu",  # known py38 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_1_shape0_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_2_shape2_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_2_shape0_cpu",  # known py311 fail
    "TestTransformersCPU.test_decoder_padding_and_src_mask_bool_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_2_shape3_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_1_shape3_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_1_shape2_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_1_shape1_cpu",  # known py311 fail
    "TestAttnMasksCPU.test_causal_variants_causal_variant_2_shape1_cpu",  # known py311 fail
    "TestFunctionalAutogradBenchmark.test_fast_tasks",  # flaky?
    "TestFrameworkUtils.test_filtering_env_var",  # known py38 fail
    "TestAsArrayCPU.test_default_device_cpu",  # known py38 fail
    "TestAsArrayCPU.test_astensor_consistency_cpu",  # known py311 fail
    "TestTensorCreationCPU.test_vander_types_cpu_complex128",  # known py311 fail
    "TestTensorCreationCPU.test_vander_types_cpu_complex64",  # known py311 fail
    "TestTensorCreationCPU.test_torch_polar_cpu_float32",  # known py311 fail
    "TestTensorCreationCPU.test_torch_polar_cpu_float64",  # known py311 fail
    "TestSWAUtils.test_averaged_model_all_devices_ema_True",  # flaky
    "TestSWAUtils.test_averaged_model_exponential_use_multi_avg_fn_True_use_buffers_False",  # flaky
    "TestSWAUtils.test_averaged_model_exponential_use_multi_avg_fn_True_use_buffers_True",  # flaky
    "TestOpenMP_ParallelFor.test_n_threads",  # known py311 fail
    "TestNativeFunctions.test_intlist_error_with_overload",  # known py311 fail
    "TestMkldnnFusion.test_single_conv",  # known py311 fail
    "TestTorchDlPackCPU.test_dlpack_export_is_conj_cpu",  # known py311 fail
    "TestPythonDispatcher.test_quantized_structured_not_implemented",  # known py38 fail
    "TestLazyReuseIr.testAdd",  # known py311 fail
    "TestLazyReuseIr.testAddSubFallback",  # known py311 fail
    "TestLazyReuseIr.testBatchNorm",  # known py311 fail
    "TestLazyReuseIr.testAddSub",  # known py311 fail
    "TestVerifyCorrectness.test_example_inputs",  # known py311 fail
    "RecompileTests.test_aliasing_guard_failures",  # known py311 fail
    "TestPythonAutograd.test_backwards1",  # known py311 fail
    "DynamicShapesExportTests.test_retracibility_dict_container_inp_out_dynamic_shapes",  # Takes way too long
    "DynamicShapesExportTests.test_retracibility_dynamic_shapes",  # takes way too long
    "DynamicShapesExportTests.test_retracibility_nested_list_out_dynamic_shapes",  # takes way too long
    "DynamoProfilerTests.test_dynamo_timed_profiling_backend_compile",  # known py311 fail
    "OptimizerTests.test_adadelta",  # known py311 fail
    "NopTests.test_extended_args",  # known py311 fail
    "MiscTests.test_exception_table_parsing",  # known py311 fail
    "MiscTests.test_py311_jump_offset",  # known py311 fail
    "MiscTests.test_linetable_311_writer1",  # known py311 fail
    "MiscTests.test_itertools_infinite_count",  # known py311 fail
    "MiscTests.test_exception_table_e2e",  # known py311 fail
    "MiscTests.test_linetable_311_writer2",  # known py311 fail
    "MiscTests.test_exception_table_e2e_2",  # known py311 fail
    "MiscTests.test_itertools_accumulate_tensors_user_defined",  # known py311 fail
    "MiscTests.test_itertools_accumulate_tensors_kwargs",  # known py311 fail
    "MiscTests.test_itertools_accumulate_tensors_builtins",  # known py311 fail
    "InteropTests.test_fx_fn",  # known py311 fail
    "HigherOrderOpTests.test_access_module_attr",  # known py311 fail
    "FrameInitTests.test_frame_init",  # known py311 fail
    "DynamicShapesMiscTests.test_tolist_0d_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_cond_pytree_operands_with_non_tensor_leaves_dynamic_shapes",  # known py311 fail  # noqa: B950
    "DynamicShapesHigherOrderOpTests.test_output_with_dict_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_multiple_outputs_diff_dims_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_wrap_subgraph_name_is_valid_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_over_vmap_two_inputs_dynamic_shapes",  # known py311 fail
    "DynamicShapesFunctionTests.test_math_radians_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_map_lowers_to_graph_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_type_copy_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_inference_mode_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_two_inputs_tuple_in_dims_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_get_cache_entry_dynamic_shapes",  # known py311 fail
    "DynamicShapesSubGraphTests.test_no_graph_break_on_item_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_tuple_nonzero_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_numpy_subdtype_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_modules_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_capture_value_created_in_subgraph_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_yield_send_to_subgenerator_graph_break_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_list_self_reference_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_vmap_source_fn_stack_dynamic_shapes",  # known py311 fail
    "DynamicShapesSubGraphTests.test_dynamic_duck_size_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_inline_closure_not_loaded_by_parent_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_numpy_non_torch_dtype_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_multiple_outputs_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_reformer_sorting_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_linetable_311_writer1_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_cond_raise_user_error_on_non_tensor_operands_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_free_tensor_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_list_nonzero_free_function_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_side_effect_in_body_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tracing_tree_map_only_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tensor_dict2_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tracing_nested_py_tree_mixed_all_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_dynamic_shapes_implicit_guard_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_numpy_int_constant_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_nn_module_getattr_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_yield_gen_and_from_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_map_cond_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_multiple_invocation_out_dims_dynamic_shapes",  # known py311 fail  # noqa: B950
    "DynamicShapesMiscTests.test_set_aliasing_recompiles_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_reformer_min_chunk_len_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_map_multi_return_dynamic_shapes",  # known py311 fail
    "DynamicShapesNNModuleTests.test_unsupportedmodule_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_numpy_iter_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_reformer_train_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_mandelbrot_numpy_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_infinite_repeat_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_numpy_force_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_list_nonzero_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tracing_nested_py_tree_tuples_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_seq_append_list_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_dynamic_dim_cleanup_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_infinite_cycle_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_map_source_fn_stack_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_cond_branches_no_arguments_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_nested_closure_mutation_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_tensor_data_kwarg_dynamic_shapes",  # known py311 fail
    "DynamicShapesFunctionTests.test_partials_as_input_partials_mod_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tolist_scalar_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_numpy_with_builtin_type_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_compare_shapes_with_constant_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_frozenset_torch_func_contains_dynamic_shapes",  # known py311 fail
    "DynamicShapesFunctionTests.test_is_contiguous_frame_counts_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_rewrite_assert_noop_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_map_symint_input_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_relative_import_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_raise_on_relationship_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_threading_local_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_reformer_eval_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_groupby_pure_python_default_identify_func_dynamic_shapes",  # known py311 fail  # noqa: B950
    "DynamicShapesMiscTests.test_release_input_memory_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_decomp_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_none_control_flow_free_func_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_longformer_chunk_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_disable_capture_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_free_function_and_class_method_multiarg_diff_dynamic_shapes",  # known py311 fail  # noqa: B950
    "DynamicShapesReproTests.test_boxes_len_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_list_slice_mul_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tensor_build_list_unpack_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_create_rand_mask_from_inputs_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_repeat_interleave_graphbreaks_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_grad_state_mutated_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_release_module_memory_dynamic_shapes",  # known py311 fail
    "DynamicShapesCtxManagerTests.test_disable_saved_tensors_hooks_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_wrap_kwarg_recompile_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_slice_input_dynamic_shapes",  # known py311 fail
    "DynamicShapesFunctionTests.test_fstrings2_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tracing_py_tree_tensor_subclass_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_multi_import_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_not_none_control_flow_pos_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_map_cond_param_buffer_lifted_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_over_vmap_captured_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_out_variants_with_resizing_on_graph_inputs_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_repeat_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_add_to_set_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tensor_dict1_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_convert_boxes_to_pooler_format_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_kwargs_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_exception_table_e2e_2_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_free_const_dynamic_shapes",  # known py311 fail
    "DynamicShapesFunctionTests.test_default_dict_lambda_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_guard_failure_fn_shape_control_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_merge_criteria_processor_list1_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_accumulate_tensors_builtins_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_numpy_not_ndarray_recompiles_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_function_in_skipfiles_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_grad_pytree_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_recompile_on_global_state_change_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_return_nested_function_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_cond_raise_user_error_on_non_list_operands_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_restricted_list_subclass1_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_raise_guard_full_constraint_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_optim_state_references_cleared_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_boolarg_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_nested_closure_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_map_pytree_return_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_access_module_attr_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_rewrite_assert_with_non_string_msg_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_no_raise_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tensor_item_capture_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tracing_py_tree_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_grad_non_tensor_input_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_grad_two_tensor_all_grad_has_aux_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_simple_set_usage_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_grad_fn_with_kwargs_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_numpy_size_attr_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_numpy_torch_operators_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_dunder_new_function_inlining_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_yield_from_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_empty_list_contains_with_jump_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_raise_on_backend_error_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_not_return_const_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_grad_with_side_effect_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_deque_input_dynamic_shapes",  # known py311 fail
    "DynamicShapesCtxManagerTests.test_autograd_profiler_enabled_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_linetable_311_writer2_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_recursive_map_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_dataclass_fields_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_list_aliasing_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_cond_source_fn_stack_dynamic_shapes",  # known py311 fail
    "DynamicShapesSubGraphTests.test_dynamic_zero_inference_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_functools_wraps_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_illegal_op_graph_break_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_is_compiling_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_relative_import_no_modulename_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_iter_set_dynamic_shapes",  # known py311 fail
    "DynamicShapesSubGraphTests.test_dynamic_getitem_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_negative_shape_guard_dynamic_shapes",  # known py311 fail
    "DynamicShapesFunctionTests.test_default_dict_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_deterministic_algorithms_mutated_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_str_format_assert2_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_not_none_control_flow_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_pytree_inputs_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_map_subgraph_name_is_valid_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tolist_kd_dynamic_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_dynamic_dim_range_constraint_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_raise_guard_partial_constraint_no_graph_break_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_any_all_symnode_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_numpy_recompilation_scalar_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_inplace_view_on_graph_input_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_dict_order_keys_modules_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_py311_jump_offset_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_addr_alpha_beta_out_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_cond_pytree_operands_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_untracked_inputs_in_constraints_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_py_guards_mark_dynamic_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_rewrite_assert_with_msg_dynamic_shapes",  # known py311 fail
    "DynamicShapesSubGraphTests.test_dynamic_kwarg_dynamic_shapes",  # known py311 fail
    "DynamicShapesNNModuleTests.test_lazy_module_no_cls_to_become_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_release_scope_memory_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_callpacked_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_listcomp_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_infinite_count_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_namedtuple1_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_mark_dynamic_conflict_dynamic_dim_dynamic_shapes",  # known py311 fail
    "DynamicShapesNNModuleTests.test_self_mutating1_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_user_getattribute_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_accumulate_tensors_default_sum_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_size_typematch_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_inplace_param_update_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_nested_tuple_output_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_closure_out_of_scope_cell_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_not_none_control_flow_free_func_dynamic_shapes",  # known py311 fail  # noqa: B950
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_multiple_invocation_in_dims_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_accumulate_tensors_user_defined_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_nan_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_numpy_readonly_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_multi_dynamic_dim_constraint_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_hf_t5_forward_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tolist_kd_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_capture_untracked_global_nested_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_two_inputs_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_grad_has_aux_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_free_function_and_class_method_dynamic_shapes",  # known py311 fail  # noqa: B950
    "DynamicShapesExportTests.test_export_with_constant_none_control_flow_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_add_sub_alpha_out_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_jit_trace_errors_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_setattr_mutation1_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_multiple_outputs_out_dims_tuple_dynamic_shapes",  # known py311 fail  # noqa: B950
    "DynamicShapesMiscTests.test_numpy_array_of_arrays_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_method_on_module_invoke_twice_dynamic_shapes",  # known py311 fail  # noqa: B950
    "DynamicShapesSubGraphTests.test_dynamic_order_dependence_dynamic_shapes",  # known py311 fail
    "DynamicShapesNNModuleTests.test_unsupportedmethod_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_exception_table_parsing_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_batchnorm_e2e_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_merge_criteria_processor_list2_dynamic_shapes",  # known py311 fail
    "DynamicShapesSubGraphTests.test_enumerate_not_break_graph_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_chunk_reformer_ff_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_with_builtin_type_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tracing_nested_py_tree_dicts_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_no_raise_guard_partial_constraint_across_break_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_method_on_module_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_cond_nested_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_requires_grad_guards_with_grad_mode2_dynamic_shapes",  # known py311 fail
    "DynamicShapesCtxManagerTests.test_disable_saved_tensors_hooks_prev_disabled_nested_dynamic_shapes",  # known py311 fail  # noqa: B950
    "DynamicShapesMiscTests.test_cond_side_effects_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_nested_optimize_run_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_nested_optimize_decorator_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_nested_optimize_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_typing_typevar_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_side_effects_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_intermediary_tensor_grad_access_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_no_raise_guard_partial_constraint_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_dtypes_no_graphbreaks_dynamic_shapes",  # known py311 fail
    "DynamicShapesFunctionTests.test_partials_as_input_partials_lambda_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_mark_static_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tuple_mul_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_deque_append_left_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_many_views_with_mutation_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_hf_xsoftmax_training_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_accumulate_symint_default_sum_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_do_paste_mask_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_torch_seed_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_infinite_repeat_mutation_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_grad_two_tensor_has_aux_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_numpy_tolist_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_dict_order_keys_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_free_function_and_class_method_multiarg_dynamic_shapes",  # known py311 fail  # noqa: B950
    "DynamicShapesHigherOrderOpTests.test_cond_branches_no_arguments_no_closure_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_pure_python_accumulate_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tolist_1d_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_exported_graph_serialization_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_exception_table_e2e_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_namedtuple2_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_fallback_on_python_primitives_output_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_size_input_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_guard_failure_fn_tensor_iter_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_accumulate_tensors_kwargs_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_dict_order_keys_tensors_dynamic_shapes",  # known py311 fail
    "DynamicShapesCtxManagerTests.test_disable_saved_tensors_hooks_prev_disabled_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_grad_with_graph_break_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_module_in_skipfiles_dynamic_shapes",  # known py311 fail
    "DynamicShapesHigherOrderOpTests.test_cond_subgraph_name_is_valid_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tensor_dict3_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_torch_inference_mode_ctx_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_dictcomp_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_grad_dynamic_shapes",  # known py311 fail
    "DynamicShapesFunctionTests.test_default_dict_closure_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_issue175_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_dict_values_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_grad_disable_capture_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_itertools_groupby_pure_python_key_func_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_backend_match_guard_dynamic_shapes",  # known py311 fail
    "DynamicShapesFuncTorchHigherOrderOpTests.test_grad_freevar_python_scalar_dynamic_shapes",  # known py311 fail
    "DynamicShapesMiscTests.test_tracing_nested_py_tree_dynamic_shapes",  # known py311 fail
    "DynamicShapesReproTests.test_validate_model_kwargs_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_preserve_constraints_as_metadata_scalar_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_cond_raise_user_error_on_unsupported_pred_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_multi_dynamic_dim_unsafe_relationship_dynamic_shapes",  # known py311 fail
    "DynamicShapesExportTests.test_export_with_constant_free_function_dynamic_shapes",  # known py311 fail
    "DecoratorTests.test_allow_in_graph",  # known py311 fail
    "InPlaceCompilationTests.test_compilation",  # known py311 fail
    "DynamicShapesAotAutogradFallbackTests.test_aot_sequence_nr_dynamic_shapes",  # weird
    "ExportTests.test_predispatch_with_for_out_dtype",  # weird
    "ExportTests.test_predispatch_with_for_out_dtype_nested",  # weird
    "MiscTests.test_auto_functionalize_on_view",  # weird
    "MiscTests::test_auto_functionalize_optional",  # weird
    "MiscTests::test_auto_functionalize_with_returns",  # weird
    "MiscTests::test_generate_trivial_abstract_impl",  # weird
    "RecompileUxTests.test_drop_cache_on_skip",  # weird
    "ReproTests.test_optim_state_references_cleared",  # weird
    "ReproTests.test_reformer_train",  # weird
    "TraceRuleTests.test_torch_name_rule_map_updated",  # weird
    "TestTorchFunctionOverride.test_TensorBase_H___get__",  # known py311 fail
    "TestCheckpoint.test_checkpoint_trigger",  # known py38 fail
    "TestBottleneck.test_bottleneck_cpu_only",  # known py38 fail
    "TestUnaryUfuncsCPU.test_sinc_cpu_float64",  # known py38 fail
    "TestUnaryUfuncsCPU.test_silu_complex_cpu_complex64",  # known py38 fail
    "TestUnaryUfuncsCPU.test_exp_cpu_complex128",  # known py38 fail
    "TestUnaryUfuncsCPU.test_special_i0_i1_vs_scipy_cpu_float32",  # known py38 fail
    "TestUnaryUfuncsCPU.test_silu_complex_cpu_complex128",  # known py38 fail
    "TestUnaryUfuncsCPU.test_special_i0_i1_vs_scipy_cpu_float64",  # known py38 fail
    "TestUnaryUfuncsCPU.test_log1p_complex_cpu_complex64",  # known py38 fail
    "TestUnaryUfuncsCPU.test_exp_cpu_complex64",  # known py38 fail
    "TestUnaryUfuncsCPU.test_log1p_complex_cpu_complex128",  # known py38 fail
    "TestUnaryUfuncsCPU.test_special_i0_i1_vs_scipy_cpu_bfloat16",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_int32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_int64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_int16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_bfloat16",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int64_noncontiguous_False_cpu_float64",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int64_noncontiguous_True_cpu_complex64",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int32_noncontiguous_False_cpu_complex128",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_float32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_int64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_int16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_float32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_bool",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int32_noncontiguous_True_cpu_complex64",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_int16",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int64_noncontiguous_True_cpu_complex64",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_csr_to_block_csr_blocksize_4_cpu_float64_int64",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int32_noncontiguous_True_cpu_float64",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_int64",  # known py38 fail
    "TestSparseCSRCPU.test_csr_to_block_csr_blocksize_4_cpu_float64_int32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_complex128",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_bfloat16",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_int8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_float64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_int64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_float16",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int32_noncontiguous_False_cpu_complex64",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int64_noncontiguous_False_cpu_float64",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int64_noncontiguous_False_cpu_complex128",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_uint8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_int32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_bool",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_int32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_uint8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_uint8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_int16",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int64_noncontiguous_True_cpu_float32",  # known py38 fail
    "TestSparseCSRCPU.test_dense_to_from_sparse_compressed_SparseCSC_NonBatched_NonHybrid_cpu",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_float16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_int64",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_bfloat16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_float64",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_int8",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_complex128",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_to_sparse_compressed_SparseCSR_cpu_float64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_complex64",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int32_noncontiguous_True_cpu_float32",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int64_noncontiguous_True_cpu_float64",  # known py38 fail
    "TestSparseCSRCPU.test_csr_to_block_csr_blocksize_2_cpu_float64_int64",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_int16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_int8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_complex64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_complex64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_float64",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int32_noncontiguous_True_cpu_complex128",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_float64",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_bfloat16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_int64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_int64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_complex64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_uint8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_int16",  # known py38 fail
    "TestSparseCSRCPU.test_dense_to_from_sparse_compressed_SparseBSR_NonBatched_NonHybrid_cpu",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int32_noncontiguous_False_cpu_float64",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_bfloat16",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int64_noncontiguous_True_cpu_complex128",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_float16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_float32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_int16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_int16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_bfloat16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_int8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_int32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_float32",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int32_noncontiguous_True_cpu_float64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_float16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_float64",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int64_noncontiguous_False_cpu_float32",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_float16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_float16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_float64",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int32_noncontiguous_False_cpu_float32",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_uint8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_int32",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_int64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_complex128",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int64_noncontiguous_True_cpu_float32",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_float32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_float64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_int8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_complex128",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_float64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_bool",  # known py38 fail
    "TestSparseCSRCPU.test_mm_errors_cpu_float32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_bool",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_bool",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_uint8",  # known py38 fail
    "TestSparseCSRCPU.test_dense_to_from_sparse_compressed_SparseCSR_NonBatched_NonHybrid_cpu",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_complex128",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_int8",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int32_noncontiguous_True_cpu_complex64",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int32_noncontiguous_False_cpu_complex128",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_int32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_int32",  # known py38 fail
    "TestSparseCSRCPU.test_addmm_errors_cpu_float32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_float32",  # known py38 fail
    "TestSparseCSRCPU.test_dense_to_from_sparse_compressed_SparseBSR_Batched_NonHybrid_cpu",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_complex64",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int64_noncontiguous_False_cpu_complex64",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_complex64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_float32",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_complex64",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_to_sparse_compressed_SparseCSC_cpu_float64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_float64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_float16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_uint8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_int16",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int64_noncontiguous_False_cpu_float32",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_complex128",  # known py38 fail
    "TestSparseCSRCPU.test_dense_to_from_sparse_compressed_SparseCSR_Batched_NonHybrid_cpu",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_int64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_bool",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_bool",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_bfloat16",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_complex64",  # known py38 fail
    "TestSparseCSRCPU.test_csr_to_block_csr_blocksize_2_cpu_float64_int32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_bfloat16",  # known py38 fail
    "TestSparseCSRCPU.test_dense_to_from_sparse_compressed_SparseBSC_Batched_NonHybrid_cpu",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_complex128",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_complex128",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_int16",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_uint8",  # known py38 fail
    "TestSparseCSRCPU.test_dense_to_from_sparse_compressed_SparseCSC_Batched_NonHybrid_cpu",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_float16",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_complex128",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_bfloat16",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csc_to_dense_cpu_float16",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int32_noncontiguous_False_cpu_complex64",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_int8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_uint8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_complex128",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_uint8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int32_cpu_float64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_int32",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int64_noncontiguous_False_cpu_complex64",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_int8",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int64_noncontiguous_True_cpu_complex128",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int32_noncontiguous_False_cpu_float64",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_complex64",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int32_noncontiguous_True_cpu_float32",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_2_int32_noncontiguous_False_cpu_float32",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_bool",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_bool",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_int8",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int64_cpu_float32",  # known py38 fail
    "TestSparseCSRCPU.test_sparse_csr_to_dense_cpu_float32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSR_int32_cpu_float32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_int32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int64_cpu_int64",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int32_noncontiguous_True_cpu_complex128",  # known py38 fail  # noqa: B950
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int64_noncontiguous_True_cpu_float64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_complex64",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_bool",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseCSC_int64_cpu_int8",  # known py38 fail
    "TestSparseCSRCPU.test_block_triangular_solve_block_size_3_int64_noncontiguous_False_cpu_complex128",  # known py38 fail  # noqa: B950
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int64_cpu_float16",  # known py38 fail
    "TestSparseCSRCPU.test_dense_to_from_sparse_compressed_SparseBSC_NonBatched_NonHybrid_cpu",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSR_int32_cpu_int32",  # known py38 fail
    "TestSparseCompressedCPU.test_select_copy_SparseBSC_int32_cpu_bfloat16",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseBSC_masked_fast_cpu",  # known py38 fail
    "TestSparseCPU.test_index_select_parallelization_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_spdiags_cpu_uint8",  # known py38 fail
    "TestSparseCPU.test_sparse_to_numpy_cpu",  # known py38 fail
    "TestSparseCPU.test_shared_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sparse_sparse_mul_cpu_float32",  # known py38 fail
    "TestSparseCPU.test_asin_arcsin_cpu_int32",  # known py38 fail
    "TestSparseCPU.test_cat_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_add_zeros_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_mask_hybrid_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_index_select_exhaustive_index_large_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_select_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_masked_cpu_float16",  # known py38 fail
    "TestSparseCPU.test_sparse_dense_mul_cpu_int8",  # known py38 fail
    "TestSparseCPU.test_index_select_parallelization_cpu_complex128",  # known py38 fail
    "TestSparseAnyCPU.test_gradcheck_mm_SparseCOO_sparse_slow_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_asin_arcsin_cpu_int64",  # known py38 fail
    "TestSparseCPU.test_sparse_add_coalesce_cpu_float32",  # known py38 fail
    "TestSparseAnyCPU.test_constructor_autograd_SparseCSR_cpu",  # known py38 fail
    "TestSparseCPU.test_zeros_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_clone_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_permute_sparse_cpu_float64",  # known py38 fail
    "TestSparseAnyCPU.test_constructor_autograd_SparseCOO_cpu",  # known py38 fail
    "TestSparseCPU.test_basic_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_sparse_cpu_float32",  # known py38 fail
    "TestSparseCPU.test_norm_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_mask_hybrid_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_factory_cpu_float32",  # known py38 fail
    "TestSparseAnyCPU.test_gradcheck_mm_SparseCOO_masked_fast_cpu_complex128",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCOO_nonmasked_slow_cpu",  # known py38 fail
    "TestSparseCPU.test_factory_type_inference_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_dense_mul_cpu_int16",  # known py38 fail
    "TestSparseCPU.test_print_uncoalesced_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_index_select_empty_and_non_contiguous_index_cpu_float64",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseBSC_masked_slow_cpu",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCSC_nonmasked_fast_cpu",  # known py38 fail
    "TestSparseAnyCPU.test_check_sparse_tensor_invariants_SparseBSR_cpu",  # known py38 fail
    "TestSparseCPU.test_resize_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_masked_cpu_complex64",  # known py38 fail
    "TestSparseCPU.test_neg_negative_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_factory_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_int8",  # known py38 fail
    "TestSparseAnyCPU.test_constructor_autograd_SparseBSC_cpu",  # known py38 fail
    "TestSparseCPU.test_asin_arcsin_cpu_float32",  # known py38 fail
    "TestSparseCPU.test_sparse_mask_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_factory_type_inference_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_select_no_type_promotion_cpu_uint8",  # known py38 fail
    "TestSparseCPU.test_sparse_dense_mul_cpu_bool",  # known py38 fail
    "TestSparseCPU.test_sparse_dense_mul_cpu_float32",  # known py38 fail
    "TestSparseCPU.test_index_select_empty_and_non_contiguous_index_cpu_complex128",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseBSR_masked_fast_cpu",  # known py38 fail
    "TestSparseAnyCPU.test_check_sparse_tensor_invariants_SparseCOO_cpu",  # known py38 fail
    "TestSparseCPU.test_mv_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_sparse_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_masked_cpu_float32",  # known py38 fail
    "TestSparseCPU.test_sparse_spdiags_cpu_int8",  # known py38 fail
    "TestSparseCPU.test_sparse_add_coalesce_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_spdiags_cpu_bool",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_sparse_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_ctor_is_coalesced_with_gradcheck_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_spdiags_cpu_int32",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCSR_masked_fast_cpu",  # known py38 fail
    "TestSparseCPU.test_neg_negative_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_complex64",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_bool",  # known py38 fail
    "TestSparseCPU.test_transpose_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_factory_type_inference_cpu_int64",  # known py38 fail
    "TestSparseCPU.test_div_rounding_mode_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_legacy_new_device_cpu",  # known py38 fail
    "TestSparseCPU.test_empty_like_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_pickle_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_to_dense_hybrid_masked_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_index_select_exhaustive_index_large_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_is_nonzero_cpu",  # known py38 fail
    "TestSparseCPU.test_log1p_cpu_uint8",  # known py38 fail
    "TestSparseCPU.test_log1p_cpu_int32",  # known py38 fail
    "TestSparseCPU.test_sparse_broadcast_to_cpu_complex128",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCSR_masked_slow_cpu",  # known py38 fail
    "TestSparseCPU.test_sparse_add_coalesce_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_to_dense_hybrid_sparse_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_asin_arcsin_cpu_uint8",  # known py38 fail
    "TestSparseCPU.test_sparse_sum_cpu_float64",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCOO_masked_slow_cpu",  # known py38 fail
    "TestSparseCPU.test_sparse_spdiags_cpu_int16",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_sparse_cpu_float16",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_float64",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCSR_nonmasked_slow_cpu",  # known py38 fail
    "TestSparseCPU.test_log1p_cpu_float32",  # known py38 fail
    "TestSparseAnyCPU.test_gradcheck_mm_SparseCOO_masked_fast_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_sparse_cpu_bfloat16",  # known py38 fail
    "TestSparseCPU.test_log1p_cpu_int16",  # known py38 fail
    "TestSparseCPU.test_sparse_mm_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_hsmm_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_permute_masked_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sparse_spdiags_cpu_float32",  # known py38 fail
    "TestSparseAnyCPU.test_gradcheck_mm_SparseCOO_masked_slow_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_legacy_new_cpu",  # known py38 fail
    "TestSparseCPU.test_print_coalesced_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_mm_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sparse_dense_mul_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sparse_dense_mul_cpu_complex64",  # known py38 fail
    "TestSparseCPU.test_sparse_dense_mul_cpu_uint8",  # known py38 fail
    "TestSparseCPU.test_zeros_like_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_unsqueeze_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_basic_ops_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_factory_type_inference_cpu_complex64",  # known py38 fail
    "TestSparseCPU.test_factory_copy_cpu",  # known py38 fail
    "TestSparseCPU.test_small_nnz_coalesced_cpu",  # known py38 fail
    "TestSparseCPU.test_sparse_sparse_mul_cpu_int8",  # known py38 fail
    "TestSparseCPU.test_dsmm_cpu_float64",  # known py38 fail
    "TestSparseAnyCPU.test_gradcheck_mm_SparseCOO_sparse_fast_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_factory_cpu_float16",  # known py38 fail
    "TestSparseCPU.test_factory_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_asin_arcsin_cpu_int16",  # known py38 fail
    "TestSparseAnyCPU.test_check_sparse_tensor_invariants_SparseCSR_cpu",  # known py38 fail
    "TestSparseCPU.test_select_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_change_tensor_metadata_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_isnan_cpu",  # known py38 fail
    "TestSparseCPU.test_contig_hybrid_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_index_select_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_select_no_type_promotion_cpu_int32",  # known py38 fail
    "TestSparseCPU.test_zeros_like_cpu_complex128",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseBSR_nonmasked_slow_cpu",  # known py38 fail
    "TestSparseCPU.test_index_select_cpu_complex128",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCOO_masked_fast_cpu",  # known py38 fail
    "TestSparseCPU.test_log1p_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_saddmm_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sparse_sparse_mul_cpu_int16",  # known py38 fail
    "TestSparseCPU.test_sparse_broadcast_to_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_asin_arcsin_cpu_int8",  # known py38 fail
    "TestSparseCPU.test_factory_type_inference_cpu_float16",  # known py38 fail
    "TestSparseCPU.test_index_select_exhaustive_index_small_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_select_no_type_promotion_cpu_int64",  # known py38 fail
    "TestSparseCPU.test_norm_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sparse_dense_mul_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_float32",  # known py38 fail
    "TestSparseCPU.test_new_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_addmm_cpu_bfloat16",  # known py38 fail
    "TestSparseCPU.test_sparse_addmm_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_to_dense_hybrid_sparse_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_cat_cpu_float64",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseBSC_nonmasked_slow_cpu",  # known py38 fail
    "TestSparseCPU.test_sparse_spdiags_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_spdiags_cpu_complex128",  # known py38 fail
    "TestSparseAnyCPU.test_gradcheck_mm_SparseCOO_sparse_slow_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_mm_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_factory_cpu_complex64",  # known py38 fail
    "TestSparseCPU.test_sparse_dense_mul_cpu_int64",  # known py38 fail
    "TestSparseCPU.test_sparse_add_out_bfloat16_cpu_float32",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_masked_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_factory_type_inference_cpu_float32",  # known py38 fail
    "TestSparseCPU.test_index_select_exhaustive_index_small_cpu_float64",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCSR_nonmasked_fast_cpu",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_int32",  # known py38 fail
    "TestSparseCPU.test_resize_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_addmm_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_add_dense_sparse_mismatch_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_bmm_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_basic_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sparse_sparse_mul_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_clone_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_log1p_cpu_int64",  # known py38 fail
    "TestSparseCPU.test_sparse_spdiags_cpu_int64",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_int16",  # known py38 fail
    "TestSparseCPU.test_to_dense_hybrid_masked_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_uint8",  # known py38 fail
    "TestSparseCPU.test_add_dense_sparse_mismatch_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_contig_hybrid_cpu_float64",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseBSR_nonmasked_fast_cpu",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCSC_masked_slow_cpu",  # known py38 fail
    "TestSparseCPU.test_sparse_add_coalesce_cpu_complex64",  # known py38 fail
    "TestSparseCPU.test_log1p_cpu_int8",  # known py38 fail
    "TestSparseCPU.test_permute_masked_cpu_float64",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCSC_nonmasked_slow_cpu",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseBSC_nonmasked_fast_cpu",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_sparse_cpu_complex64",  # known py38 fail
    "TestSparseAnyCPU.test_check_sparse_tensor_invariants_SparseCSC_cpu",  # known py38 fail
    "TestSparseCPU.test_div_rounding_mode_cpu_float32",  # known py38 fail
    "TestSparseCPU.test_any_cpu",  # known py38 fail
    "TestSparseMeta.test_basic",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCSC_masked_fast_cpu",  # known py38 fail
    "TestSparseAnyCPU.test_constructor_autograd_SparseBSR_cpu",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_masked_cpu_bfloat16",  # known py38 fail
    "TestSparseCPU.test_saddmm_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_shared_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_transpose_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_assign_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_permute_sparse_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_spadd_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_sparse_mul_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sparse_dense_mul_cpu_int32",  # known py38 fail
    "TestSparseCPU.test_asin_arcsin_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_change_tensor_metadata_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sparse_sparse_mul_cpu_int32",  # known py38 fail
    "TestSparseCPU.test_sparse_sparse_mul_cpu_complex64",  # known py38 fail
    "TestSparseCPU.test_unsqueeze_cpu_complex128",  # known py38 fail
    "TestSparseAnyCPU.test_constructor_autograd_SparseCSC_cpu",  # known py38 fail
    "TestSparseCPU.test_sparse_spdiags_cpu_complex64",  # known py38 fail
    "TestSparseCPU.test_add_zeros_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_new_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_contig_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_empty_like_cpu_complex128",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseCOO_nonmasked_fast_cpu",  # known py38 fail
    "TestSparseAnyCPU.test_check_sparse_tensor_invariants_SparseBSC_cpu",  # known py38 fail
    "TestSparseAnyCPU.test_gradcheck_mm_SparseCOO_masked_slow_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_sparse_mul_cpu_uint8",  # known py38 fail
    "TestSparseCPU.test_sparse_sparse_mul_cpu_int64",  # known py38 fail
    "TestSparseCPU.test_select_no_type_promotion_cpu_int16",  # known py38 fail
    "TestSparseAnyCPU.test_as_sparse_gradcheck_SparseBSR_masked_slow_cpu",  # known py38 fail
    "TestSparseCPU.test_select_no_type_promotion_cpu_int8",  # known py38 fail
    "TestSparseCPU.test_zeros_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sparse_mask_cpu_complex128",  # known py38 fail
    "TestSparseAnyCPU.test_gradcheck_mm_SparseCOO_sparse_fast_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_contig_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_masked_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_coalesce_transpose_mm_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_int64",  # known py38 fail
    "TestReductionsCPU.test_argminmax_multiple_cpu_float16",  # known py38 fail
    "TestReductionsCPU.test_argminmax_multiple_cpu_int16",  # known py38 fail
    "TestReductionsCPU.test_histogramdd_cpu_float32",  # known py38 fail
    "TestReductionsCPU.test_argminmax_multiple_cpu_int8",  # known py38 fail
    "TestReductionsCPU.test_tensor_compare_ops_empty_cpu",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_float32",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_float64",  # known py38 fail
    "TestReductionsCPU.test_argminmax_multiple_cpu_float64",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_uint8",  # known py38 fail
    "TestReductionsCPU.test_argminmax_multiple_cpu_uint8",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_int8",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_int16",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_int32",  # known py38 fail
    "TestReductionsCPU.test_argminmax_multiple_cpu_float32",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_complex64",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_int64",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_complex128",  # known py38 fail
    "TestReductionsCPU.test_argminmax_multiple_cpu_int64",  # known py38 fail
    "TestReductionsCPU.test_tensor_reduce_ops_empty_cpu",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_float16",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_bool",  # known py38 fail
    "TestReductionsCPU.test_argminmax_multiple_cpu_int32",  # known py38 fail
    "TestReductionsCPU.test_tensor_compare_ops_argmax_argmix_kthvalue_dim_empty_cpu",  # known py38 fail
    "TestReductionsCPU.test_histogram_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_broadcast_tensors_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_adaptive_avg_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_isneginf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_polygamma_polygamma_n_1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_ldexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_triplet_margin_with_distance_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_gather_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_interpolate_nearest_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_hstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_resolve_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_gt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_binary_cross_entropy_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_pinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_new_empty_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_pinv_hermitian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_var_mean_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_eigh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_cholesky_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_argsort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cumsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_inner_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_diagflat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_atleast_1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_max_unpool2d_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_guards_equal",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_var_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_byte_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_tril_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_log_softmax_with_dtype_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_addmm_decomposed_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_heaviside_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_digamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_randn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_tensorinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_ldl_factor_ex_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_varargs",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_embedding_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_argwhere_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_index_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_dsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_jiterator_4inputs_with_extra_args_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_normal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_i1e_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_NumpyNonzeroCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_i0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_rand_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_unfold_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_modified_bessel_k0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_l1_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cos_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_triu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_ldl_factor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_equal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_i0_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_pre_dispatch_mode_stack",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_shifted_chebyshev_polynomial_w_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_square_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_floor_divide_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_cond_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_xlog1py_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_shifted_chebyshev_polynomial_u_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_diag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_binary_cross_entropy_with_logits_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_interpolate_bicubic_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_corrcoef_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_general_cosine_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_ifft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_view_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_NumpySplitCopyWithIntCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_std_mean_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_modified_bessel_k0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_zeros_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_scatter_reduce_amin_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_pr_86917",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_ldexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_lu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_hardshrink_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_tan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_resolve_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_new_full_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cumsum_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_empty_like_doesnt_burn_in_defaults",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_polygamma_special_polygamma_n_0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_clamp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_pinverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_meshgrid_variadic_tensors_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_aminmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_dropout_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_ifftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_adaptive_avg_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_half_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_sqrt_size",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_argmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_triplet_margin_with_distance_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_broadcast_to_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_kthvalue_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_matrix_power_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_lt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_reshape_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive__native_batch_norm_legit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_reshape_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_new_ones_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_asin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_cross_entropy_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_remainder_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_view_as_complex_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_pre_dispatch_linear",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_reciprocal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_topk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_conv3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_var_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_transpose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nan_to_num_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_new_empty",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fmod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_hardshrink_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_bilinear_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_svd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_log_normal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_topk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_polygamma_special_polygamma_n_0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_shifted_chebyshev_polynomial_t_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_div_floor_rounding_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_combinations_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_addcmul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_fractional_max_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sparse_sampled_addmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_copysign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_log_normal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_randint_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_shifted_chebyshev_polynomial_u_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_selu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_deg2rad_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_tensorsolve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signal_windows_kaiser_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_multi_dot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sum_to_size_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_adaptive_avg_pool3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_triplet_margin_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_permute_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_interpolate_nearest-exact_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_smooth_l1_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logical_or_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_count_nonzero_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive__upsample_bilinear2d_aa_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_unsqueeze_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_bool_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_bfloat16_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_matmul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_float_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signbit_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_proxy_tensor",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_max_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_take_along_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_NumpyViewCopyCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_irfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_ldl_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_square_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_dsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_permute_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_unique_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_pad_circular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_hardswish_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_empty_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_diagflat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_max_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_round_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_lu_factor_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_max_unpool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_avg_pool3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inner_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_exp2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_matrix_rank_hermitian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_bucketize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_combinations_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_log10_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_bfloat16_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_complex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_pad_constant_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace___getitem___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_interpolate_bicubic_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cartesian_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_max_reduction_no_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_diag_embed_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_histc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_instance_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_vander_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nonzero_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_make_fx_overloads",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_svd_lowrank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive___getitem___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_expm1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_lt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_min_reduction_no_dim_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_guard_upperbound_range_refinement_multivariate",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_norm_nuc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_column_stack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_index_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_argsort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_bessel_y1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_unfold_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_take_along_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_max_unpool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_argwhere_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_max_pool1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signal_windows_hamming_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_unbind_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_inv_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_squeeze_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_index_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_float_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_interpolate_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_to_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_argmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_take_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cdist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_addbmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_tensorinv_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_pre_dispatch_mode_stack",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_polygamma_polygamma_n_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_polygamma_polygamma_n_0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_uniform_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_max_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logspace_tensor_overload_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_empty_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_grid_sampler_2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_true_divide_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_conv_transpose3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_gt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_repeat_interleave_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_airy_ai_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_diagonal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nonzero_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_double_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_soft_margin_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_ihfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_hermite_polynomial_he_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_pad_replicate_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_polygamma_special_polygamma_n_0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_bessel_j0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_matrix_rank_hermitian_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_solve_triangular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_clamp_min_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_take_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_broadcast_to_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_dropout_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_frac_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_polar_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_round_decimals_0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_minimum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_normal_number_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_avg_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_geometric_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_modified_bessel_i1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_combinations_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_scatter_reduce_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_clone_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_glu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_var_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_pixel_shuffle_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_bernoulli_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_zero__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_matrix_rank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_local_response_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_interpolate_nearest_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_val_metadata_mutation",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_softplus_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_min_reduction_with_dim_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_max_unpool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logspace_tensor_overload_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_mvlgamma_mvlgamma_p_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_pad_circular_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_chebyshev_polynomial_w_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_new_zeros_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_bfloat16_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_index_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_pad_reflect_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_max_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_dropout3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_broadcast_shapes_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_empty_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_resolve_conj_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logical_not_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_equal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cummin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_bessel_j1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_householder_product_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_conj_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_ldl_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_ldl_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_scatter_reduce_mean_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_vector_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_lu_unpack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mvlgamma_mvlgamma_p_1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_ne_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_polygamma_polygamma_n_3_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_repeat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_entr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_softplus_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_dot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_complex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_ctc_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_scaled_modified_bessel_k1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_select_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_airy_ai_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_round_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_nll_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_stack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_rsub_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_varargs",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_expand_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_rfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_ones_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_jiterator_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_rrelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_digamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_clone_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_erf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_spherical_bessel_j0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_new_full_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sparse_sampled_addmm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_legendre_polynomial_p_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sigmoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_erfc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_empty_permuted_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_squeeze_multiple_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_angle_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_split_with_sizes_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace___rmod___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_std_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_max_unpool2d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_abs_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace__native_batch_norm_legit_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_max_unpool3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_to_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_lu_unpack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_multi_head_attention_forward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_eq_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_column_stack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_alpha_dropout_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_resize_as__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_blackman_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_conv_transpose3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_shifted_chebyshev_polynomial_v_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_ndtr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_NumpyCatCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_scaled_modified_bessel_k0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_dsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_multi_dot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_log_ndtr_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_symbolic_repeat_interleave",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_scatter_reduce_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_avg_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_pad_circular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pixel_unshuffle_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpySplitCopyCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_log2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_digamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sort_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_decomposition_interpreter",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_instance_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_lu_unpack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pad_reflect_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_isposinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_std_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_hann_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_constant_blowup",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_expand_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_index_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_tanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_inner_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_min_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_tensordot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_celu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_constant_pad_nd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_double_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_irfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_mode_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_acos_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_rand_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_rand_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_lerp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_rrelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_lu_unpack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_batch_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_byte_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_vander_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cholesky_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_view_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_le_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_ihfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_eq_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_chebyshev_polynomial_v_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out___rsub___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_flip_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_embedding_bag_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_floor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_randn_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace__segment_reduce_offsets_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_all_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_mul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_view_as_complex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signal_windows_hann_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signal_windows_general_cosine_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_lstsq_grad_oriented_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_chebyshev_polynomial_t_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_softshrink_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_polygamma_polygamma_n_2_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_isneginf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sparse_mm_reduce_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_bool_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_floor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_chebyshev_polynomial_v_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorFake.test_resnet18_backward_trace",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_trunc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logical_or_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_addcdiv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logspace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_ones_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_cosine_embedding_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_feature_alpha_dropout_without_train_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_float_power_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_return_symint",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_det_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_isneginf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_eigvals_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_NumpySplitCopyWithIntCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_rfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_max_reduction_with_dim_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_short_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_isclose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_dstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_gaussian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cfloat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_svdvals_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_vdot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_softmax_with_dtype_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_new_empty_strided_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_upsample_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_ldl_factor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_fftshift_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_polygamma_polygamma_n_2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cross_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_vsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_norm_fro_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_geometric_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_l1_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sgn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_scatter_reduce_sum_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_clamp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_copysign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_new_full_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_interpolate_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_resize_as__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_atleast_1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_rfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_max_reduction_with_dim_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_repeat_interleave_unbacked_output_size",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_trapz_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_log_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_huber_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_avg_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_ctc_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_std_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_unflatten_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_heaviside_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_grid_sample_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_einsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_uniform_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_std_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_expand_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_gather_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_var_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_as_strided_partial_views_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_i0e_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_blackman_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_jiterator_4inputs_with_extra_args_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cartesian_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sinh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_var_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_multi_dot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_lt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_unique_consecutive_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mvlgamma_mvlgamma_p_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_randn_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_NumpySplitCopyCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_zeta_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_softsign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_flatten_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_rfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signal_windows_gaussian_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_solve_triangular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_index_reduce_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_resnet18_backward_trace",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_max_reduction_no_dim_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_avg_pool1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_relu_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logical_or_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_exp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_alpha_dropout_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_searchsorted_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_new_empty_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_NumpyMulCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_bessel_j1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_ldl_factor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_ihfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_mvlgamma_mvlgamma_p_3_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_randint_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_round_decimals_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_addcdiv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_lu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_std_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_tril_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_proxy_tensor_mode_with_decomp_table_preserves_proxy",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_var_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_baddbmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_vecdot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_conv_transpose2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_float_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive___rsub___cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_isolated_graphmodule",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_norm_fro_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_div_floor_rounding_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_block_diag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_byte_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_igamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_asin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_pinv_singular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_hfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_huber_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_char_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_upsample_nearest_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_to_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_exponential_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_max_pool2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_log2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sparse_sampled_addmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_allclose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_count_nonzero_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_ones_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_cumsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_expand_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_reshape_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_polygamma_polygamma_n_3_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_dstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_atleast_1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_std_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_reciprocal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_resize_as__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_mvlgamma_mvlgamma_p_1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_movedim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_movedim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_argmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_softmin_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_diagonal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_ge_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_native_layer_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_expm1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_isin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_gelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_softshrink_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_hstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_round_decimals_neg_3_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_renorm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_solve_triangular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_flipud_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_full_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_nll_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_remainder_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_stack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_clamp_min_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive___rmatmul___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_conv2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_diagflat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_outer_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_clone_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_corrcoef_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_meshgrid_variadic_tensors_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_zeros_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_cosine_similarity_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_grid_sampler_2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_index_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_multilabel_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_conv_transpose3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_trapezoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_as_strided_partial_views_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_hfftn_cpu_float32",  # known py38 fail
    "TestFakeProxyTensor.test_free_fake",  # known py38 fail
    "TestGenericProxyTensorReal.test_decomp_of_capture",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_diagonal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_tile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_NumpyCubeCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_half_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_chebyshev_polynomial_u_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_randn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_unsqueeze_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_view_as_complex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_square_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_logsigmoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nanmedian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_lstsq_grad_oriented_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_acos_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_solve_triangular_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_le_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_int_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_unique_consecutive_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_norm_subgradients_at_zero_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_T_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_bool_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nan_to_num_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_triu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_hsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_jiterator_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logical_xor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_full_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_interpolate_nearest-exact_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_positive_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_block_diag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_log1p_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_native_dropout_backward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_round_decimals_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_bucketize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signal_windows_hann_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_isinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_exp2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_split_list_args_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_hardtanh_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_unsafe_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_hermite_polynomial_he_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out___rmatmul___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_tril_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_addr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_mode_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_shifted_chebyshev_polynomial_u_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_dropout2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_std_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_zeta_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_embedding_bag_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_split_list_args_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_constant_pad_nd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_hfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_legendre_polynomial_p_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_zero__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_scatter_reduce_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_ihfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_angle_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_searchsorted_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_shifted_chebyshev_polynomial_w_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_dist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_hermite_polynomial_h_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_scatter_reduce_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logdet_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cosh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_vector_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_ldl_factor_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_det_singular_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_zeros_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_movedim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_lu_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive___radd___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_var_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_addmm_decomposed_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_complex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_multi_head_attention_forward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorSymbolic.test_amp_cache",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_baddbmm_cpu_float32",  # known py38 fail
    "TestFakeProxyTensor.test_meta",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nan_to_num_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_ifftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_narrow_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_tensordot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_max_pool2d_with_indices_backward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorSymbolic.test_scalar_device",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_amin_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_unbacked_unification",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_eigvalsh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive___rsub___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_diagonal_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_aminmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_scatter_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_new_ones_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_cholesky_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_multilabel_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_view_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nanquantile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_combinations_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_vecdot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_view_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_trace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_xlogy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_adaptive_max_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_lu_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_kthvalue_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_constant_pad_nd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_div_trunc_rounding_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_tanhshrink_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nanquantile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_i1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_max_pool2d_with_indices_backward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_multi_dot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_polar_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_scatter_reduce_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cartesian_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_pad_replicate_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_hsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_conv_transpose1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_var_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_upsample_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestSymbolicTracing.test_invalidate_nonzero",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_clamp_max_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_squeeze_multiple_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_bilinear_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_chalf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cauchy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_softmin_with_dtype_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_new_empty_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_constant_proxy_tensor_mut",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_movedim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_exponential_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_clone_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_jiterator_binary_return_by_ref_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorSymbolic.test_strides",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_acosh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_inv_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nextafter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_ihfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_NumpyCubeCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_t_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_erfinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_std_mean_unbiased_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_arange_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_softsign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_kron_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_isnan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_i0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_polygamma_polygamma_n_4_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_svd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_resolve_conj_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_scatter_reduce_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_any_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_ldl_factor_ex_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_pixel_unshuffle_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_fft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_householder_product_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logdet_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linspace_tensor_overload_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_conj_physical_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_chebyshev_polynomial_u_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_scatter_reduce_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_max_pool2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_matrix_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_einsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_real_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_decomp_of_capture",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_acosh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_polar_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_dot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_multi_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_scaled_dot_product_attention_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_view_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_grid_sample_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_lgamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sub_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_xlog1py_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive___rdiv___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cumulative_trapezoid_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_positive_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_conv3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_exp_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_constant_blowup",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_max_reduction_no_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_NumpySortCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_tensor_split_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_make_fx_reentrant_dispatch",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_interpolate_trilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_norm_inf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_trace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_zeros_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_normal_in_place_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_addmv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_laguerre_polynomial_l_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_atanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_item_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_scatter_reduce_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_randn_like_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_constant_specialization",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_isreal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mv_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_decomp_of_capture",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_ifftn_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_pre_dispatch_linear",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_matrix_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_max_unpool3d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sinc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_to_sparse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_group_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_histogram_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_rsub_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_mish_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_pdist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_diagonal_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_quantile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_diagonal_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_max_unpool2d_grad_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_dot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_conj_physical_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_hfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_double_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_view_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_renorm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_matrix_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_dropout2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_transpose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_diag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_tanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out__segment_reduce_lengths_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_true_divide_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_asin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_leaky_relu_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cummin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_topk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_dropout_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_polygamma_polygamma_n_4_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sparse_mm_reduce_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_gaussian_nll_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_ceil_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_isfinite_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_diagonal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_ldl_factor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_avg_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_copysign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_bucketize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nanmean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_irfftn_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_debug_interpreter",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_bessel_y0_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_tensor_constants",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_i1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_addbmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_acos_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_exp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_pinv_hermitian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_jiterator_2inputs_2outputs_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_chebyshev_polynomial_v_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_multilabel_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_unique_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_baddbmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_max_unpool1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_eq_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_new_empty_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_interpolate_nearest-exact_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_index_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_prelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_select_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_trapz_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_abs_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_geometric_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_layer_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_take_along_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nonzero_static_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_index_reduce_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_floor_divide_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_log1p_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_airy_ai_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sgn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_randint_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_exponential_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_silu_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_fft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_view_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_renorm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace___rmatmul___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_max_unpool2d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_minimum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_argmax_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_broadcast_shapes",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_hypot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_max_pool1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_atan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_conv1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_lu_factor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_new_ones_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_pre_dispatch_mode_stack",  # known py38 fail
    "TestGenericProxyTensorFake.test_pickle_issue89626",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_pixel_shuffle_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_adaptive_max_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_count_nonzero_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_multinomial_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive__segment_reduce_offsets_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_jiterator_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signal_windows_cosine_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_mish_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_shifted_chebyshev_polynomial_t_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_pad_reflect_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_var_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_where_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_flatten_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive__softmax_backward_data_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_embedding_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_int_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logsumexp_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_inplace_metadata",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linspace_tensor_overload_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_elu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_softsign_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_pinv_singular_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_fft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_lgamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_geometric_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_native_dropout_backward_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_frac_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_pre_dispatch_no_grad",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_erfc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_ndtri_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_interpolate_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_logsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_diagonal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_float_power_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_interpolate_area_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorFake.test_mode_tracing_factory_function",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_acosh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_new_zeros_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_adaptive_max_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_polygamma_polygamma_n_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_cholesky_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_conv_transpose1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_log_normal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_leaky_relu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pairwise_distance_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_histc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_normal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_new_empty_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_angle_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_digamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_square_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_erfinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cauchy_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_sym_storage_offset",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_gradient_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_i0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_chebyshev_polynomial_w_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_round_decimals_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_addcmul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_jiterator_2inputs_2outputs_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logical_xor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_reciprocal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_i1e_cpu_float32",  # known py38 fail
    "TestFakeProxyTensor.test_fused_adam",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_kthvalue_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_adaptive_avg_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_pow_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_slice_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_to_sparse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_ravel_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_max_pool3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_max_reduction_no_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_acos_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_ormqr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_column_stack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_randint_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_histogramdd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cosh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_conv1d_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_arange_unbacked_output_size",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_gaussian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_hardtanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_lu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_view_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sinc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_constant_pad_nd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_pad_circular_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_mv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_argwhere_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_conj_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_log_ndtr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive___rmod___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_hermite_polynomial_h_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_cumsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_scaled_dot_product_attention_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_unfold_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_as_strided_scatter_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_min_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_std_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_native_dropout_backward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_NumpyMulCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_entr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_resize__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_prelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_poisson_nll_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_eigvals_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_make_fx_overloads",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_margin_ranking_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nan_to_num_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_true_divide_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpyNonzeroCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_irfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_lstsq_grad_oriented_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_unique_consecutive_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_inplace_metadata",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_cosine_embedding_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_lt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_tensorsolve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_softmin_with_dtype_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_hstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_var_mean_unbiased_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_adaptive_avg_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_float_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cdouble_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cosh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_entr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_ifftshift_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_slogdet_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_index_put_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_zeros_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_empty_permuted_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logdet_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fliplr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_maximum_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_allclose",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_unfold_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_round_decimals_0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_atan2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_vector_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logical_xor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_binary_cross_entropy_with_logits_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_ihfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_pad_replicate_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_tile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_isin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_batch_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_layer_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_unfold_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_size_with_tensor",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_interpolate_trilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_grid_sample_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_norm_nuc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_kl_div_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_i1e_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_lu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_full_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_eye_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_adaptive_max_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_diagonal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_cumsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_searchsorted_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_var_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_matrix_power_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_double_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_geqrf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_qr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_logsigmoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logaddexp2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_rad2deg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_cross_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_transpose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mH_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_make_fx_model_fwd_bwd",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_view_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_H_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_div_no_rounding_mode_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_expand_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_ifft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_isnan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_mv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_randn_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_view_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_addr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_vecdot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_diagonal_copy_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_proxy_tensor_mode_with_decomp_table_preserves_proxy",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_shifted_chebyshev_polynomial_t_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_ctc_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_lu_factor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sigmoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_binary_cross_entropy_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_erfinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_triangular_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_qr_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_int_input",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_rfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_NumpyCatCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_relu6_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_inner_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_modified_bessel_k1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_poisson_nll_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cos_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_repeat_interleave_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_silu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_smooth_l1_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_broadcast_shapes_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_count_nonzero_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_NumpyNMSCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_avg_pool2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_avg_pool3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_view_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_min_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_norm_fro_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_slice_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_narrow_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_geqrf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_msort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_norm_subgradients_at_zero_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive___rmul___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_lstsq_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_modified_bessel_i0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_zeros_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_einsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_isreal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_norm_subgradients_at_zero_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_index_put_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_min_reduction_with_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_threshold_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_diagonal_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_hann_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_log_ndtr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_interpolate_nearest_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_hamming_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_mvlgamma_mvlgamma_p_1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace__upsample_bilinear2d_aa_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linspace_tensor_overload_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_multi_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_huber_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_laguerre_polynomial_l_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_polygamma_polygamma_n_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_hsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_new_ones_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_squeeze_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_margin_ranking_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_mT_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_ndtri_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_quantile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_det_singular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_quantile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_ones_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_std_mean_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_diagonal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_unfold_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_jiterator_binary_return_by_ref_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_scatter_reduce_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_tensordot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_feature_alpha_dropout_with_train_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nanmean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_zeta_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_polygamma_special_polygamma_n_0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nansum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_frac_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_hardswish_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_ihfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cumulative_trapezoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_triplet_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_max_unpool1d_grad_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_ifft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_normalize_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_pca_lowrank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_bessel_j0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_qr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_inv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_hardsigmoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_split_with_sizes_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_empty_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_std_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_batch_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_isnan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_lstsq_grad_oriented_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_block_diag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_matrix_rank_hermitian_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logcumsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_hardsigmoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_hfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_feature_alpha_dropout_without_train_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_outer_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_randn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_general_cosine_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sinc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_atan2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_modified_bessel_i0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_conv2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sum_to_size_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cross_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_matmul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_eye_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_jiterator_2inputs_2outputs_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_diag_embed_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_softmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_broadcast_tensors_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_heaviside_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_atan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_lstsq_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_partial_decomp",  # known py38 fail
    "TestSymbolicTracing.test_rmethod",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_multilabel_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_clamp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_max_pool2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_xlog1py_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_bernoulli_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_eigvals_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_fft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_scaled_dot_product_attention_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_ravel_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_pad_reflect_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logcumsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_bessel_y0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_embedding_bag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_solve_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_ge_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_logsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_square_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_diagonal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_upsample_nearest_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_shifted_chebyshev_polynomial_w_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signal_windows_nuttall_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_log_softmax_with_dtype_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_matmul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_hann_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_argmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_conj_physical_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_smooth_l1_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_lu_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_scaled_modified_bessel_k1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_triplet_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linspace_tensor_overload_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_mm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_huber_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_entr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_bessel_j1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_svdvals_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive___rsub___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_renorm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_randint_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_scatter_reduce_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_hypot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_adaptive_avg_pool2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_eig_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_cosine_embedding_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_legendre_polynomial_p_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_interpolate_linear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_min_reduction_no_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_scalar_tensor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nextafter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_chebyshev_polynomial_w_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_l1_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorReal.test_tensor_constants",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_ifft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_hardsigmoid_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_tensor_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_jiterator_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_lu_factor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_pad_replicate_negative_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_cholesky_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_scatter_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_as_strided_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logical_xor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_where_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_i1e_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_byte_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_scatter_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive___rpow___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_trapezoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_var_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_frexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_short_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_make_fx_model_fwd_bwd",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpyNMSCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_NumpySplitCopyWithIntCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_embedding_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_softshrink_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_max_unpool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_index_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_lgamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_aminmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_diff_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_pdist_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_isfinite_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_laguerre_polynomial_l_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_feature_alpha_dropout_with_train_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_svd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive__segment_reduce_offsets_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_rsub_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_hfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_atleast_1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nextafter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pad_replicate_negative_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_chebyshev_polynomial_t_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_trapezoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_mT_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive___radd___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_all_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_max_unpool1d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_conv_transpose1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_polygamma_polygamma_n_2_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nanmean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cfloat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_irfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_add_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_mega_guard",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_fractional_max_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_div_floor_rounding_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_trapezoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_argmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_isclose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_log_softmax_with_dtype_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_NumpyTakeCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_modified_bessel_i1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out__upsample_bilinear2d_aa_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_NestedMapControlflowOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cholesky_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_normal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_frac_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_ifftshift_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_reshape_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_hfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_index_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_matrix_rank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_scatter_reduce_amin_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_zero__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_lu_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_diff_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_mvlgamma_mvlgamma_p_5_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_tanhshrink_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_atleast_2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_cosine_similarity_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_histc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_view_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nanmedian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_NumpyNMSCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_new_empty_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_new_full_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_exp2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sgn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_tensorinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_normal_number_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive___rmul___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_as_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_pinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_softplus_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signal_windows_nuttall_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_baddbmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_chunk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_outer_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logcumsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_addcmul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_bartlett_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_frac_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_hardtanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_slogdet_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_pinverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signal_windows_exponential_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_dstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_matrix_rank_hermitian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_narrow_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_feature_alpha_dropout_without_train_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_soft_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorFake.test_empty_like_doesnt_burn_in_defaults",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_slice_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_MapControlflowOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_shifted_chebyshev_polynomial_v_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_rfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_feature_alpha_dropout_with_train_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_nll_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_scatter_reduce_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_stack_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_guard_lowerbound_range_refinement",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_NumpyCubeCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_adaptive_max_pool3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace___rsub___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_ones_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_scalar_tensor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_scalar_tensor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_isposinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_gradient_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_pad_replicate_negative_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out___rmod___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_dropout2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signal_windows_kaiser_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_isposinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_unsqueeze_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_multinomial_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sparse_mm_reduce_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_T_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_logsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_rfft_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_neg_shape",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_max_pool3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_index_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_aminmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_argmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_isin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_pinv_singular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_fft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_flip_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_NumpySortCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_hardswish_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_ifft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nansum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_max_reduction_with_dim_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_as_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_irfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_flipud_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_lu_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_multilabel_soft_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_index_put_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_grid_sample_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_interpolate_linear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_allclose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_silu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_vector_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_solve_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_celu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_igammac_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_var_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_lu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_round_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_empty_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_celu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_kl_div_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_matrix_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_as_strided_partial_views_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_resize_as__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_ihfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_index_put_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_i1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logical_not_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_hypot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_avg_pool1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_bmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_elu_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_trace_subclasses",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_eq_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_smooth_l1_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fliplr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_unsafe_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_clamp_max_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signal_windows_general_hamming_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_dist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_clamp_min_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_repeat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_max_unpool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_addmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_scatter_reduce_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signal_windows_bartlett_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_log10_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_conv_transpose2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_unsqueeze_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_matmul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logical_and_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_native_layer_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_glu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive__native_batch_norm_legit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_hermite_polynomial_he_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_softmax_with_dtype_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_gaussian_nll_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_polygamma_polygamma_n_1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_instance_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_to_sparse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_zero__cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_view_divisibility_unbacked",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_embedding_bag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_roll_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_kl_div_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_conv_transpose2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_soft_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_irfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_randint_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_zeros_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_jiterator_unary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_inv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_celu_cpu_float32",  # known py38 fail  # noqa: B950
    "TestSymbolicTracing.test_item",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_softsign_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_svd_lowrank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_scatter_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cholesky_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_vsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_outer_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_ifftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_unsafe_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_adaptive_max_pool1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_hfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_slogdet_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_vsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_addcdiv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_hermite_polynomial_h_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_svd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_conv2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace__softmax_backward_data_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_legendre_polynomial_p_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_normalize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signal_windows_bartlett_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_pca_lowrank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_var_mean_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_diag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_fft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_ihfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_exponential_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sgn_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_split_unbacked_sizes",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_normal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_conv1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_rot90_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_trace_subclasses",  # known py38 fail
    "TestSymbolicTracing.test_expand",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_stft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_normalize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_MapControlflowOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_jiterator_2inputs_2outputs_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_index_put_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_erfcx_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sum_to_size_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_atanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_var_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_std_mean_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_jiterator_unary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_ldl_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive___rmatmul___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_real_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_count_nonzero_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_linear_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_mm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_H_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_acos_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_histogramdd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_remainder_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mode_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_batch_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cumulative_trapezoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_vander_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_vsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_gather_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_adaptive_avg_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sub_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_unbacked_unify_guard_transitivity",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_normal_number_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_corrcoef_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_mvlgamma_mvlgamma_p_5_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_scaled_modified_bessel_k0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_multi_head_attention_forward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_conj_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_matrix_rank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_pixel_unshuffle_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_binary_cross_entropy_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nan_to_num_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_randint_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_entr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_bessel_y1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_chalf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_heaviside_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_native_dropout_backward_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_var_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_multi_margin_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive___rmatmul___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_floor_divide_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_item_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_scatter_reduce_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_general_cosine_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_binary_cross_entropy_with_logits_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_eye_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_hardshrink_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_ctc_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_polygamma_polygamma_n_0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_real_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_zeta_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cdist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_log10_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_NumpyTakeCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_gather_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_tanhshrink_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive__segment_reduce_lengths_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_int_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_tanhshrink_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_inner_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_outer_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_float_power_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_fill_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_inplace_metadata",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_squeeze_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_fractional_max_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_threshold_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_unsafe_chunk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_broadcast_shapes_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_igamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_round_decimals_0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_clamp_max_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_val_metadata_mutation",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_normalize_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_log_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_hermite_polynomial_he_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_polygamma_polygamma_n_2_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_make_fx_model_double_param",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_kaiser_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linspace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_solve_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_rot90_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_fftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_nll_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out___rdiv___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_NumpyNMSCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_clamp_max_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_max_pool1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_addmv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_round_decimals_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_new_full_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_polygamma_polygamma_n_0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_tile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_gaussian_nll_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_atleast_3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_expand_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_dsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_householder_product_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_kthvalue_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_binary_cross_entropy_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_zeros_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_int_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_float_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_isin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_div_no_rounding_mode_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_xlogy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_chalf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_stft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_polygamma_polygamma_n_1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_var_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_fractional_max_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cfloat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_max_unpool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_slice_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_qr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_layer_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_abs_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_any_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_amp_cache",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_ifft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_interpolate_trilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_maximum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_eigvalsh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_irfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_max_unpool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_embedding_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_bessel_j0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_baddbmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_chunk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_polygamma_polygamma_n_0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_alpha_dropout_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_floor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_gaussian_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_chebyshev_polynomial_u_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_eq_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_modified_bessel_k1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_log_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_fftshift_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_selu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_qr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_eye_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logspace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_meshgrid_list_of_tensors_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_log_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logical_not_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_prod_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_tensor_constants",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_ldexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_uniform_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_diag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_erfcx_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_erfcx_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_igamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_min_reduction_no_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_chebyshev_polynomial_v_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive__softmax_backward_data_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_combinations_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cholesky_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_unfold_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_normal_in_place_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_pairwise_distance_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_empty_permuted_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_cosine_embedding_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_vstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_as_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_split_with_sizes_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_chebyshev_polynomial_t_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_argmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_max_unpool3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_softmin_with_dtype_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_normal_in_place_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_bfloat16_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_vdot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out___rpow___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_log_softmax_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_isclose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_meshgrid_list_of_tensors_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_general_hamming_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_trunc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_adaptive_max_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_real_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sparse_sampled_addmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_fft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_NumpyMulCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_histogram_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_cosine_similarity_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_byte_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_roll_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_atleast_2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace___rdiv___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_std_mean_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_new_empty_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_minimum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_max_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_addr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_conv2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_igammac_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_native_layer_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_adaptive_avg_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_lgamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_softmin_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_lu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_floor_divide_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_eig_cpu_float32",  # known py38 fail
    "TestRealProxyTensor.test_error_on_data_dependent_ops",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_lu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sub_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_broadcast_tensors_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_unsqueeze_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_slice_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_ihfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_std_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_broadcast_to_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logspace_tensor_overload_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_addmm_decomposed_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_tril_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_randint_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_det_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signbit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_NumpyCatCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_cumsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_cholesky_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_round_decimals_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_dsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_put_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signal_windows_cosine_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorSymbolic.test_proxy_tensor",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_ldl_factor_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_matrix_exp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_pow_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_interpolate_nearest_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_lu_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_atleast_3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cumsum_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_constant_random",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_irfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_matrix_exp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_igamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_aminmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_std_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_i1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_xlogy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sinh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_isnan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_unique_consecutive_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_cholesky_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_scalar_device",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_index_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_chebyshev_polynomial_u_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_ceil_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_kron_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_qr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_view_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_hermite_polynomial_h_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sigmoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_tan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_as_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_isin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cholesky_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_flip_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_resize_from_zero",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sparse_mm_reduce_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_take_along_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_adaptive_max_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nansum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_celu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_det_singular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_erfc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_chebyshev_polynomial_t_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_group_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_t_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_log2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_lt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_rsqrt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_linear_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_group_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_NumpyViewCopyCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_spherical_bessel_j0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_conv_transpose1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_complex_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_make_fx_with_custom_tracer_preserving_nn_module_stack",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_scatter_reduce_prod_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_matrix_exp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpyCatCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nonzero_static_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_min_reduction_no_dim_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_max_pool2d_with_indices_backward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorFake.test_make_fx_model_fwd_bwd_wgtupdate",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_split_with_sizes_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_where_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_repeat_interleave_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_div_trunc_rounding_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_narrow_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_index_reduce_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_adaptive_avg_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_hinge_embedding_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_bucketize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_stft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signal_windows_exponential_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_i1e_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_mvlgamma_mvlgamma_p_5_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_fftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_contiguous_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_geqrf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_max_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_svdvals_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_std_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_NumpySplitCopyCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_repeat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_rad2deg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive___rpow___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_unsafe_chunk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_msort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_acosh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_isfinite_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_scaled_modified_bessel_k1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_addbmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_adaptive_max_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_contiguous_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_le_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_selu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_rsub_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_chebyshev_polynomial_v_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_norm_inf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_diagflat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_spherical_bessel_j0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_bessel_y1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_hinge_embedding_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_diagonal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_feature_alpha_dropout_without_train_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_dstack_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_make_fx_model_fwd_bwd_wgtupdate",  # known py38 fail
    "TestFakeProxyTensor.test_issue82547",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_hfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_shifted_chebyshev_polynomial_w_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_expm1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_conv3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_interpolate_trilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_rsqrt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_fft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_matrix_exp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_vdot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mT_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_max_unpool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_div_no_rounding_mode_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_new_zeros_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_triplet_margin_with_distance_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_native_batch_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mode_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_svd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_polygamma_polygamma_n_4_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cartesian_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive__segment_reduce_lengths_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_pinv_singular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_deg2rad_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_polar_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_corrcoef_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_chebyshev_polynomial_t_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_fractional_max_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_pca_lowrank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_histogramdd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_tile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_long_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_conv_transpose3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_select_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_narrow_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_any_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_quantile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_pad_constant_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_lerp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_tanhshrink_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_view_copy_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_pre_dispatch_no_grad",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_gelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_matrix_power_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_hermite_polynomial_h_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_round_decimals_neg_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_mvlgamma_mvlgamma_p_1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_ihfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_norm_subgradients_at_zero_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_householder_product_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_half_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_conv_transpose1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_cross_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cumulative_trapezoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_expand_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_rfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_meshgrid_list_of_tensors_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_kl_div_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_xlog1py_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_flatten_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logical_and_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_nonidentity_transitive_guards",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_std_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_uniform_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_guard_upperbound_range_refinement",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_unfold_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_std_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logical_or_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_T_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_jiterator_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_ldexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_hfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_new_empty_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive__upsample_bilinear2d_aa_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_topk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_pinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_trace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_max_unpool3d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_angle_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_bmm_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_scalar_device",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_ihfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_minimum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_empty_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_fractional_max_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_cosine_embedding_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_dstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_index_reduce_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_dropout3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_svd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pad_constant_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_hardsigmoid_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorSymbolic.test_constant_proxy_tensor_mut",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_addmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_igammac_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_index_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_lu_factor_ex_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_kaiser_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_ifft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_grid_sampler_2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nanmedian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_div_trunc_rounding_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_long_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logical_not_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_randint_like_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_make_fx_reentrant_dispatch",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_decomposition_interpreter",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_argmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_deg2rad_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_div_no_rounding_mode_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_gelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_index_reduce_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_div_no_rounding_mode_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_std_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_tensordot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linspace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_instance_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_resolve_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cummax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_narrow_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_empty_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signal_windows_blackman_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_gt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_constant_pad_nd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_triu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_hypot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_where_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_native_layer_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_max_unpool1d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_avg_pool1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_ormqr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_mH_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_max_unpool3d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_adaptive_max_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_slice_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_adaptive_avg_pool1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cholesky_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_hypot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cross_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_asin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpyCubeCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_min_reduction_no_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_flipud_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_normalize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_chebyshev_polynomial_u_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nonzero_static_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_cross_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_eig_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_triplet_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_triplet_margin_with_distance_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_atleast_2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_feature_alpha_dropout_with_train_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_mse_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nanmedian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_remainder_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_shifted_chebyshev_polynomial_w_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_vecdot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_geqrf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_fftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_floor_divide_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_atan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_rot90_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_softmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_eye_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_index_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_movedim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_allclose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_hfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_histc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive___getitem___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_conj_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_pad_constant_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_mT_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_TripleNestedMapControlflowOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_softmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_unsafe_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_meshgrid_variadic_tensors_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_char_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_mse_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_hardshrink_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sqrt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_clamp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mT_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_addcmul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_squeeze_multiple_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_argmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_qr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive___rpow___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out__native_batch_norm_legit_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cauchy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_poisson_nll_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_max_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_interpolate_area_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cosh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_copysign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_max_unpool1d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_contiguous_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_pdist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_mul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_hfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive__native_batch_norm_legit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_item_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_char_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_as_strided_partial_views_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorFake.test_partial_decomp",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_softmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_frexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logaddexp2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_fft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nextafter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_round_decimals_neg_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_layer_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_stack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_glu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_H_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_log_normal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_isnan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_bessel_y0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_equal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_ndtr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_flip_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_bilinear_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_gt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_log1p_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_new_zeros_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_std_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_conv_transpose2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cummin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_softsign_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_partial_decomp",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_eigh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_atanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_slogdet_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_log_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_NumpySortCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_polygamma_polygamma_n_4_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_NestedMapControlflowOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_polygamma_special_polygamma_n_0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_exp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_squeeze_multiple_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_histogram_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_stft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_inv_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_broadcast_to_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_diagonal_scatter_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_pre_dispatch_functionalization_view_op",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_interpolate_area_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_unbacked_slice",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_lstsq_grad_oriented_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sum_to_size_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive___rmod___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_scaled_modified_bessel_k0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_pinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_shifted_chebyshev_polynomial_u_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_permute_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_matrix_power_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_reciprocal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_argmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_polygamma_polygamma_n_0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_empty_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_isinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_to_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cauchy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_unfold_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_tensor_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_erfc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_expm1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_tanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_inv_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_real_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_solve_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_interpolate_bicubic_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_unfold_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cov_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_resize_as__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_randn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_split_with_sizes_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_reciprocal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_as_strided_scatter_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_metadata",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_interpolate_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_logsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_kron_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_erfcx_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_searchsorted_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_ifft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_nuttall_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_threshold_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_vecdot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_max_reduction_with_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_addmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cummax_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_varargs",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_polygamma_polygamma_n_2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_conv2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_mvlgamma_mvlgamma_p_3_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorFake.test_constant_random",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_atleast_1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_bucketize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_cholesky_ex_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_i0e_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_feature_alpha_dropout_without_train_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive__segment_reduce_lengths_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_isposinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_put_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_svd_lowrank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_normalize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_atanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_jiterator_unary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_H_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_t_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_make_fx_reentrant_dispatch",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_i0e_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_eigh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_narrow_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_log_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_mse_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_var_unbiased_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_pre_dispatch_functionalization",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_einsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_meshgrid_variadic_tensors_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_zeros_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_unique_consecutive_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_ge_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_rsqrt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_permute_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_transpose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_matrix_rank_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_view_as_complex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_squeeze_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_interpolate_area_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_jiterator_binary_return_by_ref_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_rsqrt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_gradient_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_softmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_chebyshev_polynomial_w_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorSymbolic.test_make_fx_simple",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cholesky_inverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_det_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_var_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_unflatten_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_kl_div_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_lu_factor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_softmax_with_dtype_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sub_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_block_diag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_multi_dot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_einsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fmod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_new_zeros_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_bool_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_modified_bessel_k1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_lerp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_scatter_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_adv_index_batch",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_deg2rad_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_lu_factor_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cholesky_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_avg_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_randint_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_broadcast_shapes_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_matrix_power_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_relu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_shifted_chebyshev_polynomial_v_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_lu_factor_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_put_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logspace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_nll_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_modified_bessel_k1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_pinv_hermitian_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_leaky_relu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_relu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cholesky_inverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_erf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_broadcast_tensors_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_grid_sampler_2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_addcdiv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_arange_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_diagonal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_shifted_chebyshev_polynomial_u_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_threshold_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_round_decimals_0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_hstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_as_strided_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_hinge_embedding_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_digamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_addmm_decomposed_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_poisson_nll_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_atan2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_rand_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_diagonal_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_NumpySplitCopyCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_bernoulli_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_gradient_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_diagonal_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_smooth_l1_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_empty_permuted_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_unfold_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_exponential_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_where_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_gaussian_nll_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_repeat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cfloat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_tile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_ihfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_asin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_adaptive_max_pool2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_new_ones_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_pre_dispatch_linear",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_erfcx_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_squeeze_multiple_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cummax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_tensor_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_unsafe_chunk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cdouble_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_normalize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_clamp_min_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_lu_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_kron_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sigmoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_log10_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_std_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_embedding_bag_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_normalize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_rsub_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_scaled_dot_product_attention_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nanquantile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sinh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_normal_number_mean_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_dynamic_pointwise_scalar",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_lu_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_dot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_any_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace___rpow___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_relu6_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_modified_bessel_k0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pdist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logical_or_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_vstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_float_power_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_deg2rad_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_max_reduction_with_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_eigvals_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_native_batch_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linspace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_NumpyCubeCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_NumpyNonzeroCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_cumsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_qr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_resize__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_max_unpool2d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_log10_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_max_unpool3d_grad_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_unbind_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mvlgamma_mvlgamma_p_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_broadcast_tensors_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_round_decimals_0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_positive_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_resolve_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_cosine_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_hsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_diagonal_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_mish_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cholesky_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_max_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_eigh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_scaled_modified_bessel_k1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_vdot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_lu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_true_divide_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_block_diag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_abs_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_any_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_frexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_contiguous_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_flipud_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_glu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_ravel_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_diff_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_vstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_cross_entropy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_NumpyViewCopyCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_positive_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_empty_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_log_ndtr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_tensorinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_silu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nonzero_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_chunk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_logsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_isreal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_svd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_resize__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_softmax_with_dtype_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_softmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_TripleNestedMapControlflowOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_hamming_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_erf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_index_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_long_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_expm1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logical_and_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_pixel_shuffle_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_general_hamming_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signal_windows_general_cosine_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linspace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_svdvals_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_atleast_3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_dropout2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_inv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_unique_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_pinv_singular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_l1_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_argsort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_airy_ai_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_TripleNestedMapControlflowOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_full_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out___rmul___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_relu6_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_H_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_prelu_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_modified_bessel_i0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signal_windows_gaussian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_pow_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_flatten_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_item_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_jiterator_binary_return_by_ref_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_pinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_empty_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_xlogy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_mode_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_argwhere_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nonzero_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_histogram_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_empty_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_arange_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_tensor_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cdouble_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_slice_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cross_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_svd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sinc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_addmm_decomposed_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_irfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_avg_pool2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_multinomial_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_NumpyNonzeroCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_addbmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_transpose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sinc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_multilabel_soft_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_multilabel_soft_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestFakeProxyTensor.test_alias",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_rad2deg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_kaiser_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_logsigmoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_bessel_y0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_put_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_addr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_interpolate_bicubic_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sqrt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_stft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cumsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_chalf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out__segment_reduce_offsets_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_minimum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_lu_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_argwhere_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_empty_permuted_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cartesian_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_scatter_reduce_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_conv1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_embedding_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_native_layer_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_rot90_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_normalize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_upsample_nearest_cpu_float32",  # known py38 fail  # noqa: B950
    "TestSymbolicTracing.test_unary",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_empty_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_addcmul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_atleast_3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_triu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_arange_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_log_ndtr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_full_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_tensor_symfloat",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_narrow_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_take_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_rfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_unique_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive___getitem___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cumsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_eig_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signal_windows_blackman_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fmod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_cond_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_bessel_j1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_det_singular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_norm_inf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_hardshrink_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_grid_sampler_2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_ormqr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_NumpyNonzeroCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_log_softmax_with_dtype_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_softplus_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_jiterator_unary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_norm_inf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_roll_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_adaptive_avg_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_lstsq_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_argmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_huber_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_vector_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_bessel_y0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_normal_in_place_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_lu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_scatter_reduce_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cos_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_ldl_factor_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_pca_lowrank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cov_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_adaptive_max_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_svd_lowrank_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_make_fx_model_double_param",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_var_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_pow_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logspace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_i0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_modified_bessel_i0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_bartlett_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_exponential_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_round_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_svd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_ndtri_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_ormqr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_diagonal_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_margin_ranking_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_log_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fliplr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cholesky_inverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_angle_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_interpolate_linear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_rrelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_all_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cummax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_clamp_max_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logical_xor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nanquantile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_broadcast_to_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_slogdet_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_all_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_split_list_args_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_ormqr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logical_not_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_jiterator_4inputs_with_extra_args_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_broadcast_shapes_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_dist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_gather_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_pairwise_distance_cpu_float32",  # known py38 fail  # noqa: B950
    "TestSymbolicTracing.test_symint_to_tensor",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive__segment_reduce_offsets_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_pinverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_shifted_chebyshev_polynomial_t_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_div_trunc_rounding_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_linear_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_constant_unbind",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_addmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_pad_reflect_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_triangular_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_pinv_hermitian_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_log_softmax_with_dtype_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_round_decimals_neg_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_log1p_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_NumpyTakeCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_interpolate_nearest-exact_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_isfinite_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_dot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_chalf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_upsample_nearest_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_conj_physical_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_empty_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_unsafe_chunk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_hardtanh_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorFake.test_make_fx_simple",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_NumpyNMSCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_searchsorted_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_shifted_chebyshev_polynomial_v_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_fft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_max_unpool1d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_repeat_interleave_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_scaled_modified_bessel_k0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_jiterator_4inputs_with_extra_args_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_NumpyViewCopyCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_adaptive_max_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_modified_bessel_k0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_isreal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_full_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_slice_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive___rmul___cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_setitem_symint",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_irfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_argmin_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_pr_86917",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_zeta_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_chunk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_solve_triangular_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_rad2deg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_interpolate_linear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_ravel_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_vstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_pca_lowrank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_uniform_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_max_pool3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linspace_tensor_overload_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_local_response_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_layer_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_split_list_args_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cummin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_meshgrid_variadic_tensors_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_dropout3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cholesky_inverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_ctc_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_scatter_reduce_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_max_pool2d_with_indices_backward_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cov_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_native_batch_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_frexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sigmoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_pinverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pad_circular_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_erf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nanmedian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cholesky_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_ifftshift_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_local_response_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_min_reduction_with_dim_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_unbind_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_interpolate_linear_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_ceil_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cos_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nonzero_static_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_asinh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_T_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_pad_replicate_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logical_and_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_pad_replicate_negative_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_i0e_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_le_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_spherical_bessel_j0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_slice_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_i0e_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_elementwise_meta_with_sym_numbers",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_scalar_tensor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sgn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mH_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_max_unpool2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_as_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_chebyshev_polynomial_w_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_group_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace__segment_reduce_lengths_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_slice_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_triangular_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_item_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_log_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_upsample_nearest_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_copysign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_ndtr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_matrix_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_lgamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_argsort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_resolve_conj_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_nuttall_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_multinomial_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_lu_unpack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_trapz_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_addmv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_rsqrt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_conv3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_relu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_ceil_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_atleast_2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_max_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_hsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_norm_nuc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_mul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_take_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_norm_fro_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_fftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nonzero_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_binary_cross_entropy_with_logits_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_long_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_mse_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_pixel_unshuffle_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpyTakeCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_repeat_interleave_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_true_divide_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_interpolate_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_ldl_factor_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_stack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_ndtri_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linspace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_short_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nansum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_msort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_blackman_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_diff_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_tan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_det_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_alpha_dropout_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_atan2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_multi_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_cross_entropy_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_le_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_select_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_std_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_cosine_similarity_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorReal.test_resnet18_backward_trace",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_var_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_norm_nuc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cdist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_hfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_modified_bessel_i1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_relu6_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_rfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logaddexp2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_as_strided_partial_views_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_eigvalsh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_isinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_put_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_matrix_rank_hermitian_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_adaptive_avg_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nanmean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_interpolate_nearest_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_vdot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signbit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signal_windows_general_hamming_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_atanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_lerp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_alpha_dropout_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cauchy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_equal_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_strides",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_flipud_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mvlgamma_mvlgamma_p_5_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_bartlett_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_tensorsolve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpyMulCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_scaled_modified_bessel_k1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorSymbolic.test_val_metadata_mutation",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_tensorinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_native_batch_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_narrow_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive__softmax_backward_data_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_isclose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_igammac_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_cosine_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_softplus_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nonzero_static_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_multiply_shape",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_ldl_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_fftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_prelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_histc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_cosine_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_silu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_mm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_tensordot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_trunc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_log_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_conv_transpose3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive___rmod___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_fractional_max_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_matrix_exp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_selu_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_unbind_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cosh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_polygamma_polygamma_n_1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_T_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_ifftshift_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cov_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_triplet_margin_with_distance_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestSymbolicTracing.test_item_to_constructor",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_feature_alpha_dropout_with_train_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_batch_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_pixel_unshuffle_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpySplitCopyWithIntCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_irfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logical_and_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_std_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_ifft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_cumprod_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_pickle_issue89626",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_proxy_tensor_mode_with_decomp_table_preserves_proxy",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_jiterator_4inputs_with_extra_args_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_matrix_rank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_multilabel_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_prelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cov_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_abs_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_bfloat16_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_to_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_histogramdd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sqrt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_diff_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_NumpyMulCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_eig_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out__softmax_backward_data_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_cross_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_exponential_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_pairwise_distance_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_asinh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_bool_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_pad_replicate_negative_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_irfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_conv3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_hinge_embedding_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_isreal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_shifted_chebyshev_polynomial_v_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_flip_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_allclose",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_pre_dispatch_no_grad",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_unsafe_chunk_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_metadata_fresh",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cholesky_inverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_cond_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_hamming_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_isposinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_triangular_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_eigvalsh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_empty_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_ones_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_bmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_hardswish_cpu_float32",  # known py38 fail  # noqa: B950
    "TestGenericProxyTensorReal.test_proxy_tensor",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_householder_product_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_tanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_bessel_j0_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_cpu_scalar_cuda",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_resolve_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_column_stack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_max_pool2d_with_indices_backward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cdist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_irfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_log2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_div_trunc_rounding_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_mH_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_gelu_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_max_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_log_normal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_argmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_select_scatter_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_constant_unbind",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_trace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_dist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_as_strided_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_contiguous_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_dropout_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_argmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_trapz_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_round_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_mean_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_trace_subclasses",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fmod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_ravel_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_diagflat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_airy_ai_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_pinv_hermitian_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_rrelu_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_softshrink_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_reshape_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_mish_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cdist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sqrt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_tensorsolve_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_NumpyCatCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_empty_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_equal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_floor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_trunc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NestedMapControlflowOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_meshgrid_list_of_tensors_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_soft_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_adaptive_max_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_relu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_bessel_y1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_cond_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_interpolate_trilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_jiterator_2inputs_2outputs_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_histogram_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_log_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace___rmul___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_rrelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_binary_cross_entropy_with_logits_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_inv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_conv1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_argmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_jiterator_binary_return_by_ref_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signbit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_remainder_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_t_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_scatter_reduce_amax_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_index_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_tril_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_softmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_interpolate_area_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_atan2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_multi_head_attention_forward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_dropout_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_norm_inf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_unfold_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_conv_transpose2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_interpolate_bicubic_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nanquantile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_atleast_2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_log_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_cholesky_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_spherical_bessel_j0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_resolve_conj_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_gt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_isneginf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_hardtanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_half_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_polygamma_polygamma_n_4_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_unflatten_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_bessel_j0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_expand_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_triplet_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_multinomial_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_polar_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cdouble_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_l1_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_unfold_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_nuttall_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_argmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_trapezoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_exp2_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_binary_broadcast",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_maximum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_modified_bessel_k0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_addbmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_atleast_3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_index_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_leaky_relu_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_normal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_inv_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_inv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_margin_ranking_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_geometric_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_short_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_trapz_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pad_replicate_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out___radd___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_min_reduction_with_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_binary_cross_entropy_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_soft_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_elu_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_ifftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_mv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_ones_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_short_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_normal_in_place_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_ndtri_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_quantile_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_char_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_int_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_linear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_local_response_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_allclose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_atan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_floor_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_pre_dispatch_functionalization_view_op",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_split_list_args_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fmod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cos_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_maximum_cpu_float32",  # known py38 fail
    "TestFakeProxyTensor.test_use_fake_and_tensor",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_ldexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_rot90_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_native_batch_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_reshape_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_clamp_min_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace___radd___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_norm_fro_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_double_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_threshold_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_exp2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_modified_bessel_i0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_hstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_index_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cummin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_char_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_ones_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_std_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_msort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_hardsigmoid_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_half_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_expand_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_lerp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_addcdiv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_mH_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_lu_factor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_maximum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_max_unpool2d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_scalar_tensor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_acosh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logdet_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_laguerre_polynomial_l_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_eigvals_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_normal_number_mean_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logdet_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_max_unpool3d_grad_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_bessel_j1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_full_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_scaled_dot_product_attention_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_ihfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cfloat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_addr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_exponential_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_to_sparse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_ceil_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_roll_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_unbacked_unify_guard",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_rad2deg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_flatten_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive___rdiv___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_solve_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_isinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_rfftn_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_non_symint_size_spec",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_isfinite_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_cross_entropy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nextafter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_clone_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_avg_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_lstsq_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_select_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nanmean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sinh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_bessel_y1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_t_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_reshape_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_multi_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_index_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_gelu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_zero__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logspace_tensor_overload_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_argsort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_full_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_view_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fliplr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_scatter_reduce_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_reshape_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_reshape_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_permute_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_strides",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_asinh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_max_reduction_no_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_det_singular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_lstsq_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_resize__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_mean_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_isolated_graphmodule",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpyViewCopyCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_scatter_reduce_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_round_decimals_neg_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_xlog1py_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_min_reduction_with_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sum_to_size_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_pad_constant_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_erfc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sub_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_asinh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_unique_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_meshgrid_list_of_tensors_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_lu_factor_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_hfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_leaky_relu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_upsample_bilinear_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_norm_subgradients_at_zero_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_gaussian_nll_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_logsigmoid_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_erfinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_avg_pool3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_norm_nuc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_ifft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_scatter_reduce_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_triu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_log2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpySortCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_hermite_polynomial_he_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_tan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_linear_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_polygamma_polygamma_n_1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_kron_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_instance_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_roll_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_upsample_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_full_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_frexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_lu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_vander_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_guard_lowerbound_range_refinement_multivariate",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_svd_lowrank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pixel_shuffle_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_qr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_hinge_embedding_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_qr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_expand_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_igammac_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_allclose_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorFake.test_pre_dispatch_functionalization",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mvlgamma_mvlgamma_p_1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_lu_solve_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorReal.test_pre_dispatch_functionalization",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_shifted_chebyshev_polynomial_t_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_positive_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_glu_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_unfold_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_view_as_complex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_scaled_modified_bessel_k0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_logaddexp2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_full_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_argmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logaddexp2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fliplr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_adaptive_max_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_ne_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_trace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_resize__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_kthvalue_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_ge_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_log_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_irfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cholesky_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_ndtr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_pdist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_all_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_upsample_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_vander_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_modified_bessel_i1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_signal_windows_hamming_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_eigh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_conj_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_div_floor_rounding_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_cholesky_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive___radd___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_softmin_with_dtype_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_pinverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_max_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_addmv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_signbit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_grid_sample_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_randn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_narrow_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_complex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logspace_tensor_overload_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_poisson_nll_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_linalg_eigvalsh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cummax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_hfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_index_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_dropout3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_vstack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_atan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_rfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_vsplit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_unfold_copy_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_reflect_r_over_x",  # known py38 fail
    "TestGenericProxyTensorFake.test_pre_dispatch_functionalization_view_op",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_group_norm_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_unbacked_batch_resnet",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_softmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_ge_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_diag_embed_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_xlogy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_isclose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_cross_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_logcumsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nansum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_isinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_max_unpool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_log_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_det_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_argmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_exp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_native_dropout_backward_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_reshape_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logcumsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_selu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_to_sparse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cdouble_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_cat",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_bmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_ifftshift_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_clamp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logspace_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_NumpySortCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_new_empty_strided_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_geqrf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_fractional_max_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_erf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_corrcoef_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_unflatten_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_max_unpool1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_arange_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_elu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_avg_pool2d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_adaptive_avg_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_topk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_repeat_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_log_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_hardswish_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_softshrink_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_bmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_column_stack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_ne_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_rand_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_min_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_var_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_asinh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_multilabel_soft_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_conj_physical_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_laguerre_polynomial_l_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_fractional_max_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_rfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_jiterator_unary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_rfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_erfinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_fftshift_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_trunc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_bernoulli_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_adaptive_avg_pool1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_renorm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sparse_mm_reduce_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_triangular_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_interpolate_nearest-exact_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_ihfft_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_diag_embed_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_argmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive__upsample_bilinear2d_aa_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_isneginf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_tensorsolve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_var_mean_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_ne_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_local_response_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_cholesky_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_ne_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_unbind_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_cross_entropy_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cumulative_trapezoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_fractional_max_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_unsafe_split_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_logsigmoid_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out___getitem___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_float_power_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_msort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_svdvals_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_pixel_shuffle_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_cond_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_dropout3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_scatter_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_bernoulli_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_rfftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_softmax_with_dtype_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_cross_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_dropout2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_ones_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_heaviside_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_diag_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_cosine_similarity_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_zeros_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_resolve_conj_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_log1p_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_NumpyTakeCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_igamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_multilabel_soft_margin_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_svd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mvlgamma_mvlgamma_p_5_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_add_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_take_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_div_floor_rounding_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_boolean_index",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_tan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_general_hamming_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_MapControlflowOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_tanh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_margin_ranking_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_dist_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_unflatten_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_long_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_softmin_with_dtype_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_prod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_ndtr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_pow_cpu_float32",  # known py38 fail
    "TestGenericProxyTensorSymbolic.test_mode_tracing_factory_function",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_chunk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_index_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_NumpySplitCopyCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_pairwise_distance_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_fftshift_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sparse_sampled_addmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_relu6_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fft_fftshift_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_legendre_polynomial_p_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_ones_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_adaptive_avg_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_multi_head_attention_forward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sinh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_matmul_cpu_float32",  # known py38 fail
    "TestSymbolicTracing.test_repeat_interleave",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_i1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_normalize_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_addmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_squeeze_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_mish_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_NumpySplitCopyWithIntCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_addmv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_gradient_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_var_mean_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_modified_bessel_i1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_take_along_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_min_binary_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive___rdiv___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_mse_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_randn_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_diag_embed_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_histogramdd_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sqrt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_modified_bessel_k1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_elu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cat_cpu_float32",  # known py38 fail
    "TestRefsCPU.test_infinite_loop_from_py_dispatcher_cpu",  # known py38 fail
    "TestReductions.test_all",  # known py38 fail
    "TestReductions.test_mean_grad_case_1d",  # known py38 fail
    "TestBasicsCPU.test_contiguous_cpu",  # known py38 fail
    "TestReductions.test_mean",  # known py38 fail
    "TestBasicsCPU.test_softmax_cpu",  # known py38 fail
    "TestReductions.test_mean_dim_grad",  # known py38 fail
    "TestReductions.test_amin_grad",  # known py38 fail
    "TestBasicsCPU.test_invalid_sparse_csr_values_cpu",  # known py38 fail
    "TestReductions.test_sum",  # known py38 fail
    "TestReductions.test_mean_grad_case_1e",  # known py38 fail
    "TestReductions.test_mean_grad_case_1f",  # known py38 fail
    "TestBasicsCPU.test_where_cpu",  # known py38 fail
    "TestReductions.test_prod_grad",  # known py38 fail
    "TestBasicsCPU.test_invalid_sparse_coo_values_cpu",  # known py38 fail
    "TestReductions.test_amax_grad",  # known py38 fail
    "TestReductions.test_sum_grad",  # known py38 fail
    "TestReductions.test_mean_grad_case_1b",  # known py38 fail
    "TestReductions.test_prod",  # known py38 fail
    "TestReductions.test_amax",  # known py38 fail
    "TestReductions.test_amin",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_uint8_uint8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int8_int64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int8_int16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_pow_scalar_overloads_mem_overlap_cpu_float64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int64_float64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int32_int8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int64_int8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int32_float64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int64_uint8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float64_uint8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int32_int32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float16_int8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float64_float16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int32_int16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_sub_typing_cpu",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int8_float32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_uint8_float32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_bool_int8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int16_int32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float64_int8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_uint8_int32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_add_cpu",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float32_float32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int16_bool",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_uint8_bool",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float16_bool",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int64_int32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_bool_float64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_bool_float16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float16_int64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float64_float64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float32_int8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float64_bool",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int8_float64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_uint8_int16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int16_uint8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_bool_float32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int8_int32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int32_int64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_uint8_int64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float32_bool",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_gradients_cpu_float64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int16_int64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float16_float64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int8_float16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int32_uint8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_bool_int32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_uint8_float64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float64_int32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float16_float32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_bool_int64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float32_int16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float16_float16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float64_float32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_bool_bool",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int32_bool",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int16_int16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int64_int16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float16_int16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_int_tensor_pow_neg_ints_cpu",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_uint8_int8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_bool_uint8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float32_uint8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int8_bool",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_uint8_float16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int64_float32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float32_float16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_bool_int16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int8_uint8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float64_int64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float16_uint8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int64_float16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int8_int8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int64_bool",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float16_int32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int16_float16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int16_float64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float32_float64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float64_int16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int32_float32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_shift_limits_cpu_uint8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float32_int64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int32_float16",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int16_float32",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int64_int64",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_int16_int8",  # known py38 fail
    "TestBinaryUfuncsCPU.test_long_tensor_pow_floats_cpu",  # known py38 fail
    "TestBinaryUfuncsCPU.test_xlogy_xlog1py_cpu_float32_int32",  # known py38 fail
    "TestFXExperimental.test_optimize_for_inference_cpu",  # known py38 fail
    "TestForeachCPU.test_add_scalar_with_empty_list_and_empty_tensor_cpu_int32",  # known py38 fail
    "TestForeachCPU.test_add_scalar_with_empty_list_and_empty_tensor_cpu_int64",  # known py38 fail
    "TestForeachCPU.test_add_scalar_with_empty_list_and_empty_tensor_cpu_int16",  # known py38 fail
    "TestForeachCPU.test_add_scalar_with_empty_list_and_empty_tensor_cpu_int8",  # known py38 fail
    "TestForeachCPU.test_add_scalar_with_empty_list_and_empty_tensor_cpu_uint8",  # known py38 fail
}

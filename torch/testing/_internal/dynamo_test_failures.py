# mypy: ignore-errors

# NOTE: [dynamo_test_failures.py]
#
# We generate xFailIfTorchDynamo* for all tests in `dynamo_expected_failures`
# We generate skipIfTorchDynamo* for all tests in `dynamo_skips`
#
# For an easier-than-manual way of generating and updating these lists,
# see scripts/compile_tests/update_failures.py
#
# If you're adding a new test, and it's failing PYTORCH_TEST_WITH_DYNAMO=1,
# either add the appropriate decorators to your test or list them in this file.
#
# *These are not exactly unittest.expectedFailure and unittest.skip. We'll
# always execute the test and then suppress the signal, if necessary.
# If your tests crashes, or is slow, please use @skipIfTorchDynamo instead.

# Tests that run without strict mode in PYTORCH_TEST_WITH_INDUCTOR=1.
# Please don't add anything to this list.
FIXME_inductor_non_strict = {
    "test_modules",
    "test_ops",
    "test_ops_gradients",
    "test_torch",
}

# We generate unittest.expectedFailure for all of the following tests
# when run under PYTORCH_TEST_WITH_DYNAMO=1.
# see NOTE [dynamo_test_failures.py] for more details
#
# This lists exists so we can more easily add large numbers of failing tests,
dynamo_expected_failures = {
    "TestCppExtensionJIT.test_cpp_frontend_module_has_up_to_date_attribute",
    "TestCppExtensionJIT.test_custom_compound_op_autograd",
    "TestCppExtensionJIT.test_cpp_frontend_module_has_up_to_date_attributes",
    "TestCppExtensionOpenRgistration.test_open_device_registration",
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
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_float32",
    "TestLinalgCPU.test_inverse_cpu_complex128",
    "TestLinalgCPU.test_norm_dtype_cpu_complex128",
    "TestLinalgCPU.test_householder_product_cpu_float64",
    "TestLinalgCPU.test_addr_integral_cpu_int64",
    "TestLinalgCPU.test_norm_vector_cpu_float32",
    "TestLinalgCPU.test_solve_cpu_complex128",
    "TestLinalgCPU.test_lobpcg_torchscript_cpu_float64",
    "TestLinalgCPU.test_solve_cpu_float32",
    "TestLinalgCPU.test_addr_integral_cpu_int16",
    "TestLinalgCPU.test_norm_vector_cpu_float64",
    "TestLinalgCPU.test_addmm_sizes_cpu_float64",
    "TestLinalgCPU.test_norm_dtype_cpu_float64",
    "TestLinalgCPU.test_addr_integral_cpu_int8",
    "TestLinalgCPU.test_pinv_cpu_float32",
    "TestLinalgCPU.test_addr_integral_cpu_uint8",
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_complex128",
    "TestLinalgCPU.test_addr_integral_cpu_int32",
    "TestLinalgCPU.test_solve_cpu_complex64",
    "TestLinalgCPU.test_solve_cpu_float64",
    "TestLinalgCPU.test_addmm_sizes_cpu_float32",
    "TestLinalgCPU.test_norm_bfloat16_and_half_cpu_float16",
    "TestLinalgCPU.test_householder_product_cpu_complex64",
    "TestLinalgCPU.test_inverse_cpu_float64",
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_complex64",
    "TestLinalgCPU.test_pinv_cpu_complex64",
    "TestLinalgCPU.test_geqrf_cpu_complex128",
    "TestLinalgCPU.test_geqrf_cpu_complex64",
    "TestLinalgCPU.test_slogdet_errors_and_warnings_cpu_float64",
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
    "TestArange.test_infinite",
    "TestArrayConstruction.test_array_empty",
    "TestAttributes.test_fill_readonly",
    "TestArrayAttributeDeletion.test_multiarray_writable_attributes_deletion",
    "TestMinMax.test_scalar",
    "TestFromBuffer.test_basic_little_dtype2",
    "TestArrayCreationCopyArgument.test_striding_not_ok",
    "TestArange.test_require_range",
    "TestArange.test_nan_step",
    "TestArrayAttributeDeletion.test_multiarray_not_writable_attributes_deletion",
    "TestLexsort.test_datetime",
    "TestMinMax.test_axis",
    "TestLexsort.test_mixed",
    "TestAttributes.test_fill_struct_array",
    "TestFromBuffer.test_empty",
    "TestAssignment.test_assignment_broadcasting",
    "TestAttributes.test_set_stridesattr",
    "TestStats.test_out",
    "TestScalarIndexing.test_invalid_subscript",
    "TestWhere.test_error",
    "TestBool.test_sum_2",
    "TestScalarIndexing.test_invalid_newaxis",
    "TestScalarIndexing.test_invalid_subscript_assignment",
    "TestFromBuffer.test_basic_little_dtype1",
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
    "TestArgmax.test_combinations_data66",
    "TestArgmax.test_combinations_data65",
    "TestArgmax.test_combinations_data63",
    "TestArgmax.test_combinations_data62",
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
    "TestSortComplex.test_sort_real_type_in_g_type_out_G",
    "TestMeshgrid.test_invalid_arguments",
    "TestGradient.test_specific_axes",
    "TestDelete.test_slices",
    "TestPercentile.test_extended_axis_invalid",
    "TestGradient.test_second_order_accurate",
    "TestDiff.test_prepend",
    "TestQuantile.test_quantile_monotonic_method_averaged_inverted_cdf",
    "TestBincount.test_with_incorrect_minlength",
    "TestSortComplex.test_sort_real_type_in_H_type_out_F",
    "TestDiff.test_n",
    "TestMeshgrid.test_indexing",
    "TestQuantile.test_quantile_monotonic_method_closest_observation",
    "TestFlip.test_axes",
    "TestCov.test_fweights",
    "TestDiff.test_append",
    "TestPercentile.test_scalar_q",
    "TestMedian.test_extended_axis_invalid",
    "TestQuantile.test_quantile_monotonic_method_hazen",
    "TestQuantile.test_quantile_monotonic_method_normal_unbiased",
    "TestSetOps.test_in1d_table_timedelta_fails",
    "TestUnique.test_unique_axis_errors",
    "TestSetOps.test_setdiff1d",
    "TestSetOps.test_in1d_timedelta_kind_sort",
    "TestSetOps.test_in1d_timedelta_kind0",
    "TestUnique.test_unique_axis",
    "TestConstant.test_check_constant_float3",
    "TestConstant.test_check_constant_pad_2d",
    "TestConcatenate.test_exceptions",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_large_concatenate_axis_None",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_concatenate",  # torch_np/numpy_tests/core/test_shape_base
    "TestVstack.test_empty_input",  # torch_np/numpy_tests/core/test_shape_base
    "TestVstack.test_non_iterable",  # torch_np/numpy_tests/core/test_shape_base
    "TestStackMisc.test_stack",  # torch_np/numpy_tests/core/test_shape_base
    "TestConcatenate.test_bad_out_shape",  # torch_np/numpy_tests/core/test_shape_base
    "TestHstack.test_non_iterable",  # torch_np/numpy_tests/core/test_shape_base
    "TestHstack.test_empty_input",  # torch_np/numpy_tests/core/test_shape_base
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
    "TestRequire.test_require_each",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_int32_inout_casting_unsafe",  # torch_np/numpy_tests/core/test_numeric
    "TestMoveaxis.test_errors",  # torch_np/numpy_tests/core/test_numeric
    "TestNonzeroAndCountNonzero.test_count_nonzero_axis",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_clip_func_takes_out",  # torch_np/numpy_tests/core/test_numeric
    "TestRollaxis.test_exceptions",  # torch_np/numpy_tests/core/test_numeric
    "TestRequire.test_C_and_F_simul",  # torch_np/numpy_tests/core/test_numeric
    "TestNonarrayArgs.test_dunder_round_edgecases_val_2147483647_ndigits_-1",  # torch_np/numpy_tests/core/test_numeric
    "TestClip.test_simple_complex",  # torch_np/numpy_tests/core/test_numeric
    "TestBroadcast.test_broadcast_single_arg",  # torch_np/numpy_tests/core/test_numeric
    "TestRequire.test_unknown_requirement",  # torch_np/numpy_tests/core/test_numeric
    "TestBroadcast.test_broadcast_error_kwargs",  # torch_np/numpy_tests/core/test_numeric
    "TestNonarrayArgs.test_dunder_round_edgecases_val_2147483647_ndigits_-9",  # torch_np/numpy_tests/core/test_numeric
    "TestNonarrayArgs.test_dunder_round_edgecases_val_2147483647_ndigits_-10",  # torch_np/numpy_tests/core/test_numeric
    "TestCross.test_broadcasting_shapes",  # torch_np/numpy_tests/core/test_numeric
    "TestIndexing.test_index_no_floats",  # torch_np/numpy_tests/core/test_indexing
    "TestBooleanIndexing.test_boolean_indexing_weirdness",  # torch_np/numpy_tests/core/test_indexing
    "TestBooleanIndexing.test_bool_as_int_argument_errors",  # torch_np/numpy_tests/core/test_indexing
    "TestFloatNonIntegerArgument.test_non_integer_argument_errors",  # torch_np/numpy_tests/core/test_indexing
    "TestIndexing.test_slicing_no_floats",  # torch_np/numpy_tests/core/test_indexing
    "TestFloatNonIntegerArgument.test_reduce_axis_float_index",  # torch_np/numpy_tests/core/test_indexing
    "TestEinsum.test_different_paths_dtype_e",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_B",  # torch_np/numpy_tests/core/test_einsum
    "TestEinsum.test_different_paths_dtype_b",  # torch_np/numpy_tests/core/test_einsum
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
    "TestViewOpsLAZY.test_advanced_indexing_assignment_lazy",  # test_view_ops
    "TestOldViewOpsCPU.test_crow_col_indices_cpu",  # test_view_ops
    "TestViewOpsLAZY.test_advanced_indexing_nonview_lazy",  # test_view_ops
    "TestTypePromotionCPU.test_alpha_mismatch_cpu",  # test_type_promotion
    "TestTypePromotionCPU.test_alternate_result_cpu",  # test_type_promotion
    "TestTypeHints.test_doc_examples",  # test_type_hints
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_float32_cpu_float32",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_float64_cpu_float64",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_float16_cpu_float16",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_bfloat16_cpu_bfloat16",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_float32_cpu_float32",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_float64_cpu_float64",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_0_float16_cpu_float16",
    "TestSDPACPU.test_fused_sdp_choice_cpu_type_dense_dropout_0_7_bfloat16_cpu_bfloat16",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape0_cpu",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape1_cpu",
    "TestAttnMasksCPU.test_is_causal_and_mask_fails_cpu",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape2_cpu",
    "TestAttnMasksCPU.test_is_causal_equals_upper_left_shape3_cpu",
    "TestAttnMasksCUDA.test_causal_variants_causal_variant_1_shape0_cuda",
    "TestAttnMasksCUDA.test_causal_variants_causal_variant_1_shape1_cuda",
    "TestAttnMasksCUDA.test_causal_variants_causal_variant_1_shape2_cuda",
    "TestAttnMasksCUDA.test_causal_variants_causal_variant_1_shape3_cuda",
    "TestAttnMasksCUDA.test_causal_variants_causal_variant_2_shape0_cuda",
    "TestAttnMasksCUDA.test_causal_variants_causal_variant_2_shape1_cuda",
    "TestAttnMasksCUDA.test_causal_variants_causal_variant_2_shape2_cuda",
    "TestAttnMasksCUDA.test_causal_variants_causal_variant_2_shape3_cuda",
    "TestAttnMasksCUDA.test_is_causal_and_mask_fails_cuda",
    "TestAttnMasksCUDA.test_is_causal_equals_upper_left_shape0_cuda",
    "TestAttnMasksCUDA.test_is_causal_equals_upper_left_shape1_cuda",
    "TestAttnMasksCUDA.test_is_causal_equals_upper_left_shape2_cuda",
    "TestAttnMasksCUDA.test_is_causal_equals_upper_left_shape3_cuda",
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
    "TestTensorBoardUtils.test_numpy_vid_uint8",  # test_tensorboard
    "TestTensorProtoSummary.test_complex_tensor_proto",  # test_tensorboard
    "TestTensorBoardSummary.test_image_with_one_channel",  # test_tensorboard
    "TestTensorBoardEmbedding.test_embedding_64",  # test_tensorboard
    "TestTensorBoardSummary.test_video",  # test_tensorboard
    "TestTensorProtoSummary.test_int_tensor_proto",  # test_tensorboard
    "TestTensorProtoSummary.test_empty_tensor_proto",  # test_tensorboard
    "TestTensorBoardSummary.test_uint8_image",  # test_tensorboard
    "TestAsArrayCPU.test_copy_list_cpu_float64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_int64",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_int32",  # test_tensor_creation_ops
    "TestAsArrayCPU.test_copy_list_cpu_float32",  # test_tensor_creation_ops
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
    "TestSubclass.test_param_invariants_non_wrapper_tensor_tensor_requires_grad_False",  # test_subclass
    "TestSubclass.test_param_invariants_sparse_tensor_tensor_requires_grad_True",  # test_subclass
    "TestSubclass.test_param_invariants_diag_tensor_below_tensor_requires_grad_True",  # test_subclass
    "TestSubclass.test_param_invariants_diag_tensor_below_tensor_requires_grad_False",  # test_subclass
    "TestSubclass.test_param_invariants_non_wrapper_tensor_tensor_requires_grad_True",  # test_subclass
    "TestSubclass.test_parametrization_non_wrapper_tensor_leave_parametrized_False",  # test_subclass
    "TestSubclass.test_type_propagation_non_wrapper_tensor_as_param_False",  # test_subclass
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
    "TestSubclassSerialization.test_tensor_subclass_getstate_overwrite",  # test_serialization
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
    "TestNN.test_Sequential_imul",  # test_nn
    "TestNN.test_upsampling_bfloat16",  # test_nn
    "TestNNDeviceTypeCPU.test_triplet_margin_with_distance_loss_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_CTCLoss_no_batch_dim_reduction_sum_use_module_form_False_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_nll_loss_empty_tensor_reduction_sum_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_upsamplingTrilinear3d_align_corners_False_memory_format0_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_upsamplingTrilinear3d_align_corners_False_memory_format1_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_batchnorm_grad_cpu",  # test_nn
    "TestNN.test_interpolate",  # test_nn
    "TestNNDeviceTypeCPU.test_upsamplingTrilinear3d_align_corners_True_memory_format0_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_upsamplingTrilinear3d_align_corners_True_memory_format1_cpu",  # test_nn
    "TestNN.test_fb_fc_packed",  # test_nn
    "TestFusionEval.test_fuse_module_eval_numerics",  # test_nn
    "TestNNDeviceTypeCPU.test_invalid_reduction_strings_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_nll_loss_total_weight_is_zero_cpu",  # test_nn
    "TestNNDeviceTypeCPU.test_nll_loss_empty_tensor_reduction_mean_cpu",  # test_nn
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
    "FakeTensorOperatorInvariants.test_like_ops",  # test_fake_tensor
    "FakeTensorConverterTest.test_memoized_conversion_from_meta",  # test_fake_tensor
    "FakeTensorOperatorInvariants.test_non_kwarg_only_device",  # test_fake_tensor
    "FakeTensorOperatorInvariants.test_tensor_constructors_all_have_kwarg_device",  # test_fake_tensor
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_conv1d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_instance_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_conv3d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_instance_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_conv1d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_group_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_group_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_conv3d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_conv2d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_conv2d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_layer_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_group_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_layer_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_conv3d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_sum_nn_functional_instance_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_conv1d_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weight_per_sample_grad_mean_nn_functional_layer_norm_cpu_float64",  # test_expanded_weights  # noqa: B950
    "TestExpandedWeightFunctionalCPU.test_expanded_weights_per_sample_grad_input_no_grad_nn_functional_conv2d_cpu_float64",  # test_expanded_weights  # noqa: B950
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
    "TestModuleHooks.test_forward_hooks_named_tuple_False",  # nn/test_module_hooks
    "TestModuleHooks.test_full_backward_hooks_named_tuple_False",  # nn/test_module_hooks
    "TestModuleHooks.test_forward_hooks_named_tuple_True",  # nn/test_module_hooks
    "TestModuleHookNN.test_hook_inplace",  # nn/test_module_hooks
    "TestModuleHooks.test_full_backward_hooks_named_tuple_True",  # nn/test_module_hooks
    "TestModuleHooks.test_forward_pre_hooks_named_tuple_False",  # nn/test_module_hooks
    "TestModuleHooks.test_full_backward_pre_hooks_named_tuple_False",  # nn/test_module_hooks
    "TestLazyModules.test_lazy_batchnorm2d_state",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv3d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv_transposed1d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv2d",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_instancenorm3d_state",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_batchnorm3d_state",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_conv_transpose1d_pickle",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_instancenorm2d",  # nn/test_lazy_modules
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
    "TestLazyModules.test_lazy_conv1d",  # nn/test_lazy_modules
    "TestLazyModules.test_linear",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_module_buffer",  # nn/test_lazy_modules
    "TestLazyModules.test_lazy_batchnorm1d_state",  # nn/test_lazy_modules
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
    "TestJacCPU.test_against_reference_correctness_different_devices_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_against_reference_default_arg_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_against_reference_multi_input_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_against_reference_multi_input_multi_output_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_against_reference_simple_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_against_reference_unrelated_outputs_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_against_reference_zero_dim_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_argnums_defaults_to_zero_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_aux_pytree_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_chunk_jacrev_composition__preallocate_and_copy_False_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_chunk_jacrev_composition__preallocate_and_copy_True_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_dimensionality_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_empty_output_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_hessian_simple_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_inplace_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_jac_with_non_tensor_args_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_multiple_inputs_outputs_pytree_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_multiple_inputs_pytree_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_multiple_outputs_multiple_argnums_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_multiple_outputs_single_argnums_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_outputs_can_any_pytree_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_unrelated_input_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestJacCPU.test_unrelated_output_jacfwd_cpu",  # functorch/test_eager_transforms.py
    "TestVmapJvpInplaceViewCPU.test_all_dual_base_inplace_cpu",  # functorch/test_eager_transforms.py
    "TestVmapJvpInplaceViewCPU.test_all_dual_base_view_inplace_cpu",  # functorch/test_eager_transforms.py
    "TestVmapJvpInplaceViewCPU.test_all_dual_no_view_cpu",  # functorch/test_eager_transforms.py
    "TestVmapJvpInplaceViewCPU.test_right_dual_base_prop_cpu",  # functorch/test_eager_transforms.py
    "TestVmapJvpInplaceViewCPU.test_right_dual_view_prop_cpu",  # functorch/test_eager_transforms.py
    "TestHessianCPU.test_hessian_vectorize_correctness_multi_input_cpu",  # functorch/test_eager_transforms.py
    "TestHessianCPU.test_hessian_vectorize_correctness_simple_cpu",  # functorch/test_eager_transforms.py
    "TestHessianCPU.test_hessian_vectorize_correctness_unrelated_outputs_cpu",  # functorch/test_eager_transforms.py
    "TestHessianCPU.test_jacfwd_different_levels_cpu",  # functorch/test_eager_transforms.py
    "TestExamplesCorrectnessCPU.test_ensemble_regression_mechanism_functional_call_cpu",  # functorch/test_eager_transforms.py
    "TestExamplesCorrectnessCPU.test_ensemble_regression_mechanism_make_functional_cpu",  # functorch/test_eager_transforms.py
    "TestExamplesCorrectnessCPU.test_find_learning_rate_ensembling_AlphaDropout_mechanism_functional_call_cpu",  # functorch/test_eager_transforms.py
    "TestExamplesCorrectnessCPU.test_find_learning_rate_ensembling_AlphaDropout_mechanism_make_functional_cpu",  # functorch/test_eager_transforms.py
    "TestExamplesCorrectnessCPU.test_find_learning_rate_ensembling_Dropout_mechanism_functional_call_cpu",  # functorch/test_eager_transforms.py
    "TestExamplesCorrectnessCPU.test_find_learning_rate_ensembling_Dropout_mechanism_make_functional_cpu",  # functorch/test_eager_transforms.py
    "TestExamplesCorrectnessCPU.test_find_learning_rate_ensembling_FeatureAlphaDropout_mechanism_functional_call_cpu",  # functorch/test_eager_transforms.py
    "TestExamplesCorrectnessCPU.test_find_learning_rate_ensembling_FeatureAlphaDropout_mechanism_make_functional_cpu",  # functorch/test_eager_transforms.py
    "TestHigherOrderOperatorInteractionCPU.test_vmap_grad_sum_cpu",  # functorch/test_eager_transforms.py
    "TestFunctionalizeCPU.test_multioutput_view_cpu",  # functorch/test_eager_transforms.py
    "TestFunctionalizeCPU.test_simple_view_cpu",  # functorch/test_eager_transforms.py
    "TestFunctionalizeCPU.test_vmap_functionalize_jvp_cpu",  # functorch/test_eager_transforms.py
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
    "FuncTorchHigherOrderOpTests.test_vmap_with_conditional_graph_break",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_with_graph_break",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_with_graph_break_2",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_with_graph_break_lambda",  # dynamo/test_higher_order_ops
    "FuncTorchHigherOrderOpTests.test_vmap_previous_illegal_op_no_graph_break",  # dynamo/test_higher_order_ops
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
    "TestTorchFunctionOverride.test_tensor_subclass_propagation",  # test_overrides
    "TestNamedTuple.test_max",  # test_overrides
    "TestTorchFunctionMode.test_mode_notimplemented_loop",  # test_overrides
    "TestTorchFunctionMode.test_disable_enable_subclass",  # test_overrides
    "TestTorchFunctionOverride.test_mean_semantics",  # test_overrides
    "TestGradCheckOverride.test_gradcheck",  # test_overrides
    "TestTorchFunctionOverride.test_Tensor___cuda_array_interface_____get__",  # test_overrides
    "TestTorchFunctionMode.test_modes_return_notimplemented",  # test_overrides
    "TestIterator.test_iterator",  # test_overrides
    "TestTorchFunctionMode.test_nested_modes_with_python_has_torch_function",  # test_overrides
    "TestPickle.test_pickle",  # test_overrides
    "TestGradNewOnesOverride.test_newones",  # test_overrides
    "TestTorchFunctionOverride.test_precedence_semantics",  # test_overrides
    "TestNamedTupleAPI.test_namedtuple_return",  # test_namedtuple_return_api
    "TestVmapAPILegacy.test_accepts_nested_inputs",  # test_legacy_vmap
    "TestVmapAPILegacy.test_nested_out_dims",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_inplace_manyview_cpu",  # test_legacy_vmap
    "TestVmapAPILegacy.test_functools_partial",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_unrelated_output_cpu",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_index_cpu",  # test_legacy_vmap
    "TestVmapAPILegacy.test_non_zero_in_dims",  # test_legacy_vmap
    "TestVmapAPILegacy.test_single_input",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_chunk",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_mul_cpu",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_unrelated_output_multiple_grad_cpu",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_select",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_binary_pointwise_ops",  # test_legacy_vmap
    "TestVmapAPILegacy.test_non_tensor_output_raises",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_binary_cross_entropy_cpu",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_diagonal",  # test_legacy_vmap
    "TestVmapAPILegacy.test_nonzero_out_dims",  # test_legacy_vmap
    "TestVmapAPILegacy.test_unsupported_op_err_msg",  # test_legacy_vmap
    "TestVmapAPILegacy.test_batched_gradient_basic",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_slice",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_sigmoid_cpu",  # test_legacy_vmap
    "TestVmapAPILegacy.test_out_dims_and_num_outputs_mismatch_err_msg",  # test_legacy_vmap
    "TestVmapAPILegacy.test_noop_in_inner_vmap",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_new_empty_strided",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_is_floating_point",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_split",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_fill_and_zero_inplace",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_is_complex",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_as_strided",  # test_legacy_vmap
    "TestVmapAPILegacy.test_nested_with_different_map_dim",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_new_zeros",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_logsumexp_cpu",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_log1p_cpu",  # test_legacy_vmap
    "TestVmapAPILegacy.test_grad_unsupported_interaction",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_transpose",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_clamp",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_stride",  # test_legacy_vmap
    "TestVmapAPILegacy.test_multiple_outputs_error_cases",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_log_cpu",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_reshape",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_inplace_on_view_cpu",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_new_empty",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_lgamma_cpu",  # test_legacy_vmap
    "TestVmapAPILegacy.test_multiple_out_dims",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_result_type",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_sum_dim",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_to",  # test_legacy_vmap
    "TestVmapAPILegacy.test_backward_unsupported_interaction",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_comparison_ops",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_is_contiguous",  # test_legacy_vmap
    "TestVmapAPILegacy.test_multiple_outputs",  # test_legacy_vmap
    "TestVmapAPILegacy.test_out_dim_out_of_bounds_err_msg",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_view",  # test_legacy_vmap
    "TestVmapBatchedGradientLegacyCPU.test_div_cpu",  # test_legacy_vmap
    "TestVmapAPILegacy.test_out_dims_edge_case",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_clone",  # test_legacy_vmap
    "TestVmapAPILegacy.test_in_dim_not_in_tensor_err_msg",  # test_legacy_vmap
    "TestVmapAPILegacy.test_fallback_with_undefined_grad",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_no_random_op_support",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_unbind",  # test_legacy_vmap
    "TestVmapAPILegacy.test_non_default_in_dims_out_dims",  # test_legacy_vmap
    "TestVmapOperatorsLegacy.test_T_numpy",  # test_legacy_vmap
    "TestNamedTensor.test_expand",  # test_namedtensor
    "TestNamedTensor.test_masked_fill",  # test_namedtensor
    "TestNamedTensor.test_addmv",  # test_namedtensor
    "TestNamedTensor.test_cummax_cummin",  # test_namedtensor
    "TestNamedTensor.test_rename_rename_map",  # test_namedtensor
    "TestNamedTensor.test_mm",  # test_namedtensor
    "TestNamedTensor.test_no_save_support",  # test_namedtensor
    "TestNamedTensor.test_dot",  # test_namedtensor
    "TestNamedTensor.test_using_unseen_uninterned_string_refcounts",  # test_namedtensor
    "TestNamedTensor.test_has_names",  # test_namedtensor
    "TestNamedTensor.test_unflatten",  # test_namedtensor
    "TestNamedTensor.test_rename_",  # test_namedtensor
    "TestNamedTensor.test_binary_ops",  # test_namedtensor
    "TestNamedTensor.test_set_names_property",  # test_namedtensor
    "TestNamedTensor.test_info_smoke",  # test_namedtensor
    "TestNamedTensor.test_logcumsumexp",  # test_namedtensor
    "TestNamedTensor.test_tensor_grad_is_unnamed",  # test_namedtensor
    "TestNamedTensor.test_logical_not",  # test_namedtensor
    "TestNamedTensor.test_as_strided",  # test_namedtensor
    "TestNamedTensor.test_rename_globber",  # test_namedtensor
    "TestNamedTensor.test_bmm",  # test_namedtensor
    "TestNamedTensor.test_flatten",  # test_namedtensor
    "TestNamedTensor.test_reduction_fns",  # test_namedtensor
    "TestNamedTensor.test_unary_propagate_names_fns",  # test_namedtensor
    "TestNamedTensor.test_detach",  # test_namedtensor
    "TestNamedTensor.test_size",  # test_namedtensor
    "TestNamedTensor.test_addcmul_addcdiv",  # test_namedtensor
    "TestNamedTensor.test_big_tensor_repr_has_names",  # test_namedtensor
    "TestNamedTensor.test_unsupported_op_error_msg",  # test_namedtensor
    "TestNamedTensor.test_addmm",  # test_namedtensor
    "TestNamedTensor.test_pow_special",  # test_namedtensor
    "TestNamedTensor.test_autograd_ignores_names",  # test_namedtensor
    "TestNamedTensor.test_index_fill",  # test_namedtensor
    "TestNamedTensor.test_masked_select",  # test_namedtensor
    "TestNamedTensor.test_comparison_ops",  # test_namedtensor
    "TestNamedTensor.test_diagonal",  # test_namedtensor
    "TestNamedTensor.test_bitwise_not",  # test_namedtensor
    "TestNamedTensor.test_equal",  # test_namedtensor
    "TestNamedTensor.test_rename",  # test_namedtensor
    "TestNamedTensor.test_select",  # test_namedtensor
    "TestNamedTensor.test_no_pickle_support",  # test_namedtensor
    "TestNamedTensor.test_factory_coverage",  # test_namedtensor
    "TestNamedTensor.test_split_fns_propagates_names",  # test_namedtensor
    "TestNamedTensor.test_matmul",  # test_namedtensor
    "TestNamedTensor.test_autograd_smoke",  # test_namedtensor
    "TestNamedTensor.test_tensor_from_named_tensor",  # test_namedtensor
    "TestNamedTensor.test_copy_transpose",  # test_namedtensor
    "TestNamedTensor.test_using_seen_interned_string_doesnt_bump_refcount",  # test_namedtensor
    "TestNamedTensor.test_factory_edge_cases",  # test_namedtensor
    "TestNamedTensor.test_max_pooling",  # test_namedtensor
    "TestNamedTensor.test_autograd_warns_named_grad",  # test_namedtensor
    "TestNamedTensor.test_cdist",  # test_namedtensor
    "TestNamedTensor.test_transpose_variants",  # test_namedtensor
    "TestNamedTensor.test_bernoulli",  # test_namedtensor
    "TestNamedTensor.test_no_multiprocessing_support",  # test_namedtensor
    "TestNamedTensor.test_any_all",  # test_namedtensor
    "TestNamedTensor.test_out_fn_semantics",  # test_namedtensor
    "TestNamedTensor.test_cat",  # test_namedtensor
    "TestNamedTensor.test_noncontig_contiguous",  # test_namedtensor
    "TestNamedTensor.test_stride",  # test_namedtensor
    "TestNamedTensor.test_logical_ops",  # test_namedtensor
    "TestNamedTensor.test_mv",  # test_namedtensor
    "TestNamedTensor.test_using_unseen_interned_string_bumps_refcount_permanently",  # test_namedtensor
    "TestNamedTensor.test_resize",  # test_namedtensor
    "TestFX.test_pytree_concrete",  # test_fx
    "TestCommonPass.test_correctness_CSEPass_Mutation_cpu",  # test_fx
    "TestFX.test_custom_traceback_raised_when_exception_source_is_graphmodule",  # test_fx
    "TestConstFold.test_check_skip_folding_quant_dequant_pattern",  # test_fx
    "TestFX.test_immutable_list_pytree_ops",  # test_fx
    "TestCommonPass.test_correctness_CSEPass_TakeList_cpu",  # test_fx
    "TestPassManager.test_pass_manager",  # test_fx
    "TestCommonPass.test_correctness_CSEPass_MutationMetadata_cpu",  # test_fx
    "TestCommonPass.test_correctness_CSEPass_MutationTorchTensorCall_cpu",  # test_fx
    "TestCommonPass.test_correctness_CSEPass_MutationInput_cpu",  # test_fx
    "TestFX.test_immutable_dict_pytree_ops",  # test_fx
    "TestCommonPass.test_correctness_factory_CSEPass_MutationFactory_cpu",  # test_fx
    "TestCommonPass.test_correctness_factory_CSEPass_FactoryFunctionCall_cpu",  # test_fx
    "TestCommonPass.test_correctness_CSEPass_ReturnList_cpu",  # test_fx
    "TestFXAPIBackwardCompatibility.test_public_api_surface",  # test_fx
    "TestContentStoreCPU.test_repeated_hash_cpu",  # test_content_store
    "TestLazyTensor.test_tensor_ctr",  # lazy/test_ts_opinfo
    "TestAnalyze.test_trace_dependencies",  # test_package
    "TestProfilerTree.test_profiler_experimental_tree_with_memory",  # profiler/test_profiler_tree
    "TestProfilerTree.test_profiler_experimental_tree_with_memory_and_stack",  # profiler/test_profiler_tree
    "TestProfilerTree.test_profiler_experimental_tree_with_record_function",  # profiler/test_profiler_tree
    "TestProfilerTree.test_profiler_experimental_tree_with_stack_and_torch_dispatch",  # profiler/test_profiler_tree
    "TestProfilerTree.test_profiler_experimental_tree_with_stack_and_torch_function",  # profiler/test_profiler_tree
    "TestTorchTidyProfiler.test_allocation_ids_with_other_ops",  # profiler/test_profiler
    "TestExperimentalUtils.test_profiler_synchronized_dataloader_pattern",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_impl_reuse",  # profiler/test_profiler
    "TestExperimentalUtils.test_profiler_pattern_matcher_json_report",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_tensorimpl_invalidation_full",  # profiler/test_profiler
    "TestProfiler.test_profiler_tracing",  # profiler/test_profiler
    "TestProfiler.test_is_profiler_enabled",  # profiler/test_profiler
    "TestExperimentalUtils.test_utils_compute_idle_time",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_optimizer_parameters_sgd",  # profiler/test_profiler
    "TestExperimentalUtils.test_profiler_name_pattern",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_extra_fields",  # profiler/test_profiler
    "TestProfiler.test_flops",  # profiler/test_profiler
    "TestProfiler.test_profiler_correlation_id",  # profiler/test_profiler
    "TestProfiler.test_source_multithreaded_open_in_scope_work_in_main_thread_True",  # profiler/test_profiler
    "TestProfiler.test_source_multithreaded_close_in_scope_work_in_main_thread_True",  # profiler/test_profiler
    "TestProfiler.test_source",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_allocation_ids",  # profiler/test_profiler
    "TestRecordFunction.test_record_function",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_optimizer_parameters_adam",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_tensor_properties",  # profiler/test_profiler
    "TestProfiler.test_profiler_fwd_bwd_link",  # profiler/test_profiler
    "TestProfiler.test_concrete_inputs_profiling",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_tensorimpl_invalidation_scalar_args",  # profiler/test_profiler
    "TestProfiler.test_guarded_record_function_fast",  # profiler/test_profiler
    "TestExperimentalUtils.test_profiler_optimizer_single_tensor_pattern",  # profiler/test_profiler
    "TestExperimentalUtils.test_utils_compute_self_time",  # profiler/test_profiler
    "TestProfiler.test_high_level_trace",  # profiler/test_profiler
    "TestRecordFunction.test_datapipe_with_record_function_fork",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_allocations",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_module_and_optimizer_ids",  # profiler/test_profiler
    "TestExperimentalUtils.test_utils_compute_queue_depth_when_no_cuda_events",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_allocation_id_uniqueness",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_sparse_tensors",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_optimizer",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_tensorimpl_invalidation_keep_alive",  # profiler/test_profiler
    "TestExperimentalUtils.test_utils_compute_queue_depth",  # profiler/test_profiler
    "TestExperimentalUtils.test_profiler_pattern_match_helper",  # profiler/test_profiler
    "TestProfiler.test_export_stacks",  # profiler/test_profiler
    "TestProfiler.test_source_multithreaded_basic_work_in_main_thread_True",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_mkldnn_tensors",  # profiler/test_profiler
    "TestRecordFunction.test_datapipe_with_record_function",  # profiler/test_profiler
    "TestProfiler.test_memory_profiler",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_tensor_lists",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_pointers_and_ids",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_nnmodule_params",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_tensorimpl_invalidation_set",  # profiler/test_profiler
    "TestTorchTidyProfiler.test_scalar_ins",  # profiler/test_profiler
    "TestProfiler.test_profiler_op_event_args",  # profiler/test_profiler
    "TestProfiler.test_source_multithreaded_complex_work_in_main_thread_True",  # profiler/test_profiler
    "TestProfiler.test_source_multithreaded_multiple_preexisting_work_in_main_thread_True",  # profiler/test_profiler
    "TestAOTAutograd.test_input_mutation_aliases_and_output_alias",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_aliases_bases_out_of_order",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_inference_mode",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_is_output",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_set__input_mutation",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_aliases_intermediate_multiple_mixed",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_some_outputs_dont_require_grad_view",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_data_and_metadata_mutation_aliases_other_input",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_output_view_metadata_mutate_multiple",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_data_and_metadata_mutation",  # functorch/test_aotdispatch
    "TestPythonKeyCPU.test_make_fx_vjp_cpu",  # functorch/test_aotdispatch
    "TestPartitioning.test_min_cut_partitioner_output_tensor_shape_tensor",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_subclass_metadata_mutation_req_grad_False",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_output_view_simple",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_aliases_intermediate_multi_output_view",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_aliases_intermediate_returned_multiple_times",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_metadata_mutation_aliases",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_aliases_input_multi_output_view",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_subclass_metadata_mutation_req_grad_True",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_aliases_other_input2",  # functorch/test_aotdispatch
    "TestAOTDispatch.test_aot_dispatch_input_mutation_and_output_alias",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_aliased_with_mutation_output_alias",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_invalid_dupe_left_bias",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_aliases_multiple_inputs_get_correct_one",  # functorch/test_aotdispatch
    "TestAOTDispatch.test_aot_dispatch_input_mutation",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_new_inp_requires_grad_now",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_view_detach",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_false_aliasing",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_default_partitioner_saves_symints_not_tensors_for_bw",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_invalid_dupe",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_aliases_intermediate_and_returned_different_grad",  # functorch/test_aotdispatch
    "TestPartitioning.test_min_cut_partitioner_recomputable_ops",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_all_alias_types",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_set__and_data_mutation_good",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_set__and_data_mutation_bad",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_set__nop",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_dupe_arg_torture",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_noncontiguous",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_output_view_mutate_multiple",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_aliases_intermediate_and_returned",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_some_outputs_dont_require_grad_non_view",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_some_output_requires_grad_input_doesnt",  # functorch/test_aotdispatch
    "TestAOTDispatch.test_aot_dispatch_output_alias",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_and_output_view",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_invalid_dupe_fake",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_aliases_intermediate_multiple",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_alias_everything",  # functorch/test_aotdispatch
    "TestPythonKeyCPU.test_make_fx_jacrev_cpu",  # functorch/test_aotdispatch
    "TestPartitioning.test_default_partitioner_output_tensor_shape_tensor",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_aliases_other_input",  # functorch/test_aotdispatch
    "TestPartitioning.test_contiguous",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_outputs_are_aliased",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_aliases_and_none_require_gradients",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_aliases_intermediate_and_returned_flipped",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_view_and_inplace_view",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_metadata",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_dict",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_output_op_depending_on_symint",  # functorch/test_aotdispatch
    "TestAOTAutograd.test_input_mutation_output_view_multiple",  # functorch/test_aotdispatch
    "TestUnflatten.test_unflatten_container_type",  # export/test_unflatten
    "TestDeserialize.test_tensor_tensor_list",  # export/test_serialize
    "SerDesExportTestExport.test_constrain_size_with_constrain_value_serdes",  # export/test_serdes
    "SerDesExportTestDynamismExpression.test_export_inline_constraints_serdes",  # export/test_serdes
    "SerDesExportTestExport.test_nn_module_stack_serdes",  # export/test_serdes
    "SerDesExportTestExport.test_basic_non_strict_real_tensor_serdes",  # export/test_serdes
    "SerDesExportTestExport.test_external_call_non_strict_real_tensor_serdes",  # export/test_serdes
    "SerDesExportTestExport.test_constrain_size_with_various_cases_serdes",  # export/test_serdes
    "SerDesExportTestExport.test_constrain_size_in_eager_serdes",  # export/test_serdes
    "SerDesExportTestExport.test_non_strict_dynamic_shapes_serdes",  # export/test_serdes
    "SerDesExportTestExport.test_basic_non_strict_fake_tensor_serdes",  # export/test_serdes
    "SerDesExportTestExport.test_non_strict_dynamic_shapes_suggested_fixes_serdes",  # export/test_serdes
    "SerDesExportTestExport.test_cond_with_module_stack_export_with_serdes",  # export/test_serdes
    "SerDesExportTestExport.test_nn_module_stack_shared_submodule_serdes",  # export/test_serdes
    "RetraceExportTestExport.test_constrain_size_in_eager_retraceability",  # export/test_retraceability
    "RetraceExportTestExport.test_cond_with_module_stack_export_with_retraceability",  # export/test_retraceability
    "RetraceExportTestExport.test_non_strict_dynamic_shapes_suggested_fixes_retraceability",  # export/test_retraceability  # noqa: B950
    "RetraceExportTestExport.test_constrain_size_with_constrain_value_retraceability",  # export/test_retraceability
    "RetraceExportTestDynamismExpression.test_export_inline_constraints_retraceability",  # export/test_retraceability
    "RetraceExportTestExport.test_nn_module_stack_retraceability",  # export/test_retraceability
    "RetraceExportTestExport.test_nn_module_stack_shared_submodule_retraceability",  # export/test_retraceability
    "RetraceExportTestExport.test_constrain_size_with_various_cases_retraceability",  # export/test_retraceability
    "RetraceExportTestExport.test_non_strict_dynamic_shapes_retraceability",  # export/test_retraceability
    "TestPasses.test_views_op_having_view_copy",  # export/test_passes
    "TestPasses.test_functionalize_inline_contraints",  # export/test_passes
    "NonStrictExportTestExport.test_non_strict_dynamic_shapes_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_basic_non_strict_fake_tensor_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_nn_module_stack_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_param_util_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_cond_with_module_stack_export_with_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_non_strict_dynamic_shapes_suggested_fixes_non_strict",  # export/test_export_nonstrict  # noqa: B950
    "NonStrictExportTestExport.test_raise_user_error_when_guard_on_data_dependent_operation_non_strict",  # export/test_export_nonstrict  # noqa: B950
    "NonStrictExportTestExport.test__scaled_dot_product_flash_attention_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_to_module_with_mutated_buffer_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_basic_non_strict_real_tensor_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_external_call_non_strict_real_tensor_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_export_decomps_dynamic_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_to_module_with_mutated_buffer_multiple_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_export_with_wrong_inputs_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_fqn_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_sym_sqrt_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_nn_module_stack_shared_submodule_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_buffer_util_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_export_decomps_simple_non_strict",  # export/test_export_nonstrict
    "NonStrictExportTestExport.test_to_module_with_mutated_buffer_multiple_update_sub_later_non_strict",  # export/test_export_nonstrict  # noqa: B950
    "TestExport.test_non_strict_dynamic_shapes",  # export/test_export
    "TestExport.test_cond_with_module_stack_export_with",  # export/test_export
    "TestExport.test_constrain_size_in_eager",  # export/test_export
    "TestExport.test_nn_module_stack",  # export/test_export
    "TestExport.test_basic_non_strict_fake_tensor",  # export/test_export
    "TestExport.test_constrain_size_with_various_cases",  # export/test_export
    "TestExport.test_external_call_non_strict_real_tensor",  # export/test_export
    "TestDynamismExpression.test_export_inline_constraints",  # export/test_export
    "TestExport.test_basic_non_strict_real_tensor",  # export/test_export
    "TestExport.test_constrain_size_with_constrain_value",  # export/test_export
    "TestExport.test_nn_module_stack_shared_submodule",  # export/test_export
    "TestExport.test_non_strict_dynamic_shapes_suggested_fixes",  # export/test_export
    "TestExperiment.test_with_buffer_as_submodule",  # export/test_experimental
    "ExampleTests.test_exportdb_supported_case_constrain_as_size_example",  # export/test_db
    "ExampleTests.test_exportdb_supported_case_assume_constant_result",  # export/test_db
    "TestOpCPU.test_addmm_cpu_bfloat16",  # test_jit_llga_fuser
    "TestOpCPU.test_cat_cpu_float32",  # test_jit_llga_fuser
    "TestOpCPU.test_addmm_cpu_float32",  # test_jit_llga_fuser
    "TestOpCPU.test_add_scalar_cpu_bfloat16",  # test_jit_llga_fuser
    "TestOpCPU.test_add_scalar_cpu_float32",  # test_jit_llga_fuser
    "TestOpCPU.test_cat_cpu_bfloat16",  # test_jit_llga_fuser
    "TestOpCPU.test_identity_binary_cpu_bfloat16",  # test_jit_llga_fuser
    "TestOpCPU.test_identity_binary_cpu_float32",  # test_jit_llga_fuser
    "TestTorch.test_type",  # test_torch
    "TestTorch.test_cuda_not_built",  # test_torch
    "TestTorchDeviceTypeCPU.test_nondeterministic_resize_quantized_cpu_quint4x2",  # test_torch
    "TestTorch.test_map",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_fmod_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_uniform_kstest_cpu_float16",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_lerp_cpu",  # test_torch
    "TestTorch.test_parsing_int64",  # test_torch
    "TestTorchDeviceTypeCPU.test_exponential_kstest_cpu_bfloat16",  # test_torch
    "TestTorch.test_parsing_intlist",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_eq_cpu",  # test_torch
    "TestTorch.test_contains",  # test_torch
    "TestTorch.test_new",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_map2_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_ne_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_gt_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_uniform_kstest_cpu_bfloat16",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_div_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_nondeterministic_resize_quantized_cpu_quint8",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_lt_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_memory_format_operators_cpu",  # test_torch
    "TestTorch.test_pin_memory",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_masked_fill_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_nondeterministic_alert_MaxUnpool2d_cpu_float64",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_sub_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_le_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_nondeterministic_resize_quantized_cpu_qint32",  # test_torch
    "TestTorchDeviceTypeCPU.test_exponential_kstest_cpu_float16",  # test_torch
    "TestTorchDeviceTypeCPU.test_nondeterministic_resize_quantized_cpu_qint8",  # test_torch
    "TestTorchDeviceTypeCPU.test_untyped_storage_meta_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_remainder_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_nondeterministic_alert_MaxUnpool1d_cpu_float32",  # test_torch
    "TestTorchDeviceTypeCPU.test_uniform_kstest_cpu_float64",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_add_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_addcmul_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_nondeterministic_resize_quantized_cpu_quint2x4",  # test_torch
    "TestTorchDeviceTypeCPU.test_exponential_kstest_cpu_float64",  # test_torch
    "TestTorchDeviceTypeCPU.test_uniform_kstest_cpu_float32",  # test_torch
    "TestTorchDeviceTypeCPU.test_nondeterministic_alert_MaxUnpool2d_cpu_float32",  # test_torch
    "TestTorchDeviceTypeCPU.test_nondeterministic_alert_MaxUnpool3d_cpu_float32",  # test_torch
    "TestTorch.test_upsample_nearest2d_meta",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_map_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_nondeterministic_alert_MaxUnpool1d_cpu_float64",  # test_torch
    "TestTorch.test_upsample_nearest1d_meta",  # test_torch
    "TestTorchDeviceTypeCPU.test_normal_kstest_cpu_float32",  # test_torch
    "TestTorchDeviceTypeCPU.test_normal_kstest_cpu_float64",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_pow_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_copy_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_nondeterministic_alert_MaxUnpool3d_cpu_float64",  # test_torch
    "TestTorchDeviceTypeCPU.test_normal_kstest_cpu_float16",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_masked_scatter_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_ge_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_atan2_cpu",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_mul_cpu",  # test_torch
    "TestTorch.test_tensoriterator_output_setup",  # test_torch
    "TestTorchDeviceTypeCPU.test_broadcast_fn_addcdiv_cpu",  # test_torch
    "TestTorch.test_parsing_double",  # test_torch
    "TestTorchDeviceTypeCPU.test_exponential_kstest_cpu_float32",  # test_torch
    "TestTorchDeviceTypeCPU.test_deterministic_empty_cpu_uint64",  # test_torch
    "TestAutograd.test_checkpoint_detects_non_determinism",  # test_autograd
    "TestAutograd.test_gradcheck_backward_mul_by_grad_output",  # test_autograd
    "TestAutogradForwardModeBatchedGrad.test_out_of_place_basic",  # test_autograd
    "TestAutogradLogging.test_logging",  # test_autograd
    "TestAutograd.test_custom_function_cycle",  # test_autograd
    "TestAutogradForwardMode.test_detach_view_tracking",  # test_autograd
    "TestAutograd.test_return_duplicate_inplace",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_set_early_stop_no_recompution_needed",  # test_autograd
    "TestAutograd.test_backward_with_inputs",  # test_autograd
    "TestAutograd.test_setitem",  # test_autograd
    "TestAutogradDeviceTypeCPU.test_inplace_on_view_python_cpu",  # test_autograd
    "TestAutograd.test_custom_function_save_for_forward",  # test_autograd
    "TestAutograd.test_sparse_mm_backward",  # test_autograd
    "TestAutograd.test_checkpointing_without_reentrant_with_context_fn",  # test_autograd
    "TestAutograd.test_custom_function_saved_tensors",  # test_autograd
    "TestAutograd.test_custom_function_forward_mode_wrong_formula",  # test_autograd
    "TestAutogradInferenceMode.test_inference_mode_decorator",  # test_autograd
    "TestAutogradForwardMode.test_forward_level_cleanup",  # test_autograd
    "TestAutograd.test_gradcheck_check_forward_or_backward_only",  # test_autograd
    "TestAutogradDeviceTypeCPU.test_inplace_on_view_modify_base_cpu",  # test_autograd
    "TestAutograd.test_gradcheck_forward_ad_batched_grad",  # test_autograd
    "TestAutograd.test_custom_function_non_tensor_inputs_outputs",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_non_tensor_inputs_and_outputs_early_stop_True",  # test_autograd
    "TestAutograd.test_autograd_views_codegen",  # test_autograd
    "TestAutograd.test_profiler_aggregation_table",  # test_autograd
    "TestAutograd.test_profiler_propagation",  # test_autograd
    "TestAutograd.test_profiler_seq_nr",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_kwargs_early_stop_False",  # test_autograd
    "TestAutograd.test_checkpointing_non_reentrant_autocast_cpu",  # test_autograd
    "TestAutograd.test_named_tensor_for_complex_views",  # test_autograd
    "TestAutograd.test_set_grad_generator_functions_recursive",  # test_autograd
    "TestAutograd.test_increment_version",  # test_autograd
    "TestAutograd.test_record_function_callbacks",  # test_autograd
    "TestAutograd.test_save_on_cpu_and_checkpoint",  # test_autograd
    "TestAutogradForwardMode.test_make_dual_torch_dispatch",  # test_autograd
    "TestAutogradDeviceTypeCPU.test_sparse_ctor_getter_backward_cpu_complex128",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_non_tensor_inputs_and_outputs_early_stop_False",  # test_autograd
    "TestAutograd.test_gradcheck_nondeterministic",  # test_autograd
    "TestAutograd.test_custom_function_forward_mode_forward_is_no_op",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_set_early_stop",  # test_autograd
    "TestAutograd.test_version_counter",  # test_autograd
    "TestAutograd.test_set_grad_generator_functions",  # test_autograd
    "TestAutograd.test_record_function",  # test_autograd
    "TestAutograd.test_custom_function_forward_mode_view_checks",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_early_stop_True",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_two_children_early_stop_True",  # test_autograd
    "TestAutograd.test_gradcheck_check_no_differentiable_outputs",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_two_children_early_stop_False",  # test_autograd
    "TestAutograd.test_custom_autograd_repeated_grad_grad",  # test_autograd
    "TestAutograd.test_setitem_mask",  # test_autograd
    "TestAutogradDeviceTypeCPU.test_sparse_ctor_getter_backward_cpu_float64",  # test_autograd
    "TestAutograd.test_anomaly_mode_no_check_nan",  # test_autograd
    "TestAutograd.test_return_duplicate",  # test_autograd
    "TestAutogradForwardMode.test_create_new_zeros_with_same_meta",  # test_autograd
    "TestAutogradInferenceMode.test_inference_mode_tensor_creation",  # test_autograd
    "TestAutograd.test_set_grad_coroutines",  # test_autograd
    "TestAutograd.test_no_grad_copy_sparse",  # test_autograd
    "TestAutograd.test_set_grad_coroutines_exit",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_reentrant_backwards_early_stop_True",  # test_autograd
    "TestAutograd.test_saved_tensor_hooks_custom_function_intermediates",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_reentrant_backwards_early_stop_False",  # test_autograd
    "TestAutograd.test_custom_autograd_no_early_free",  # test_autograd
    "TestAutograd.test_checkpointing_without_reentrant_custom_function_works",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_kwargs_early_stop_True",  # test_autograd
    "TestAutograd.test_gradcheck_forward_ad",  # test_autograd
    "TestAutograd.test_access_saved_tensor_twice_without_recomputation_works",  # test_autograd
    "TestAutograd.test_hook_closure_cycle_use_custom_function_True_use_tensor_hook_False",  # test_autograd
    "TestAutograd.test_accumulate_grad_tensor_reference",  # test_autograd
    "TestAutogradInferenceMode.test_inference_mode_inf_tensor_in_inf_mode_inplace_op",  # test_autograd
    "TestAutogradDeviceTypeCPU.test_scatter_index_reduce_prod_gradgrad_error_cpu",  # test_autograd
    "TestAutograd.test_nested_anomaly_detect_nan",  # test_autograd
    "TestAutograd.test_nested_anomaly_printstack_cleanup",  # test_autograd
    "TestAutograd.test_post_accumulate_grad_hook_gets_cleaned_up",  # test_autograd
    "TestAutogradInferenceMode.test_inference_mode_context_manager",  # test_autograd
    "TestAutograd.test_hook_none",  # test_autograd
    "TestAutograd.test_set_data_tensorimpl_type",  # test_autograd
    "TestAutograd.test_autograd_simple_views_python",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_early_stop_False",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_same_graph_early_stop_False",  # test_autograd
    "TestAutograd.test_profiler_shapes",  # test_autograd
    "TestAutograd.test_profiler",  # test_autograd
    "TestNestedCheckpoint.test_nested_checkpoint_same_graph_early_stop_True",  # test_autograd
    "TestAutograd.test_custom_function_forward_mode_inplace_checks",  # test_autograd
    "TestAutograd.test_record_function_legacy",  # test_autograd
    "TestBaseStructuredSparsifier.test_constructor",  # test_ao_sparsity
    "TestBaseStructuredSparsifier.test_prepare_linear",  # test_ao_sparsity
    "TestFxComposability.test_q_prep_fx_s_prep_ref_conv",  # test_ao_sparsity
    "TestFxComposability.test_s_prep_before_q_prep_fx",  # test_ao_sparsity
    "TestBaseStructuredSparsifier.test_step_linear",  # test_ao_sparsity
    "TestComposability.test_convert_without_squash_mask",  # test_ao_sparsity
    "TestComposability.test_s_prep_before_qat_prep",  # test_ao_sparsity
    "TestComposability.test_qat_prep_before_s_prep",  # test_ao_sparsity
    "TestFxComposability.test_q_prep_fx_before_s_prep",  # test_ao_sparsity
    "TestFakeSparsity.test_jit_trace",  # test_ao_sparsity
    "TestFakeSparsity.test_masking_logic",  # test_ao_sparsity
    "TestComposability.test_fusion_before_s_prep",  # test_ao_sparsity
    "TestFxComposability.test_s_prep_before_qat_prep_fx",  # test_ao_sparsity
    "TestFxComposability.test_s_prep_q_prep_fx_ref",  # test_ao_sparsity
    "TestComposability.test_s_prep_before_q_prep",  # test_ao_sparsity
    "TestBaseSparsifier.test_state_dict",  # test_ao_sparsity
    "TestComposability.test_q_prep_before_s_prep",  # test_ao_sparsity
    "TestComposability.test_s_prep_before_fusion",  # test_ao_sparsity
    "TestBaseStructuredSparsifier.test_prepare_conv2d",  # test_ao_sparsity
    "TestQuantizeFx.test_conv_transpose_relu_not_reference",  # test_quantization
    "TestPT2ERepresentation.test_qdq",  # test_quantization
    "TestQuantizeFx.test_custom_module_class",  # test_quantization
    "TestQuantizedConv.test_qconv2d_sum_relu_float_output_pt2e",  # test_quantization
    "TestStaticQuantizedModule.test_batch_norm3d",  # test_quantization
    "TestQuantizeFxOps.test_conv_transpose_1d",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn2d.test_qat_conv_bn_relu_fusion",  # test_quantization
    "TestQuantizeFxOps.test_fixed_qparams_ops_qint8",  # test_quantization
    "TestStaticQuantizedModule.test_channel_shuffle",  # test_quantization
    "TestQuantizedTensor.test_qtensor_resize",  # test_quantization
    "TestQuantizeFx.test_state_dict",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_extract_weights_conv_fun_qat",  # test_quantization
    "TestQuantizeFx.test__convert_to_reference_decomposed_fx_dynamic_quant",  # test_quantization
    "TestQuantizedOps.test_qtanh",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn2d.test_qat_conv_bn_fusion",  # test_quantization
    "TestQuantizePT2E.test_fold_quantize",  # test_quantization
    "TestQuantizeFx.test_static_lstm_with_custom_fixed_qparams",  # test_quantization
    "TestEqualizeFx.test_input_weight_equalization_graphs",  # test_quantization
    "TestComparatorOps.test_compare_tensor_scalar",  # test_quantization
    "TestQuantizedFunctionalOps.test_grid_sample",  # test_quantization
    "TestQuantizeFxOps.test_chunk",  # test_quantization
    "TestXNNPACKQuantizer.test_gru",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn2d.test_prepare_qat_conv_bn_fusion_getitem_placeholder",  # test_quantization
    "TestDynamicQuantizedModule.test_dynamic_conv3d",  # test_quantization
    "TestQuantizeFx.test_quantized_model_type",  # test_quantization
    "TestQuantizedOps.test_equal",  # test_quantization
    "TestQuantizedOps.test_qelu",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn2d.test_qat_conv_bn_relu_fusion_no_conv_bias",  # test_quantization
    "TestQuantizedTensor.test_qtensor_equal",  # test_quantization
    "TestQuantizedTensor.test_qtensor_index_put_cpu",  # test_quantization
    "TestQuantizedConv.test_qconv3d_relu",  # test_quantization
    "TestQuantizedConv.test_qconv3d",  # test_quantization
    "TestXNNPACKQuantizer.test_propagate_annotation",  # test_quantization
    "TestQuantizedTensor.test_choose_qparams_optimized",  # test_quantization
    "TestXNNPACKQuantizer.test_linear_gru",  # test_quantization
    "TestDynamicQuantizedModule.test_cell_api",  # test_quantization
    "TestQuantizedOps.test_interpolate",  # test_quantization
    "TestQuantizeFx.test_conv_transpose_reference",  # test_quantization
    "TestPT2ERepresentation.test_conv2d",  # test_quantization
    "TestQuantizeFxOps.test_embedding",  # test_quantization
    "TestQuantizedTensor.test_qtensor_float_assignment",  # test_quantization
    "TestFXNumericSuiteNShadows.test_qconfig_multi_mapping_from_list",  # test_quantization
    "TestXNNPACKQuantizer.test_conv_linear",  # test_quantization
    "TestQuantizedOps.test_qadd_broadcast",  # test_quantization
    "TestFXNumericSuiteNShadows.test_qconfig_multi_mapping_ordering",  # test_quantization
    "TestFXNumericSuiteNShadows.test_linear_relu_mod",  # test_quantization
    "TestQuantizedOps.test_sigmoid_non_observed",  # test_quantization
    "TestStaticQuantizedModule.test_sigmoid",  # test_quantization
    "TestQuantizedOps.test_mean",  # test_quantization
    "TestQuantizeFx.test_shape_followed_by_quantized_op",  # test_quantization
    "TestQuantizedTensor.test_decomposed_quantize_per_tensor_bfloat16_input",  # test_quantization
    "TestQuantizeFxOps.test_clamp",  # test_quantization
    "TestQuantizeFxOps.test_conv_module",  # test_quantization
    "TestQuantizedOps.test_qmul_relu_different_qparams",  # test_quantization
    "TestQuantizeFx.test_attention",  # test_quantization
    "TestQuantizeFxOps.test_conv_transpose_2d",  # test_quantization
    "TestStaticQuantizedModule.test_relu",  # test_quantization
    "TestQuantizedOps.test_linear_bias_unpack",  # test_quantization
    "TestPT2ERepresentation.test_dynamic_linear",  # test_quantization
    "TestQuantizeFxModels.test_resnet_base",  # test_quantization
    "TestQuantizeFxOps.test_qbatch_norm",  # test_quantization
    "TestQNNPackOps.test_qnnpack_sigmoid",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_extract_weights_linear_fun_qat",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn1d.test_qat_update_shared_qspec",  # test_quantization
    "TestQuantizeFxModels.test_qat_embedding_linear",  # test_quantization
    "TestQuantizePT2E.test_fold_all_ops_before_quantize",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_int8_shadows_fp32_coverage",  # test_quantization
    "TestQuantizePT2E.test_fold_quantize_per_channel",  # test_quantization
    "TestQuantizationDocs.test_quantization_doc_qat",  # test_quantization
    "TestQuantizedOps.test_custom_module_lstm",  # test_quantization
    "TestStaticQuantizedModule.test_pool_api",  # test_quantization
    "TestQuantizeFx.test_quant_output_always_observed",  # test_quantization
    "TestQuantizeEagerOps.test_functional_module",  # test_quantization
    "TestFakeQuantizeOps.test_learnable_backward_per_channel_cuda",  # test_quantization
    "TestFXNumericSuiteNShadows.test_partial_qconfig_mapping",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn1d.test_qat_conv_no_bias",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_layer_names",  # test_quantization
    "TestQuantizedConv.test_qconv1d_unpack",  # test_quantization
    "TestQuantizeFx.test_custom_module_class_input_has_duplicate_nodes",  # test_quantization
    "TestXNNPACKQuantizer.test_linear_relu",  # test_quantization
    "TestSerialization.test_linear_relu_package_quantization_transforms",  # test_quantization
    "TestDynamicQuantizedModule.test_gru_api",  # test_quantization
    "TestQuantizeFx.test_qconfig_for_call_method",  # test_quantization
    "TestXNNPACKQuantizer.test_conv1d_with_conv2d",  # test_quantization
    "TestQuantizedOps.test_qsoftmax",  # test_quantization
    "TestQuantizedEmbeddingOps.test_embedding_bag_2bit",  # test_quantization
    "TestObserver.test_per_tensor_observers",  # test_quantization
    "TestQuantizedTensor.test_qtensor_per_channel_load_save",  # test_quantization
    "TestQuantizedOps.test_max_pool2d_nhwc",  # test_quantization
    "TestFXGraphMatcher.test_simple_fun",  # test_quantization
    "TestEqualizeFx.test_selective_equalization",  # test_quantization
    "TestQuantizeFx.test__convert_to_reference_decomposed_fx",  # test_quantization
    "TestQuantizeFx.test_remove_qconfig",  # test_quantization
    "TestBackendConfig.test_backend_op_config_set_input_type_to_index",  # test_quantization
    "TestQuantizedLinear.test_qlinear_relu",  # test_quantization
    "TestQuantizePT2E.test_constant_prop_preserve_metadata",  # test_quantization
    "TestQuantizedTensor.test_qtensor_permute",  # test_quantization
    "TestQuantizedTensor.test_quantize_per_channel_sub_byte",  # test_quantization
    "TestStaticQuantizedModule.test_conv3d_relu_api",  # test_quantization
    "TestPT2ERepresentation.test_add_relu",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn2d.test_qat_conv_bn_fusion_no_conv_bias",  # test_quantization
    "TestQuantizeFxOps.test_layer_norm",  # test_quantization
    "TestQuantizeFxOps.test_add_relu",  # test_quantization
    "TestQuantizedOps.test_qthreshold",  # test_quantization
    "TestXNNPACKQuantizer.test_dynamic_linear_with_conv",  # test_quantization
    "TestQuantizeFx.test_custom_module_class_input_has_multiple_users",  # test_quantization
    "TestXNNPACKQuantizer.test_add_mul_scalar",  # test_quantization
    "TestQuantizedTensor.test_qtensor_load_save",  # test_quantization
    "TestFXNumericSuiteNShadows.test_add_loggers_functions",  # test_quantization
    "TestFXNumericSuiteNShadows.test_linear_mod",  # test_quantization
    "TestXNNPACKQuantizer.test_add_and_inplace_add",  # test_quantization
    "TestQuantizeFxOps.test_elu",  # test_quantization
    "TestQuantizeFx.test_conv_lowering",  # test_quantization
    "TestQuantizedFunctionalOps.test_conv1d_api",  # test_quantization
    "TestQuantizeFx.test_lowering_functional_linear_with_kwargs",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn1d.test_qat_conv_bn_fusion_literal_args",  # test_quantization
    "TestQuantizedTensor.test_per_tensor_qtensor_to_memory_format",  # test_quantization
    "TestQuantizedOps.test_cat_nhwc",  # test_quantization
    "TestStaticQuantizedModule.test_conv2d_relu_api",  # test_quantization
    "TestQuantizeFx.test_prepare_custom_config_set_standalone_module_class",  # test_quantization
    "TestQuantizeFxOps.test_mul",  # test_quantization
    "TestQuantizedTensor.test_qtensor_quant_dequant",  # test_quantization
    "TestQuantizeFx.test_qconfig_module_name_regex",  # test_quantization
    "TestQuantizeFx.test_qconfig_module_type",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn2d.test_qat_inplace_add_relu",  # test_quantization
    "TestStaticQuantizedModule.test_embedding_api",  # test_quantization
    "TestQuantizePT2E.test_speed",  # test_quantization
    "TestStaticQuantizedModule.test_dropout",  # test_quantization
    "TestQNNPackOps.test_qnnpack_sigmoid_sweep",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_match_activations_fun_qat",  # test_quantization
    "TestQuantizedOps.test_qclamp",  # test_quantization
    "TestQuantizedOps.test_avg_pool2d",  # test_quantization
    "TestQuantizedOps.test_add_scalar_relu",  # test_quantization
    "TestQuantizedTensor.test_decomposed_dequantize_per_tensor",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_int8_shadows_int8_mod",  # test_quantization
    "TestQuantizedTensor.test_qtensor_dtypes",  # test_quantization
    "TestQuantizedOps.test_quantized_equal",  # test_quantization
    "TestQuantizeFx.test_fold_quant_dequant",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn1d.test_prepare_qat_conv_bn_fusion_getitem_placeholder",  # test_quantization
    "TestFakeQuantizeOps.test_fake_quant_per_channel_qparam_range",  # test_quantization
    "TestQuantizedTensor.test_fp16_saturate_op",  # test_quantization
    "TestQuantizedFunctionalOps.test_conv2d_api",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn1d.test_qat_conv_bn_fusion",  # test_quantization
    "TestQuantizedOps.test_avg_pool2d_nhwc",  # test_quantization
    "TestQuantizeFxOps.test_quantized_conv_relu",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn2d.test_qat_update_shared_qspec",  # test_quantization
    "TestQuantizeFx.test_fp32_sum",  # test_quantization
    "TestQuantizedTensor.test_per_channel_qtensor_to_memory_format",  # test_quantization
    "TestQuantizeFx.test_dict_output",  # test_quantization
    "TestQuantizedConv.test_qconv2d_pt2e",  # test_quantization
    "TestQuantizedLinear.test_qlinear_unpack",  # test_quantization
    "TestQuantizeFx.test_lowering_functional_conv_with_kwargs",  # test_quantization
    "TestQuantizePT2E.test_reentrant",  # test_quantization
    "TestFXNumericSuiteNShadows.test_add_loggers_conv_bn_relu_fusion_quant",  # test_quantization
    "TestQuantizedLinear.test_qlinear_qnnpack_free_memory_and_unpack",  # test_quantization
    "TestDynamicQuantizedOps.test_qrnncell",  # test_quantization
    "TestFXNumericSuiteNShadows.test_functions",  # test_quantization
    "TestQuantizedOps.test_qmul_broadcast",  # test_quantization
    "TestQuantizeFx.test_dequantize",  # test_quantization
    "TestDynamicQuantizedModule.test_dynamic_convtranspose3d",  # test_quantization
    "TestQuantizeFx.test_static_lstm_consume_tuple",  # test_quantization
    "TestXNNPACKQuantizer.test_conv_linear_no_permute",  # test_quantization
    "TestReferenceQuantizedModule.test_rnn_cell",  # test_quantization
    "TestBackendConfig.test_backend_op_config_add_dtype_config",  # test_quantization
    "TestStaticQuantizedModule.test_conv2d_add",  # test_quantization
    "TestQuantizedConv.test_qconv2d_relu_pt2e",  # test_quantization
    "TestPT2ERepresentation.test_add",  # test_quantization
    "TestQuantizedEmbeddingOps.test_embedding_bag_4bit",  # test_quantization
    "TestQuantizedTensor.test_qtensor_channel_float_assignment",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_match_activations_fun_ptq",  # test_quantization
    "TestXNNPACKQuantizer.test_mul_float32_max",  # test_quantization
    "TestFXNumericSuiteNShadows.test_add_loggers_linear_mod_quant_quant",  # test_quantization
    "TestQuantizeFx.test_standalone_module_float_interface",  # test_quantization
    "TestFXNumericSuiteNShadows.test_custom_functions_and_tracer",  # test_quantization
    "TestQuantizeFxOps.test_multiple_qconfigs_for_single_value",  # test_quantization
    "TestQuantizedOps.test_leaky_relu_observed_output",  # test_quantization
    "TestFakeQuantizeOps.test_learnable_forward_per_tensor_cuda",  # test_quantization
    "TestQuantizedTensor.test_repeat",  # test_quantization
    "TestStaticQuantizedModule.test_linear_leaky_relu",  # test_quantization
    "TestBitsCPU.test_subclass_cpu",  # test_quantization
    "TestFakeQuantizeOps.test_learnable_backward_per_channel_cpu",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_add_shadow_loggers_fun_ptq",  # test_quantization
    "TestQuantizeFx.test_static_lstm",  # test_quantization
    "TestQuantizeFx.test_qconfig_qat_module_type",  # test_quantization
    "TestQuantizedOps.test_mul_scalar_relu",  # test_quantization
    "TestQuantizedTensor.test_qtensor_per_channel_permute",  # test_quantization
    "TestStaticQuantizedModule.test_batch_norm2d",  # test_quantization
    "TestDynamicQuantizedModule.test_dynamic_conv2d",  # test_quantization
    "TestGenerateNumericDebugHandle.test_quantize_pt2e_preserve_handle",  # test_quantization
    "TestQuantizedTensor.test_qtensor_sub_byte_not_aligned_cols",  # test_quantization
    "TestQuantizeFx.test_conv_linear_reference",  # test_quantization
    "TestQuantizePT2E.test_composable_quantizer_linear_conv",  # test_quantization
    "TestBackendConfig.test_backend_op_config_from_dict",  # test_quantization
    "TestFakeQuantizeOps.test_learnable_backward_per_tensor_cuda",  # test_quantization
    "TestXNNPACKQuantizer.test_linear_with_dynamic_shape",  # test_quantization
    "TestQuantizedOps.test_empty_batch",  # test_quantization
    "TestQuantizeFx.test_symmetric_qnnpack_qconfig_mapping",  # test_quantization
    "TestQuantizedEmbeddingOps.test_embedding_bag_2d_indices",  # test_quantization
    "TestQuantizeFx.test_symmetric_qnnpack_qat_qconfig_mapping",  # test_quantization
    "TestQuantizePT2E.test_save_load",  # test_quantization
    "TestPT2ERepresentation.test_qdq_per_channel",  # test_quantization
    "TestQuantizeFxOps.test_prelu",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_match_activations_fqn",  # test_quantization
    "TestQuantizeFx.test_assert_on_size_after_quant_layer",  # test_quantization
    "TestQuantizedConv.test_qconv1d_pt2e",  # test_quantization
    "TestQuantizeFx.test_conv_linear_not_reference",  # test_quantization
    "TestFakeQuantizeOps.test_forward_per_channel",  # test_quantization
    "TestQuantizeFx.test_qconfig_none",  # test_quantization
    "TestQuantizeFx.test__convert_to_reference_decomposed_fx_per_channel_quant",  # test_quantization
    "TestPadding.test_reflection_pad2d",  # test_quantization
    "TestStaticQuantizedModule.test_quant_dequant_api",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_add_shadow_loggers_fun_qat",  # test_quantization
    "TestDynamicQuantizedOps.test_dynamic_conv2d",  # test_quantization
    "TestQuantizedOps.test_qsoftmax_qnnpack",  # test_quantization
    "TestStaticQuantizedModule.test_prelu",  # test_quantization
    "TestQuantizedEmbeddingOps.test_embedding_bag_byte",  # test_quantization
    "TestQuantizedConv.test_qconv3d_pt2e",  # test_quantization
    "TestQuantizedOps.test_qcelu",  # test_quantization
    "TestReferenceQuantizedModule.test_sparse",  # test_quantization
    "TestQuantizedOps.test_max_pool3d",  # test_quantization
    "TestPadding.test_reflection_pad1d",  # test_quantization
    "TestQuantizedConv.test_qconv2d_sum_relu_pt2e",  # test_quantization
    "TestDynamicQuantizedOps.test_qlstmGRU",  # test_quantization
    "TestFakeQuantizeOps.test_fixed_qparams_fq_module",  # test_quantization
    "TestQuantizeFxOps.test_qmatmul",  # test_quantization
    "TestQuantizeFx.test_conv_transpose_not_reference",  # test_quantization
    "TestUtils.test_get_fqn_to_example_inputs_default_kwargs",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_extract_weights_linear_fun_ptq",  # test_quantization
    "TestQuantizeFxOps.test_cat",  # test_quantization
    "TestQuantizeFx.test_sequential",  # test_quantization
    "TestFakeQuantizeOps.test_learnable_backward_per_tensor_cpu",  # test_quantization
    "TestQuantizedTensor.test_qtensor_quantize_per_channel",  # test_quantization
    "TestDynamicQuantizedModule.test_linear_api",  # test_quantization
    "TestQuantizedTensor.test_qtensor_unsqueeze",  # test_quantization
    "TestQuantizedFunctionalOps.test_relu_api",  # test_quantization
    "TestDynamicQuantizedModule.test_dynamic_convtranspose1d",  # test_quantization
    "TestQuantizeFxOps.test_add",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn2d.test_qat_conv_bn_fusion_literal_args",  # test_quantization
    "TestQuantizedTensor.test_qtensor_fill_per_channel",  # test_quantization
    "TestQuantizeFx.test_dynamic_with_fusion",  # test_quantization
    "TestQuantizeFx.test_convert_qconfig_mapping",  # test_quantization
    "TestQuantizeFx.test_save_observer_state_dict",  # test_quantization
    "TestDynamicQuantizedOps.test_dynamic_convtranspose1d",  # test_quantization
    "TestQuantizePT2E.test_derived_qspec",  # test_quantization
    "TestDynamicQuantizedModule.test_dynamic_convtranspose2d",  # test_quantization
    "TestFuseFx.test_fuse_conv_bn_add_relu_lowering",  # test_quantization
    "TestQuantizedConv.test_qconv2d_sum_pt2e",  # test_quantization
    "TestDynamicQuantizedOps.test_dynamic_convtranspose2d",  # test_quantization
    "TestPT2ERepresentation.test_maxpool2d",  # test_quantization
    "TestQuantizeFx.test_lowering_functional_conv_transpose_with_kwargs",  # test_quantization
    "TestQuantizedOps.test_avg_pool3d_nhwc",  # test_quantization
    "TestQuantizeFx.test_qparams_buffers",  # test_quantization
    "TestStaticQuantizedModule.test_instance_norm",  # test_quantization
    "TestQuantizeFxOps.test_functional_conv",  # test_quantization
    "TestXNNPACKQuantizer.test_qat_dynamic_linear",  # test_quantization
    "TestQuantizedLinear.test_qlinear",  # test_quantization
    "TestQuantizeFx.test_no_obs_between_unmatched_node_and_copy_node",  # test_quantization
    "TestStaticQuantizedModule.test_conv1d_relu_api",  # test_quantization
    "TestXNNPACKQuantizer.test_linear",  # test_quantization
    "TestQuantizeFxOps.test_norm_weight_bias",  # test_quantization
    "TestQuantizeFxOps.test_reshape_fp16",  # test_quantization
    "TestQuantizeFx.test_packed_weight_fused_op",  # test_quantization
    "TestStaticQuantizedModule.test_embedding_bag_api",  # test_quantization
    "TestQuantizedOps.test_advanced_indexing",  # test_quantization
    "TestQuantizeFx.test_conv_bn_relu",  # test_quantization
    "TestQuantizeFx.test_qconfig_for_call_func",  # test_quantization
    "TestQuantizedConv.test_qconv3d_unpack",  # test_quantization
    "TestFakeQuantizeOps.test_fq_module_per_tensor",  # test_quantization
    "TestDynamicQuantizedOps.test_qlinear",  # test_quantization
    "TestStaticQuantizedModule.test_layer_norm",  # test_quantization
    "TestQuantizedOps.test_leaky_relu",  # test_quantization
    "TestFakeQuantize.test_fq_module_per_channel",  # test_quantization
    "TestQuantizeFxOps.test_getitem",  # test_quantization
    "TestQuantizeFx.test_mixed_dtypes",  # test_quantization
    "TestQuantizeFx.test_linear_tanh_lowering",  # test_quantization
    "TestStaticQuantizedModule.test_conv3d_api",  # test_quantization
    "TestStaticQuantizedModule.test_linear_tanh",  # test_quantization
    "TestQuantizedOps.test_sigmoid",  # test_quantization
    "TestQuantizedConv.test_qconv2d_unpack",  # test_quantization
    "TestQuantizedOps.test_qgelu",  # test_quantization
    "TestQuantizedTensor.test_qtensor_view",  # test_quantization
    "TestUtils.test_get_fqn_to_example_inputs_complex_args",  # test_quantization
    "TestQuantizeFxOps.test_embedding_bag",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn1d.test_qat_inplace_add_relu",  # test_quantization
    "TestQuantizePT2E.test_embedding_conv_linear_quantization",  # test_quantization
    "TestFakeQuantizeOps.test_learnable_forward_per_channel_cuda",  # test_quantization
    "TestPT2ERepresentation.test_static_linear",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn1d.test_qat_conv_bn_fusion_no_conv_bias",  # test_quantization
    "TestQuantizedOps.test_qadd_relu_different_qparams",  # test_quantization
    "TestQuantizeFxOps.test_qbatch_norm_relu",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_shadow_activations_fqn",  # test_quantization
    "TestDynamicQuantizedOps.test_dynamic_conv3d",  # test_quantization
    "TestFXNumericSuiteNShadows.test_add_loggers_linear_mod_quant_fp32",  # test_quantization
    "TestFXNumericSuiteNShadows.test_qconfig_multi_mapping_end_to_end",  # test_quantization
    "TestQuantizedConv.test_qconv2d",  # test_quantization
    "TestFXNumericSuiteNShadows.test_logger_enabled_and_save_activations_flags",  # test_quantization
    "TestXNNPACKQuantizer.test_add_mul_long",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn1d.test_qat_conv_bn_relu_fusion",  # test_quantization
    "TestDynamicQuantizedModule.test_lstm_api",  # test_quantization
    "TestComparatorOps.test_compare_tensor_tensor",  # test_quantization
    "TestQuantizeFxModels.test_qat_functional_linear",  # test_quantization
    "TestQuantizeFxOps.test_functional_linear",  # test_quantization
    "TestQuantizedTensor.test_per_channel_qtensor_creation_cpu",  # test_quantization
    "TestQuantizedOps.test_max_pool2d_cudnn",  # test_quantization
    "TestQNNPackOps.test_qnnpack_mul",  # test_quantization
    "TestUtils.test_get_fqn_to_example_inputs_simple",  # test_quantization
    "TestQuantizeEagerQATNumerics.test_conv_bn_folded_vs_unfolded",  # test_quantization
    "TestQNNPackOps.test_qnnpack_maxpool2d",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_extract_weights_conv_fun_ptq",  # test_quantization
    "TestQuantizeFx.test_qconfig_module_name_object_type_order",  # test_quantization
    "TestDynamicQuantizedModule.test_dynamic_conv1d",  # test_quantization
    "TestEqualizeFx.test_input_weight_equalization_convert",  # test_quantization
    "TestFXNumericSuiteNShadows.test_add_loggers_linear_mod_fp32_quant",  # test_quantization
    "TestQNNPackOps.test_qnnpack_add",  # test_quantization
    "TestStaticQuantizedModule.test_conv2d_api",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn2d.test_qat_conv_bn_bias_derived_qspec",  # test_quantization
    "TestQuantizedOps.test_std",  # test_quantization
    "TestBitsCPU.test_cat_cpu",  # test_quantization
    "TestFakeQuantizeOps.test_forward_per_tensor",  # test_quantization
    "TestQNNPackOps.test_qnnpack_tanh",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_loggers_preserve_qat_numerics",  # test_quantization
    "TestDynamicQuantizedOps.test_qlinear_legacy",  # test_quantization
    "TestQuantizedOps.test_cat",  # test_quantization
    "TestXNNPACKQuantizer.test_conv1d",  # test_quantization
    "TestQuantizedOps.test_qadd_relu_same_qparams",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_shadow_loggers_preserve_qat_numerics",  # test_quantization
    "TestXNNPACKQuantizer.test_dynamic_linear",  # test_quantization
    "TestXNNPACKQuantizer.test_dynamic_linear_int4_weight",  # test_quantization
    "TestFakeQuantizeOps.test_learnable_forward_per_channel_cpu",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_int8_shadows_fp32_simple",  # test_quantization
    "TestXNNPACKQuantizer.test_conv2d",  # test_quantization
    "TestQuantizeFx.test_linear_leaky_relu_lowering",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn1d.test_qat_conv_bn_bias_derived_qspec",  # test_quantization
    "TestFakeQuantizeOps.test_backward_per_tensor",  # test_quantization
    "TestQuantizedTensor.test_qtensor_int_repr",  # test_quantization
    "TestQuantizedEmbeddingOps.test_embedding_2d_indices",  # test_quantization
    "TestQuantizedTensor.test_qtensor_sub_byte_aligned_cols",  # test_quantization
    "TestQuantizeFxOps.test_leaky_relu",  # test_quantization
    "TestFXGraphMatcher.test_simple_fusion",  # test_quantization
    "TestXNNPACKQuantizer.test_mul_and_inplace_mul",  # test_quantization
    "TestQuantizedOps.test_max_pool2d",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn2d.test_qat_preserve_source_fn_stack",  # test_quantization
    "TestQuantizeFx.test_qparams_fqn",  # test_quantization
    "TestFakeQuantizeOps.test_backward_per_channel",  # test_quantization
    "TestQuantizeFx.test_conv_transpose_relu_reference",  # test_quantization
    "TestQuantizedConv.test_qconv2d_hardtanh_pt2e",  # test_quantization
    "TestQuantizedTensor.test_decomposed_quantize_per_tensor",  # test_quantization
    "TestQNNPackOps.test_qnnpack_add_broadcast",  # test_quantization
    "TestQuantizeFxOps.test_linear_module",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn2d.test_qat_conv_no_bias",  # test_quantization
    "TestFakeQuantizeOps.test_learnable_forward_per_tensor_cpu",  # test_quantization
    "TestStaticQuantizedModule.test_hard_swish",  # test_quantization
    "TestQuantizeFxModels.test_qat_embeddingbag_linear",  # test_quantization
    "TestStaticQuantizedModule.test_conv2d_add_relu",  # test_quantization
    "TestQuantizedOps.test_interpolate3d",  # test_quantization
    "TestDynamicQuantizedOps.test_dynamic_convtranspose3d",  # test_quantization
    "TestQNNPackOps.test_qnnpack_relu",  # test_quantization
    "TestFXNumericSuiteCoreAPIs.test_int8_shadows_int8_fun",  # test_quantization
    "TestStaticQuantizedModule.test_conv1d_api",  # test_quantization
    "TestQuantizationDocs.test_quantization_doc_fx",  # test_quantization
    "TestQuantizedOps.test_channel_shuffle",  # test_quantization
    "TestQuantizedOps.test_hardtanh",  # test_quantization
    "TestQuantizeFx.test_qconfig_function",  # test_quantization
    "TestQuantizeFx.test_ref_conv_module",  # test_quantization
    "TestBackendConfig.test_backend_op_config_set_num_tensor_args_to_observation_type",  # test_quantization
    "TestQuantizedOps.test_max_pool1d",  # test_quantization
    "TestStaticQuantizedModule.test_linear",  # test_quantization
    "TestQuantizeFxOps.test_mul_relu",  # test_quantization
    "TestQuantizePT2E.test_groupwise_per_channel_quant",  # test_quantization
    "TestQuantizeFxOps.test_hardswish",  # test_quantization
    "TestQuantizedTensor.test_qtensor_cpu",  # test_quantization
    "TestQuantizedConv.test_qconv_transpose1d",  # test_quantization
    "TestEqualizeFx.test_input_weight_equalization_results",  # test_quantization
    "TestQuantizedFunctionalOps.test_conv3d_api",  # test_quantization
    "TestQuantizeFx.test_linear_bn",  # test_quantization
    "TestStaticQuantizedModule.test_elu",  # test_quantization
    "TestQuantizeFx.test_standalone_module_quantized_interface",  # test_quantization
    "TestQuantizedOps.test_qprelu",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn1d.test_qat_preserve_source_fn_stack",  # test_quantization
    "TestQuantizedOps.test_qmatmul",  # test_quantization
    "TestPadding.test_constant_padNd",  # test_quantization
    "TestQuantizedConv.test_qconv2d_relu",  # test_quantization
    "TestQuantizedConv.test_qconv2d_add",  # test_quantization
    "TestQuantizedTensor.test_qtensor_reshape",  # test_quantization
    "TestQuantizeFx.test_ref_linear_module",  # test_quantization
    "TestQuantizePT2EQAT_ConvBn1d.test_qat_conv_bn_relu_fusion_no_conv_bias",  # test_quantization
    "TestReferenceQuantizedModule.test_rnn",  # test_quantization
    "TestQuantizedConv.test_qconv1d",  # test_quantization
    "TestQuantizedTensor.test_choose_qparams",  # test_quantization
    "TestQuantizedConv.test_qconv1d_relu",  # test_quantization
    "TestFXNumericSuiteNShadows.test_conv_bn_relu_mod",  # test_quantization
    "TestQuantizedConv.test_qconv2d_add_relu",  # test_quantization
    "TestStaticQuantizedModule.test_group_norm",  # test_quantization
    "TestStaticQuantizedModule.test_leaky_relu",  # test_quantization
    "TestQuantizedTensor.test_torch_qtensor_deepcopy",  # test_quantization
    "TestXNNPACKQuantizer.test_obs_sharing_ops",  # test_quantization
    "TestStaticQuantizedModule.test_linear_relu",  # test_quantization
    "TestQuantizedOps.test_avg_pool3d",  # test_quantization
    "TestQuantizedTensor.test_quantize_per_channel_float_qparams",  # test_quantization
    "TestQuantizedOps.test_qmul_relu_same_qparams",  # test_quantization
    "TestQuantizeFxOps.test_instance_norm",  # test_quantization
    "TestQuantizedTensor.test_qtensor_legacy_new_failure",  # test_quantization
    "TestXNNPACKQuantizerModels.test_resnet18",  # test_quantization.py
    "TestFXGraphMatcherModels.test_mobilenet_v2_qat",  # test_quantization.py
    "TestQuantizePT2EQATModels.test_qat_resnet18",  # test_quantization.py
    "TestQuantizePT2EQATModels.test_qat_mobilenet_v2",  # test_quantization.py
    "TestObserver.test_per_channel_observers",  # test_quantization.py
    "TestCustomOp.test_define_with_tags_single",  # test_custom_ops
    "TestCustomOp.test_autogen_aten_ops_are_pt2_compliant",  # test_custom_ops
    "TestCustomOp.test_define_with_tags_list",  # test_custom_ops
    "TestCustomOp.test_impl_device_cpu",  # test_custom_ops
    "TestCustomOp.test_impl_device_function",  # test_custom_ops
    "TestCustomOp.test_builtin_torchscript_ops",  # test_custom_ops
    "TestCustomOpTestingCPU.test_missing_functionalization_cpu",  # test_custom_ops
    "TestCustomOp.test_define_with_tags_tuple",  # test_custom_ops
    "TestCustomOp.test_builtin_aten_ops_are_pt2_compliant",  # test_custom_ops
    "TestGenerateOpcheckTests.test_opcheck_bad_op",  # test_custom_ops
    "TestCustomOp.test_legacy_define",  # test_custom_ops
    "TestPythonRegistration.test_alias_analysis",  # test_python_dispatch
    "TestPythonDispatch.test_torch_dispatch_mode_subclass_priority",  # test_python_dispatch
    "TestPythonDispatch.test_strides_slow_path",  # test_python_dispatch
    "TestPythonDispatch.test_invalid_ret",  # test_python_dispatch
    "TestPythonDispatch.test_dim_slowpath",  # test_python_dispatch
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_conv2d_cpu",  # test_python_dispatch
    "TestPythonDispatch.test_fancy_strides",  # test_python_dispatch
    "TestPythonDispatch.test_layout_slow_path",  # test_python_dispatch
    "TestPythonDispatch.test_dispatch_super_dont_autograd",  # test_python_dispatch
    "TestPythonDispatch.test_sizes_slow_path",  # test_python_dispatch
    "TestPythonRegistration.test_finalizer",  # test_python_dispatch
    "TestPythonDispatch.test_dispatch_super_call_list_arg",  # test_python_dispatch
    "TestPythonDispatch.test_is_contiguous_slow_path",  # test_python_dispatch
    "TestPythonRegistration.test_override_cpu_sum",  # test_python_dispatch
    "TestPythonDispatch.test_mode_with_make_subclass",  # test_python_dispatch
    "TestPythonDispatch.test_multiple_ops_subclass",  # test_python_dispatch
    "TestPythonDispatch.test_subclass_autograd_device_check",  # test_python_dispatch
    "TestPythonDispatch.test_data_ptr_respects_numel_slow_path",  # test_python_dispatch
    "TestPythonDispatch.test_make_subclass_with_modes",  # test_python_dispatch
    "TestPythonDispatch.test_dispatch_super_call",  # test_python_dispatch
    "TestPythonDispatch.test_subclass_priority",  # test_python_dispatch
    "TestPythonDispatch.test_exception_handling",  # test_python_dispatch
    "TestPythonDispatch.test_list_ret",  # test_python_dispatch
    "LoggingTests.test_trace_source_nested",  # dynamo/test_logging
    "LoggingTests.test_guards_recompiles",  # dynamo/test_logging
    "LoggingTests.test_inductor_info",  # dynamo/test_logging
    "LoggingTests.test_output_code",  # dynamo/test_logging
    "LoggingTests.test_graph_code",  # dynamo/test_logging
    "LoggingTests.test_graph_sizes",  # dynamo/test_logging
    "LoggingTests.test_recompiles",  # dynamo/test_logging
    "LoggingTests.test_inductor_error",  # dynamo/test_logging
    "LoggingTests.test_graph",  # dynamo/test_logging
    "LoggingTests.test_custom_format_exc",  # dynamo/test_logging
    "LoggingTests.test_custom_format",  # dynamo/test_logging
    "LoggingTests.test_trace_source_cond",  # dynamo/test_logging
    "LoggingTests.test_multiline_format",  # dynamo/test_logging
    "LoggingTests.test_aot_joint_graph",  # dynamo/test_logging
    "LoggingTests.test_inductor_debug",  # dynamo/test_logging
    "LoggingTests.test_bytecode",  # dynamo/test_logging
    "LoggingTests.test_graph_sizes_dynamic",  # dynamo/test_logging
    "LoggingTests.test_dynamo_error",  # dynamo/test_logging
    "LoggingTests.test_dynamo_debug",  # dynamo/test_logging
    "LoggingTests.test_aot_graphs",  # dynamo/test_logging
    "LoggingTests.test_dynamo_info",  # dynamo/test_logging
    "LoggingTests.test_graph_breaks",  # dynamo/test_logging
    "LoggingTests.test_aot",  # dynamo/test_logging
    "TestAttnBiasCPU.test_is_causal_equals_upper_left_shape2_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_is_causal_equals_upper_left_shape3_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_is_causal_and_mask_fails_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_is_causal_equals_upper_left_shape1_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_is_causal_equals_upper_left_shape0_cpu",  # test_transformers.py
}

# see NOTE [dynamo_test_failures.py] for more details
dynamo_skips = {
    "TestMatmulOperator.test_matmul_raises",
    "TestMatmulOperator.test_exceptions",
    "TestMatmulOperator.test_matmul_inplace",
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
    "TestLinalgCPU.test_matmul_small_brute_force_3d_Nd_cpu_complex64",  # known py311 fail
    "TestLinalgCPU.test_matmul_small_brute_force_3d_Nd_cpu_int64",  # known py311 fail
    "TestLinalgCPU.test_matmul_small_brute_force_3d_Nd_cpu_float32",  # known py311 fail
    "TestCustomOpTestingCPU.test_incorrect_schema_mutation_cpu",  # test_custom_ops
    "TestArgmaxArgminCommon.test_np_vs_ndarray_arr_method_argmax_np_method0",
    "TestArgmaxArgminCommon.test_np_vs_ndarray_arr_method_argmin_np_method1",
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
    "TestAttnBiasCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape3_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape0_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape1_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape2_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape3_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_CausalVariant_LOWER_RIGHT_shape0_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape2_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_CausalVariant_UPPER_LEFT_shape1_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_CausalVariant_LOWER_RIGHT_shape3_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_CausalVariant_UPPER_LEFT_shape0_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_CausalVariant_LOWER_RIGHT_shape1_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_CausalVariant_LOWER_RIGHT_shape2_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_CausalVariant_UPPER_LEFT_shape3_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_CausalVariant_LOWER_RIGHT_shape0_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_CausalVariant_UPPER_LEFT_shape2_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_CausalVariant_UPPER_LEFT_shape1_cpu",  # known py38 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_1_shape0_cpu",  # known py311 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_2_shape2_cpu",  # known py311 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_2_shape0_cpu",  # known py311 fail
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_2_shape3_cpu",  # known py311 fail
    "TestAttnBiasCUDA.test_causal_variants_compile_causal_variant_CausalVariant_LOWER_RIGHT_shape3_CUDA",  # known py38 fail
    "TestAttnBiasCUDA.test_causal_variants_compile_causal_variant_CausalVariant_UPPER_LEFT_shape0_CUDA",  # known py38 fail
    "TestAttnBiasCUDA.test_causal_variants_compile_causal_variant_CausalVariant_LOWER_RIGHT_shape1_CUDA",  # known py38 fail
    "TestAttnBiasCUDA.test_causal_variants_compile_causal_variant_CausalVariant_LOWER_RIGHT_shape2_CUDA",  # known py38 fail
    "TestAttnBiasCUDA.test_causal_variants_compile_causal_variant_CausalVariant_UPPER_LEFT_shape3_CUDA",  # known py38 fail
    "TestAttnBiasCUDA.test_causal_variants_compile_causal_variant_CausalVariant_LOWER_RIGHT_shape0_CUDA",  # known py38 fail
    "TestAttnBiasCUDA.test_causal_variants_compile_causal_variant_CausalVariant_UPPER_LEFT_shape2_CUDA",  # known py38 fail
    "TestAttnBiasCUDA.test_causal_variants_compile_causal_variant_CausalVariant_UPPER_LEFT_shape1_CUDA",  # known py38 fail
    "TestTransformersCPU.test_decoder_padding_and_src_mask_bool_cpu",  # known py311 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_2_shape3_cpu",  # known py311 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_1_shape3_cpu",  # known py311 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_1_shape2_cpu",  # known py311 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_1_shape1_cpu",  # known py311 fail
    "TestAttnBiasCPU.test_causal_variants_causal_variant_2_shape1_cpu",  # known py311 fail
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
    "MiscTests.test_auto_functionalize_optional",  # weird
    "MiscTests.::test_auto_functionalize_with_returns",  # weird
    "MiscTests.test_generate_trivial_abstract_impl",  # weird
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
    "TestBasicsCPU.test_invalid_sparse_csr_values_cpu",  # known py38 fail
    "TestBasicsCPU.test_invalid_sparse_coo_values_cpu",  # known py38 fail
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
    "TestProfiler.test_source_multithreaded_basic_work_in_main_thread_False",
    "TestProfiler.test_source_multithreaded_complex_work_in_main_thread_False",
    "TestProfiler.test_source_multithreaded_open_in_scope_work_in_main_thread_False",
    "TestProfiler.test_source_multithreaded_multiple_preexisting_work_in_main_thread_False",
    "TestProfiler.test_source_multithreaded_close_in_scope_work_in_main_thread_False",
    "TestProfiler.test_profiler_disable_fwd_bwd_link",
    "TestProfiler.test_profiler_metadata",
    "TestSerializeCustomClass.test_custom_class",
    "TestTensorExprPyBind.test_kernel_with_transpose",
    "TestTensorExprPyBind.test_kernel_with_custom_lowering",
    "TestTensorExprPyBind.test_kernel_with_expand",
    "TestTensorExprPyBind.test_kernel_with_permute",
    "TestTensorExprPyBind.test_kernel_with_scalar_inputs",
    "TestTensorExprPyBind.test_kernel_with_t",
    "TestTensorExprPyBind.test_kernel_with_tensor_inputs",
    "TestFX.test_assert",  # known py38 fail
    "TestFX.test_annotations_empty_tuple",  # known py38 fail
    "TestDataLoaderPersistentWorkers.test_early_exit",  # known py38 fail
    "TestDataLoader.test_fd_limit_exceeded",  # known py38 fail
    "TestDataLoaderPersistentWorkers.test_fd_limit_exceeded",  # known py38 fail
    "TestLazyDynamicOps.test_nonzero_dynamic",  # known py311 fail
    "TestLazyTensor.test_view_mark_step_preserved",  # known py311 fail
    "TestRepackage.test_repackage_import_indirectly_via_parent_module",  # known py311 fail
    "TestPackageScript.test_load_shared_tensors_repackaged",  # known py311 fail
    "TestFXExperimental.test_optimize_for_inference_cpu_torchvision",
    "TestXNNPACKOps.test_linear_1d_input",  # flaky
    "TestXNNPACKOps.test_conv2d",  # flaky
    "TestXNNPACKOps.test_conv2d_transpose",  # flaky
    "TestProfilerTree.test_profiler_experimental_tree",  # known py311 fail
    "TestExperiment.test_mark_strict_with_container_type",  # known py311 fail
    "TestExprHandlePyBind.test_unary_ops",  # known py38 fail
    "TestTorch.test_bmm_multithreaded",
    "TestTorch.test_terminate_handler_on_crash",  # known py38 fail
    "TestTorch.test_storage_casts",  # known py311 fail
    "TestAutograd.test_hook_closure_cycle_use_custom_function_True_use_tensor_hook_True",  # known py311 fail
    "TestQuantizeFxOps.test_fixed_qparams_ops",  # known py38 fail
    "TestQuantizeFxOps.test_int8_input_no_unnecessary_fq",  # known py38 fail
    "TestQuantizeFxOps.test_general_value_ops",  # known py38 fail
    "TestQuantizeFxOps.test_fixed_qparams_ops_wrong_qconfig",  # known py38 fail
    "TestQuantizeFxOps.test_narrow",  # known py38 fail
    "TestQuantizeFxOps.test_float_functional",  # known py38 fail
    "TestQuantizeFxOps.test_copy_node_fp32_input",  # known py38 fail
    "TestQuantizeFxOps.test_ave_pool_with_custom_cfg",  # known py38 fail
    "TestQuantizeFxOps.test_pixel_unshuffle",  # known py38 fail
    "TestQuantizeFxOps.test_quantized_add_qat",  # known py38 fail
    "TestQuantizeFxOps.test_ref_pattern_multi_use",  # known py38 fail
    "TestQuantizeFxOps.test_pixel_unshuffle_module",  # known py38 fail
    "TestQuantizeFxOps.test_pixel_shuffle",  # known py38 fail
    "TestQuantizeFxOps.test_boolean_tensor",  # known py38 fail
    "TestQuantizeFxOps.test_bmm_int_reference",  # known py38 fail
    "TestQuantizeFxOps.test_rnn_cell",  # known py38 fail
    "TestQuantizeFxOps.test_pixel_shuffle_module",  # known py38 fail
    "TestQuantizeFxOps.test_fixed_qparams_ops_fp16",  # known py38 fail
    "TestQuantizeFxOps.test_quantized_mul_qat",  # known py38 fail
    "TestQuantizeFxOps.test_general_shape_ops",  # known py38 fail
    "TestQuantizeFxOps.test_add_relu_multiple_uses_of_relu",  # known py38 fail
    "TestQuantizeFxOps.test_rnn",  # known py38 fail
    "TestQuantizeFxOps.test_linear_dynamic_fp16",  # known py38 fail
    "TestQuantizeFxOps.test_linear_static_fp16",  # known py38 fail
    "TestAutogradFallback.test_no_grad_mode_warn",  # test_autograd_fallback.py
    "TestAutogradFallback.test_no_autograd_kernel_inplace_mode_warn",  # test_autograd_fallback.py
    "TestAutogradFallback.test_no_autograd_kernel_mode_warn",  # test_autograd_fallback.py
    "TestAutogradFallback.test_undefined_grads_mode_nothing",  # test_autograd_fallback.py
    "TestAutogradFallback.test_post_autograd_returns_mix_of_requires_grad_tensors_mode_warn",  # test_autograd_fallback.py
    "TestAutogradFallback.test_no_autograd_kernel_inplace_mode_nothing",  # test_autograd_fallback.py
    "TestAutogradFallback.test_undefined_inputs_outputs_mode_nothing",  # test_autograd_fallback.py
    "TestAutogradFallback.test_inplace_on_tensor_that_does_not_require_grad_mode_nothing",  # test_autograd_fallback.py
    "TestAutogradFallback.test_post_autograd_returns_mix_of_requires_grad_tensors_mode_nothing",  # test_autograd_fallback.py
    "TestAutogradFallback.test_supports_tensor_lists_mode_warn",  # test_autograd_fallback.py
    "TestAutogradFallback.test_post_autograd_returns_leaf_mode_nothing",  # test_autograd_fallback.py
    "TestAutogradFallback.test_inplace_on_tensor_that_does_not_require_grad_mode_warn",  # test_autograd_fallback.py
    "TestAutogradFallback.test_no_grad_mode_nothing",  # test_autograd_fallback.py
    "TestAutogradFallback.test_post_autograd_returns_leaf_mode_warn",  # test_autograd_fallback.py
    "TestAutogradFallback.test_undefined_grads_mode_warn",  # test_autograd_fallback.py
    "TestAutogradFallback.test_no_autograd_kernel_mode_nothing",  # test_autograd_fallback.py
    "TestAutogradFallback.test_undefined_inputs_outputs_mode_warn",  # test_autograd_fallback.py
    "TestAutogradFallback.test_supports_tensor_lists_mode_nothing",  # test_autograd_fallback.py
    "TestReductionsCPU.test_logcumsumexp_complex_cpu_complex128",  # test_reductions.py
    "TestReductionsCPU.test_logcumsumexp_complex_cpu_complex64",  # test_reductions.py
    "TestCustomOp.test_impl_on_existing_op_with_cpu_registration_key_CompositeExplicitAutograd",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpyViewCopyCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpyCatCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestPythonDispatcher.test_lstsq",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_split_cpu_float32",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_split_list_args_cpu_float32",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpyNonzeroCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_mul_cpu_float32",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpyMulCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpySortCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_out_op_cpu",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpyTakeCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpyCubeCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_cat_cpu_float32",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_native_batch_norm_cpu_float32",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpyNMSCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpySplitCopyWithIntCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_view_cpu_float32",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpySplitCopyCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "LoggingTests.test_logs_out",  # known py38 fail
    "LoggingTests.test_distributed_rank_logging",  # known py38 fail
    "LoggingTests.test_trace_call",  # known py311 fail
    "LoggingTests.test_trace_call_graph_break",  # known py311 fail
    "LoggingTests.test_trace_call_inline_call",  # known py311 fail
    "TestPythonBuiltinOP.test_stepped_tuple_slicing",  # known py38 fail
    "TestPythonBuiltinOP.test_advancedindex",  # known py38 fail
    "TestCustomOperators.test_dynamic_op_registry",  # known py38 fail
    "TestComplex.test_complex_constants_and_ops",  # known py38 fail
    "TestPythonBuiltinOP.test_index",  # known py38 fail
    "TestHash.test_hash_tuple_nested_unhashable_type",  # test_jit
    "TestPeephole.test_peephole_dict_getitem_no_optimization_missing_key",  # test_jit
    "TestMisc.test_joined_str",  # test_jit
    "TestNnapiBackend.test_to",  # test_jit
    "TestIsinstance.test_dict_tensor",  # test_jit
    "TestPythonBuiltinOP.test_adv_indexing_list",  # test_jit
    "TestUnion.test_union_redundant_arguments_are_skipped",  # test_jit
    "TestPythonBuiltinOP.test_inf",  # test_jit
    "TestSymbolicShapeAnalysis.test_register_function_error_checking",  # test_jit
    "TestPythonBuiltinOP.test_pow",  # test_jit
    "TestTyping.test_tuple_io",  # test_jit
    "TestPeephole.test_peephole_dict_len_no_optimization_unsupported_type",  # test_jit
    "TestRemoveMutation.test_lists_append",  # test_jit
    "TestSlice.test_slice_tensor",  # test_jit
    "TestMisc.test_str_refine_any",  # test_jit
    "TestNnapiBackend.test_prelu",  # test_jit
    "TestFreezing.test_freeze_module_with_fork2",  # test_jit
    "TestPeephole.test_peephole_dict_len_no_optimization_overlapping_keys",  # test_jit
    "TestPeephole.test_peephole_with_non_output_writes",  # test_jit
    "TestCustomOperators.test_script_graph_contains_custom_op",  # test_jit
    "TestTorchbind.test_torchbind_getattr",  # test_jit
    "TestRecursiveScript.test_inner_traced_module",  # test_jit
    "TestAliasAnalysis.test_recursive_calls",  # test_jit
    "TestNnapiBackend.test_pointwise_unary",  # test_jit
    "TestDeviceAnalysis.test_device_apply",  # test_jit
    "TestList.test_mutable_list_function_inline",  # test_jit
    "TestList.test_comprehensions_two_comps",  # test_jit
    "TestNnapiBackend.test_seblock_mul",  # test_jit
    "TestTorchbind.test_torchbind_return_instance",  # test_jit
    "TestRemoveMutation.test_if_output",  # test_jit
    "TestModels.test_time_sequence_prediction",  # test_jit
    "TestRemoveMutation.test_list_indexing_removal",  # test_jit
    "TestTypesAndAnnotation.test_types_as_values",  # test_jit
    "TestAwait.test_await_multiout_save",  # test_jit
    "TestHash.test_hash_device",  # test_jit
    "TestPeephole.test_peephole_dict_len",  # test_jit
    "TestSlice.test_slice_dynamic_index",  # test_jit
    "TestGenerator.test_default_generator",  # test_jit
    "TestMisc.test_parse_ir_single_element_tensor_negative",  # test_jit
    "TestTyping.test_list_unification",  # test_jit
    "TestList.test_del",  # test_jit
    "TestAwait.test_script",  # test_jit
    "TestNnapiBackend.test_avg_pool2d",  # test_jit
    "TestIsinstance.test_list_tensor_type_true",  # test_jit
    "TestDtypeAnalysis.test_conv_no_mixed_args",  # test_jit
    "TestBackends.test_errors",  # test_jit
    "TestMisc.test_parse_ir_annotate",  # test_jit
    "TestTorchbind.test_torchbind_pickle_serialization",  # test_jit
    "TestList.test_copy_list_immutable",  # test_jit
    "TestAsync.test_async_grad_guard_with_grad",  # test_jit
    "TestUnion.test_union_branching_does_not_autoinfer_undeclared_union",  # test_jit
    "TestNnapiBackend.test_slice",  # test_jit
    "TestWarn.test_warn",  # test_jit
    "TestRemoveMutation.test_special_mapped_op",  # test_jit
    "TestWarn.test_warn_multiple_calls_same_func_diff_stack",  # test_jit
    "TestSymbolicShapeAnalysis.test_convolution_backward",  # test_jit
    "TestUnion.test_union_type_refinement_statically_false",  # test_jit
    "TestTorchbind.test_default_args",  # test_jit
    "TestUpgraders.test_aten_full_other_variants",  # test_jit
    "TestScriptDict.test_reference_semantics",  # test_jit
    "TestUnion.test_union_does_not_replace_existing_annotated_type_union",  # test_jit
    "TestTyping.test_dict_invalid_annotations",  # test_jit
    "TestWith.test_with_no_grad",  # test_jit
    "TestUnion.test_union_branching_does_not_widen_existing_inferred_type",  # test_jit
    "TestTorchbind.test_torchbind_return_tuple",  # test_jit
    "TestTorchbind.test_staticmethod",  # test_jit
    "TestUnion.test_union_variable_can_be_reassigned",  # test_jit
    "TestTorchbind.test_torchbind_def_property_readwrite",  # test_jit
    "TestTorchbind.test_torchbind_attr_exception",  # test_jit
    "TestFunctionalToInplaceActivation.test_no_functional_to_inplace",  # test_jit
    "TestTorchbind.test_torchbind_class_attr_recursive",  # test_jit
    "TestBuiltins.test_del",  # test_jit
    "TestNnapiBackend.test_mean",  # test_jit
    "TestNnapiBackend.test_reshape",  # test_jit
    "TestFrozenOptimizations.test_collapse_adjacent_conversions",  # test_jit
    "TestTorchbind.test_torchbind_python_deepcopy",  # test_jit
    "TestPythonBindings.test_aliasdb",  # test_jit
    "TestParametrization.test_scriptable",  # test_jit
    "TestMKLDNNReinplacing.test_always_alive_values",  # test_jit
    "TestAsync.test_async_script_multi_waits",  # test_jit
    "TestTyping.test_dict_type_refinement_annotation_value_mismatch",  # test_jit
    "TestScriptList.test_reference_semantics",  # test_jit
    "TestPeephole.test_peephole_arith",  # test_jit
    "TestPythonBuiltinOP.test_gather",  # test_jit
    "TestNnapiBackend.test_upsample_nearest2d",  # test_jit
    "TestList.test_copy_list_mutable",  # test_jit
    "TestWarn.test_warn_only_once",  # test_jit
    "TestPythonBuiltinOP.test_str_to_float",  # test_jit
    "TestIsinstance.test_optional",  # test_jit
    "TestCustomOperators.test_calling_scripted_custom_op",  # test_jit
    "TestUnion.test_union_T_None_is_equivalent_to_optional_T",  # test_jit
    "TestSlice.test_slice_tensor_multidim_with_dots",  # test_jit
    "TestNnapiBackend.test_multi_output",  # test_jit
    "TestSymbolicShapeAnalysis.test_squeeze_dims",  # test_jit
    "TestPeephole.test_peephole_int",  # test_jit
    "TestUnion.test_unions_of_a_single_argument_vanish",  # test_jit
    "TestTorchbind.test_profiler_custom_op",  # test_jit
    "TestTorchbind.test_torchbind_class_attribute",  # test_jit
    "TestUnion.test_check_union_annotation",  # test_jit
    "TestTypesAndAnnotation.test_optional_no_element_type_annotation",  # test_jit
    "TestList.test_comprehension_iterable",  # test_jit
    "TestUpgraders.test_aten_test_serialization",  # test_jit
    "TestPythonBuiltinOP.test_mul",  # test_jit
    "TestAwait.test_nowait",  # test_jit
    "TestBuiltins.test_del_multiple_operands",  # test_jit
    "TestTypesAndAnnotation.test_bad_types",  # test_jit
    "TestSymbolicShapeAnalysis.test_cross_entropy_loss",  # test_jit
    "TestRemoveMutation.test_aten_inplace",  # test_jit
    "TestWarn.test_warn_only_once_in_loop_func",  # test_jit
    "TestDataclasses.test_use_unregistered_dataclass_raises",  # test_jit
    "TestTorchbind.test_torchbind_optional_explicit_attr",  # test_jit
    "TestTorchbind.test_torchbind_pass_wrong_type",  # test_jit
    "TestList.test_list_variance",  # test_jit
    "TestMisc.test_subexpression_Dict_int_Future",  # test_jit
    "TestMisc.test_future_isinstance",  # test_jit
    "TestPythonBuiltinOP.test_slice",  # test_jit
    "TestPeephole.test_short_circuit_optimization",  # test_jit
    "TestPeephole.test_peephole_slice_optimization_not_applied_list_modified",  # test_jit
    "TestTyping.test_namedtuple_good_error",  # test_jit
    "TestMisc.test_subexpression_List_Future",  # test_jit
    "TestDtypeAnalysis.test_combined",  # test_jit
    "TestFunctionalBlocks.test_subgraph_creation",  # test_jit
    "TestList.test_extend_list_mutable",  # test_jit
    "TestPythonBindings.test_cu_get_functions",  # test_jit
    "TestLogging.test_trace_numeric_counter",  # test_jit
    "TestBatchMM.test_batch_mm_side_prohibited_mutation_common_side",  # test_jit
    "TestPeephole.test_peephole_dynamic",  # test_jit
    "TestTorchbind.test_torchbind_def_property_getter_setter",  # test_jit
    "TestSymbolicShapeAnalysis.test_size_and_sizes",  # test_jit
    "TestAsync.test_async_script",  # test_jit
    "TestAsync.test_async_parsing",  # test_jit
    "TestAwait.test_await_func_arg",  # test_jit
    "TestTyping.test_dict_type_refinement_annotation_key_mismatch",  # test_jit
    "TestNnapiBackend.test_softmax",  # test_jit
    "TestDataclasses.test__post_init__",  # test_jit
    "TestPeephole.test_normalized_is_op",  # test_jit
    "TestMisc.test_broadcasting_list",  # test_jit
    "TestIsinstance.test_optional_no_contained_type",  # test_jit
    "TestUnion.test_union_argument_order_is_ignored",  # test_jit
    "TestUnion.test_union_argument_order_is_ignored_container",  # test_jit
    "TestAutodiffSubgraphSlicing.test_chunk_constant_script_ad",  # test_jit
    "TestBackends.test_save_load",  # test_jit
    "TestIsinstance.test_list_tensor",  # test_jit
    "TestComplex.test_tensor_attributes",  # test_jit
    "TestRemoveMutation.test_lists_insert",  # test_jit
    "TestNnapiBackend.test_qlinear",  # test_jit
    "TestNnapiBackend.test_quantize",  # test_jit
    "TestNnapiBackend.test_unsqueeze",  # test_jit
    "TestTorchbind.test_lambda_as_constructor",  # test_jit
    "TestTyping.test_dict_comprehension_with_type_annotation",  # test_jit
    "TestAtenPow.test_aten_pow_zero_negative_exponent",  # test_jit
    "TestUnion.test_union_as_dict_key",  # test_jit
    "TestTyping.test_optional_refinement",  # test_jit
    "TestPeephole.test_peephole_type_refinements",  # test_jit
    "TestSlice.test_slice_kwarg",  # test_jit
    "TestStringFormatting.test_string_interpolation_with_too_many_arguments",  # test_jit
    "TestTorchbind.test_torchbind_getstate",  # test_jit
    "TestTyping.test_dict_comprehension_scope",  # test_jit
    "TestRemoveMutation.test_if_output_fail",  # test_jit
    "TestMisc.test_legacy_tensor_constructor",  # test_jit
    "TestBatchMM.test_batch_mm_prohibited_mutation_multiple_adds",  # test_jit
    "TestSlice.test_slice_tensor_multidim",  # test_jit
    "TestPeephole.test_peephole_slice_two_empty_args",  # test_jit
    "TestTyping.test_namedtuple_py2",  # test_jit
    "TestUnion.test_union_type_refinement_statically_true",  # test_jit
    "TestRecursiveScript.test_script_function_attribute",  # test_jit
    "TestPeephole.test_peephole",  # test_jit
    "TestAwait.test_await_python",  # test_jit
    "TestPythonBuiltinOP.test_triple",  # test_jit
    "TestTorchbind.test_torchbind_take_as_arg",  # test_jit
    "TestNnapiBackend.test_qadd",  # test_jit
    "TestTypesAndAnnotation.test_pep585_type",  # test_jit
    "TestNnapiBackend.test_detach",  # test_jit
    "TestAsync.test_async_script_multi_forks",  # test_jit
    "TestPythonBindings.test_invalidation",  # test_jit
    "TestTyping.test_for_tuple_unpack",  # test_jit
    "TestTorchbind.test_torchbind_deepcopy",  # test_jit
    "TestTorchbind.test_torchbind_instantiate_missing_class",  # test_jit
    "TestSymbolicShapeAnalysis.test_if_propagation",  # test_jit
    "TestPeephole.test_normalized_rsub",  # test_jit
    "TestPythonIr.test_param_strides",  # test_jit
    "TestPythonIr.test_permute_inputs_binding",  # test_jit
    "TestPythonIr.test_python_ir_utils",  # test_jit
    "TestPythonIr.test_python_ir_utils_graph",  # test_jit
    "TestComplex.test_complex_list_sum",  # test_jit
    "TestUnion.test_union_redundant_arguments_are_skipped_optional",  # test_jit
    "TestNnapiBackend.test_conv2d",  # test_jit
    "TestDtypeAnalysis.test_unary",  # test_jit
    "TestPeephole.test_peephole_dict_len_no_optimization_keys_might_overlap",  # test_jit
    "TestIsinstance.test_dict_no_contained_type",  # test_jit
    "TestList.test_extend_list_immutable",  # test_jit
    "TestFrozenOptimizations.test_conv_add_folding",  # test_jit
    "TestGenerator.test_generator_arg",  # test_jit
    "TestTensorBuiltins.test_method_on_number",  # test_jit
    "TestUnion.test_union_optional_of_union_is_flattened",  # test_jit
    "TestUnion.test_union_type_refinement_tuple_rhs_union",  # test_jit
    "TestList.test_no_element_type_annotation",  # test_jit
    "TestParametrization.test_traceable",  # test_jit
    "TestSymbolicShapeAnalysis.test_shape_analysis",  # test_jit
    "TestScriptProfile.test_script",  # test_jit
    "TestSymbolicShapeAnalysis.test_write",  # test_jit
    "TestPeephole.test_peephole_slice_optimization_not_applied_non_const_args",  # test_jit
    "TestNnapiBackend.test_cat",  # test_jit
    "TestList.test_mutable_list_pop_empty",  # test_jit
    "TestMisc.test_subexpression_Optional",  # test_jit
    "TestUnion.test_union_does_not_replace_existing_annotated_type",  # test_jit
    "TestTorchbind.test_torchbind_return_instance_from_method",  # test_jit
    "TestTyping.test_opt_opt_refinement",  # test_jit
    "TestIsinstance.test_tuple_tensor",  # test_jit
    "TestUpgraders.test_populated_test_upgrader_graph",  # test_jit
    "TestList.test_slice_index",  # test_jit
    "TestTyping.test_tuple_assignments",  # test_jit
    "TestAsync.test_async_python",  # test_jit
    "TestBatchMM.test_batch_mm_prohibited_mutation",  # test_jit
    "TestFreezing.test_freeze_module_with_fork_calling_module_method",  # test_jit
    "TestUnion.test_unions_of_unions_are_flattened",  # test_jit
    "TestTypeSharing.test_script_function_attribute_different",  # test_jit
    "TestTorchbind.test_torchbind_lambda_method",  # test_jit
    "TestTypesAndAnnotation.test_unimported_type_resolution",  # test_jit
    "TestUnion.test_union_redundant_arguments_are_skipped_container",  # test_jit
    "TestPythonBindings.test_cu_create_function",  # test_jit
    "TestTorchbind.test_torchbind_tracing",  # test_jit
    "TestWarn.test_warn_once_per_func_in_loop",  # test_jit
    "TestBackendsWithCompiler.test_errors",  # test_jit
    "TestSaveLoadForOpVersion.test_versioned_div_tensor_inplace",  # test_jit
    "TestList.test_to_list",  # test_jit
    "TestUpgraders.test_populated_upgrader_graph",  # test_jit
    "TestWarn.test_warn_multiple_calls_multiple_warnings",  # test_jit
    "TestLogging.test_counter_aggregation",  # test_jit
    "TestTorchbind.test_torchbind_take_instance_as_method_arg",  # test_jit
    "TestComplex.test_complex_parse",  # test_jit
    "TestTorchbind.test_torchbind_save_load",  # test_jit
    "TestPeephole.test_integer_refinement",  # test_jit
    "TestBatchMM.test_batch_mm_prohibited_mutation_if_node",  # test_jit
    "TestHash.test_hash_tensor",  # test_jit
    "TestAsync.test_trace_fork_wait_inline",  # test_jit
    "TestTensorBuiltins.test_tensor_item",  # test_jit
    "TestList.test_list_keyword",  # test_jit
    "TestTypesAndAnnotation.test_ignore_with_types",  # test_jit
    "TestPeephole.test_peephole_slice_one_empty_arg",  # test_jit
    "TestAsync.test_async_script_nested",  # test_jit
    "TestNnapiBackend.test_flatten",  # test_jit
    "TestAsync.test_future_subtyping",  # test_jit
    "TestTorchbind.test_torchbind_no_init",  # test_jit
    "TestModels.test_vae_quantized",  # test_jit
    "TestSymbolicShapeAnalysis.test_shared_shape_graph",  # test_jit
    "TestNnapiBackend.test_dequantize",  # test_jit
    "TestPeephole.test_peephole_optional_refine",  # test_jit
    "TestTorchbind.test_torchbind",  # test_jit
    "TestAwait.test_await_out_of_interpreter",  # test_jit
    "TestNnapiBackend.test_conv2d_transpose",  # test_jit
    "TestNnapiBackend.test_max_pool2d",  # test_jit
    "TestPeephole.test_peephole_list_ops",  # test_jit
    "TestTyping.test_optional_conversion",  # test_jit
    "TestNnapiBackend.test_linear",  # test_jit
    "TestPythonBuiltinOP.test_add",  # test_jit
    "TestIsinstance.test_tuple_no_contained_type",  # test_jit
    "TestTyping.test_bool_list_io",  # test_jit
    "TestPeephole.test_peephole_dict_getitem_no_optimization_dict_modified",  # test_jit
    "TestNnapiBackend.test_compile_spec_santiy",  # test_jit
    "TestDtypeAnalysis.test_custom_rules",  # test_jit
    "TestPeephole.test_peephole_len_list",  # test_jit
    "TestTyping.test_dict_in_not_in",  # test_jit
    "TestUnion.test_union_redundant_arguments_are_skipped_subtyping",  # test_jit
    "TestTensorMethods.test_getitem",  # test_jit
    "TestPeephole.test_peephole_dict_getitem_no_optimization_overlapping_keys",  # test_jit
    "TestDataclasses.test_comparators",  # test_jit
    "TestTyping.test_tuple_specialization",  # test_jit
    "TestModels.test_snli_quantized",  # test_jit
    "TestGenerator.test_script",  # test_jit
    "TestAsync.test_async_script_error",  # test_jit
    "TestUnion.test_union_with_collections",  # test_jit
    "TestList.test_list_index_not_existing",  # test_jit
    "TestStringFormatting.test_string_interpolation_with_exponent_placeholder_and_string_variable",  # test_jit
    "TestStringFormatting.test_string_interpolation_with_too_few_arguments",  # test_jit
    "TestMisc.test_unsafe_hacked_twin",  # test_jit
    "TestPeephole.test_peephole_dict_getitem_no_optimization_get_input_arg",  # test_jit
    "TestTyping.test_singleton_tuple_unpack",  # test_jit
    "TestUnion.test_union_with_scalar_values",  # test_jit
    "TestAwait.test_jit_trace",  # test_jit
    "TestBackendsWithCompiler.test_execution",  # test_jit
    "TestPeephole.test_normalized_isnot_op",  # test_jit
    "TestTyping.test_list_type_refinement_annotation_element_mismatch",  # test_jit
    "TestTorchbind.test_torchbind_def_property_just_getter",  # test_jit
    "TestNnapiBackend.test_tensor_input",  # test_jit
    "TestPythonBindings.test_graph_iterator_keepalive",  # test_jit
    "TestUnion.test_union_subclasses_larger_union",  # test_jit
    "TestPeephole.test_peephole_dict_getitem_simple",  # test_jit
    "TestBackends.test_execution",  # test_jit
    "TestPeephole.test_peephole_with_writes",  # test_jit
    "TestRecursiveScript.test_script_basic",  # test_jit
    "TestScriptProfile.test_section",  # test_jit
    "TestPeephole.test_peephole_add_zero",  # test_jit
    "TestAsync.test_trace_fork_wait",  # test_jit
    "TestAliasAnalysis.test_nested_list_construct_not_wildcard",  # test_jit
    "TestList.test_mutable_list_remove_not_existing",  # test_jit
    "TestMisc.test_parse_ir_single_element_tensor_positive",  # test_jit
    "TestNnapiBackend.test_log_softmax",  # test_jit
    "TestOpDecompositions.test_registered_decomposition",  # test_jit
    "TestStringFormatting.test_string_interpolation_with_percent_in_string",  # test_jit
    "TestBatchMM.test_batch_mm_permitted_mutation",  # test_jit
    "TestTorchbind.test_torchbind_tracing_nested",  # test_jit
    "TestNnapiBackend.test_hardtanh",  # test_jit
    "TestBatchMM.test_batch_mm_no_mutation",  # test_jit
    "TestIsinstance.test_type_refinement",  # test_jit
    "TestPeephole.test_peephole_slice_all_three_args",  # test_jit
    "TestTyping.test_tuple_keyword",  # test_jit
    "TestOpDecompositions.test_op_decomposition",  # test_jit
    "TestBatchMM.test_batch_mm_side_permitted_mutation",  # test_jit
    "TestNnapiBackend.test_pointwise_binary_const",  # test_jit
    "TestTypeSharing.test_script_function_attribute_same",  # test_jit
    "TestTypesAndAnnotation.test_type_annotate_py3",  # test_jit
    "TestPeephole.test_peephole_dict_getitem_no_optimization_unsupported_type",  # test_jit
    "TestMisc.test_subexpression_Tuple_int_int_Future",  # test_jit
    "TestMisc.test_subexpression_Future_annotate",  # test_jit
    "TestStringFormatting.test_string_interpolation_with_char_placeholder_and_true_string_variable",  # test_jit
    "TestSlice.test_tuple_slicing",  # test_jit
    "TestAwait.test_await_isinstance",  # test_jit
    "TestNnapiBackend.test_adaptive_avg_pool2d",  # test_jit
    "TestIsinstance.test_list_no_contained_type",  # test_jit
    "TestPeephole.test_peephole_no_output_aliasing",  # test_jit
    "TestStringFormatting.test_string_interpolation_with_unknown_format_specifier",  # test_jit
    "TestPeephole.test_refine_integer_values",  # test_jit
    "TestStringFormatting.test_string_interpolation_with_digit_placeholder_and_string_variable",  # test_jit
    "TestPeephole.test_peephole_dict_getitem_no_optimization_keys_might_overlap",  # test_jit
    "TestPeephole.test_peephole_list_len",  # test_jit
    "TestMisc.test_list_literal_infer",  # test_jit
    "TestIgnorableArgs.test_add_out_ignorable_args",  # test_jit
    "TestTensorBuiltins.test_scalar_to_num_conversions",  # test_jit
    "TestWarn.test_warn_once_per_func",  # test_jit
    "TestAsync.test_async_grad_guard_no_grad",  # test_jit
    "TestPythonBuiltinOP.test_random",  # test_jit
    "TestSymbolicShapeAnalysis.test_stitching_concat",  # test_jit
    "TestMisc.test_if_returning_any",  # test_jit
    "TestBatchMM.test_batch_mm_side_prohibited_mutation_uncommon_side",  # test_jit
    "TestList.test_tensor_list_index_not_existing",  # test_jit
    "TestMisc.test_script_many_decorators",  # test_jit
    "TestUnion.test_union_does_not_replace_existing_annotated_type_empty_container",  # test_jit
    "TestNnapiBackend.test_pointwise_binary",  # test_jit
    "TestTypesAndAnnotation.test_tuple_no_element_type_annotation",  # test_jit
    "TestFrozenOptimizations.test_conv_bn_folding",  # test_jit.py
    "TestArgmax.test_combinations_data58",
    "TestArgmax.test_combinations_data61",
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_1_shape0_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_2_shape0_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_1_shape3_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_2_shape2_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_1_shape2_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_2_shape1_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_1_shape1_cpu",  # test_transformers.py
    "TestCustomOpTestingCPU.test_opcheck_fails_basic_cpu",  # test_custom_ops.py
}


# verify some invariants
for test in dynamo_expected_failures.union(dynamo_skips):
    if len(test.split(".")) != 2:
        raise AssertionError(f'Invalid test name: "{test}"')

intersection = dynamo_expected_failures.intersection(dynamo_skips)
if len(intersection) > 0:
    raise AssertionError(
        "there should be no overlap between dynamo_expected_failures "
        "and dynamo_skips, got " + str(intersection)
    )

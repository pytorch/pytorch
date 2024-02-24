# mypy: ignore-errors
import os
import sys


def find_test_dir():
    if sys.platform == "win32":
        return None
    main = sys.modules["__main__"]
    file = getattr(main, "__file__", None)
    if file is None:
        # Generated file do not have a module.__file__
        return None
    main_dir = os.path.dirname(os.path.abspath(file))
    components = ["/"]
    for c in main_dir.split(os.path.sep):
        components.append(c)
        if c == "test":
            break
    test_dir = os.path.join(*components)
    assert os.path.exists(test_dir)
    return test_dir


test_dir = find_test_dir()

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

if test_dir is None:
    dynamo_expected_failures = set()
else:
    failures_directory = os.path.join(test_dir, "dynamo_expected_failures")
    dynamo_expected_failures = set(os.listdir(failures_directory))

# see NOTE [dynamo_test_failures.py] for more details
dynamo_skips = {
    "TestMatmulOperator.test_matmul_raises",
    "TestMatmulOperator.test_exceptions",
    "TestMethods.test_searchsorted_complex",
    "TestMethods.test_round",
    "TestMethods.test_searchsorted_type_specific_2",
    "TestMethods.test_dot_out_mem_overlap",
    "TestMethods.test_partition_iterative",
    "TestMethods.test_matmul_out",
    "TestMethods.test_transpose",
    "TestMethods.test_searchsorted_with_invalid_sorter",
    "TestMethods.test_compress",
    "TestConstant.test_check_constant",  # known py311 fail
    "TestFFTShift.test_fft_n",  # known py311 fail
    "TestVstack.test_generator",  # known py311 fail
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_I_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_I_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_L_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_P_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_P_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_H_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_H_operation1",  # torch_np/numpy_tests/core/test_scalarmath
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_L_operation0",  # torch_np/numpy_tests/core/test_scalarmath
    "TestOpenMP_ParallelFor.test_one_thread",  # test_openmp
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
    "TestSWAUtils.test_averaged_model_all_devices_ema_True",  # flaky
    "TestSWAUtils.test_averaged_model_exponential_use_multi_avg_fn_True_use_buffers_False",  # flaky
    "TestSWAUtils.test_averaged_model_exponential_use_multi_avg_fn_True_use_buffers_True",  # flaky
    "TestNativeFunctions.test_intlist_error_with_overload",  # known py311 fail
    "TestMkldnnFusion.test_single_conv",  # known py311 fail
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
    "TestCheckpoint.test_checkpoint_trigger",  # known py38 fail
    "TestUnaryUfuncsCPU.test_sinc_cpu_float64",  # known py38 fail
    "TestUnaryUfuncsCPU.test_special_i0_i1_vs_scipy_cpu_float32",  # known py38 fail
    "TestUnaryUfuncsCPU.test_special_i0_i1_vs_scipy_cpu_float64",  # known py38 fail
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
    "TestSparseCPU.test_factory_type_inference_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_dense_mul_cpu_int16",  # known py38 fail
    "TestSparseCPU.test_print_uncoalesced_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_index_select_empty_and_non_contiguous_index_cpu_float64",  # known py38 fail
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
    "TestSparseCPU.test_sparse_add_coalesce_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_to_dense_hybrid_sparse_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_asin_arcsin_cpu_uint8",  # known py38 fail
    "TestSparseCPU.test_sparse_sum_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_spdiags_cpu_int16",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_sparse_cpu_float16",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_float64",  # known py38 fail
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
    "TestSparseCPU.test_index_select_cpu_complex128",  # known py38 fail
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
    "TestSparseCPU.test_sparse_add_coalesce_cpu_complex64",  # known py38 fail
    "TestSparseCPU.test_log1p_cpu_int8",  # known py38 fail
    "TestSparseCPU.test_permute_masked_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_sparse_cpu_complex64",  # known py38 fail
    "TestSparseAnyCPU.test_check_sparse_tensor_invariants_SparseCSC_cpu",  # known py38 fail
    "TestSparseCPU.test_div_rounding_mode_cpu_float32",  # known py38 fail
    "TestSparseCPU.test_any_cpu",  # known py38 fail
    "TestSparseMeta.test_basic",  # known py38 fail
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
    "TestSparseAnyCPU.test_check_sparse_tensor_invariants_SparseBSC_cpu",  # known py38 fail
    "TestSparseAnyCPU.test_gradcheck_mm_SparseCOO_masked_slow_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sparse_sparse_mul_cpu_uint8",  # known py38 fail
    "TestSparseCPU.test_sparse_sparse_mul_cpu_int64",  # known py38 fail
    "TestSparseCPU.test_select_no_type_promotion_cpu_int16",  # known py38 fail
    "TestSparseCPU.test_select_no_type_promotion_cpu_int8",  # known py38 fail
    "TestSparseCPU.test_zeros_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_sparse_mask_cpu_complex128",  # known py38 fail
    "TestSparseAnyCPU.test_gradcheck_mm_SparseCOO_sparse_fast_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_contig_cpu_complex128",  # known py38 fail
    "TestSparseCPU.test_to_dense_with_gradcheck_masked_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_coalesce_transpose_mm_cpu_float64",  # known py38 fail
    "TestSparseCPU.test_sum_cpu_int64",  # known py38 fail
    "TestReductionsCPU.test_histogramdd_cpu_float32",  # known py38 fail
    "TestReductionsCPU.test_tensor_compare_ops_empty_cpu",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_uint8",  # known py38 fail
    "TestReductionsCPU.test_tensor_reduce_ops_empty_cpu",  # known py38 fail
    "TestReductionsCPU.test_all_any_vs_numpy_cpu_bool",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_ldexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_lu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_hardshrink_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_tan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_resolve_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_new_full_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_cumsum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_polygamma_special_polygamma_n_0_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_clamp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_pinverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_meshgrid_variadic_tensors_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_aminmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_dropout_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_ifftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_adaptive_avg_pool2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_half_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_reciprocal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_topk_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_conv3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_var_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_transpose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nan_to_num_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_svd_lowrank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive___getitem___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_expm1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_lt_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_min_reduction_no_dim_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_scatter_reduce_sum_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_avg_pool3d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_pad_circular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pixel_unshuffle_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpySplitCopyCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_sin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_log2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_digamma_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_sort_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_instance_norm_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_lu_unpack_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_sin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pad_reflect_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_isposinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_std_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_signal_windows_hann_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_var_unbiased_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_baddbmm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_vecdot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_conv_transpose2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_float_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive___rsub___cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nan_to_num_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_sign_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_fft_ifftn_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_narrow_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logit_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_tensordot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_logsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_max_pool2d_with_indices_backward_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_amin_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_clamp_max_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_squeeze_multiple_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_bilinear_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_chalf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cauchy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_softmin_with_dtype_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_new_empty_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_movedim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_signal_windows_exponential_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_clone_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_jiterator_binary_return_by_ref_cpu_float32",  # known py38 fail  # noqa: B950
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
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_max_reduction_no_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_NumpySortCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_tensor_split_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_median_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_isreal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_ifftn_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_bessel_y0_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_hypot_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_max_pool1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_atan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_conv1d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_lu_factor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_new_ones_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_erfc_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_ndtri_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_interpolate_bilinear_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_masked_logsumexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_diagonal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_float_power_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_interpolate_area_cpu_float32",  # known py38 fail  # noqa: B950
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
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_gradient_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_i0_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_chebyshev_polynomial_w_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_round_decimals_3_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_addcmul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_jiterator_2inputs_2outputs_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_logical_xor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_reciprocal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_i1e_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_shifted_chebyshev_polynomial_t_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_ctc_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_lu_factor_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_sigmoid_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_binary_cross_entropy_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_erfinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_triangular_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_qr_cpu_float32",  # known py38 fail
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
    "TestTEFuserDynamic.test_matmul",  # known py38 fail
    "TestTEFuserStatic.test_unary_ops",  # known py311 fail
    "TestTEFuserDynamic.test_unary_ops",  # known py311 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_softmin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_masked_fill_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_broadcast_tensors_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_heaviside_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_atan_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_lstsq_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_slice_scatter_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_MapControlflowOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_shifted_chebyshev_polynomial_v_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_fft_rfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nn_functional_feature_alpha_dropout_with_train_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_nll_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_neg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_scatter_reduce_amin_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_stack_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_normal_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_conv1d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_rot90_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_normalize_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_log_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_special_hermite_polynomial_he_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_polygamma_polygamma_n_2_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_ldl_factor_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_matrix_exp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_pow_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_interpolate_nearest_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_linalg_lu_solve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_atleast_3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cumsum_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_scatter_reduce_prod_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_matrix_exp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpyCatCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_nonzero_static_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_min_reduction_no_dim_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_max_pool2d_with_indices_backward_cpu_float32",  # known py38 fail  # noqa: B950
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
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_std_mean_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_uniform_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_interpolate_area_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_polygamma_polygamma_n_2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_nn_functional_conv2d_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_mvlgamma_mvlgamma_p_3_cpu_float32",  # known py38 fail  # noqa: B950
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
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_special_i0e_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_eigh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_narrow_copy_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_log_softmax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_mse_loss_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_var_unbiased_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_empty_like_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_addcmul_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_atleast_3d_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_triu_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_amax_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_arange_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_special_log_ndtr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_full_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive__segment_reduce_offsets_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_pinverse_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_special_shifted_chebyshev_polynomial_t_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_div_trunc_rounding_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_linear_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_fft_irfft2_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_masked_argmin_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_flipud_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_mvlgamma_mvlgamma_p_5_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_bartlett_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_tensorsolve_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_NumpyMulCustomOp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_special_scaled_modified_bessel_k1_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_tensorinv_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_native_batch_norm_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_narrow_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive__softmax_backward_data_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_isclose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_igammac_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_signal_windows_cosine_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_softplus_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nonzero_static_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_unsafe_chunk_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_split_list_args_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_fmod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cos_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_masked_logaddexp_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_maximum_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_rad2deg_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_flatten_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive___rdiv___cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_linalg_solve_ex_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_isinf_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_fft_rfftn_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_asinh_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_max_reduction_no_dim_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_cumprod_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_linalg_det_singular_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_lstsq_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_resize__cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_masked_mean_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_svd_lowrank_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_nn_functional_pixel_shuffle_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_linalg_qr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_nn_functional_hinge_embedding_loss_cpu_float32",  # known py38 fail  # noqa: B950
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_qr_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_expand_as_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_igammac_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_allclose_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_mvlgamma_mvlgamma_p_1_cpu_float32",  # known py38 fail
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_lu_solve_cpu_float32",  # known py38 fail
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
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_nn_functional_group_norm_cpu_float32",  # known py38 fail
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
    "TestBasicsCPU.test_invalid_sparse_csr_values_cpu",  # known py38 fail
    "TestBasicsCPU.test_invalid_sparse_coo_values_cpu",  # known py38 fail
    "TestBinaryUfuncsCPU.test_long_tensor_pow_floats_cpu",  # known py38 fail
    "TestBinaryUfuncsCPU.test_add_cpu",  # known py38 fail
    "TestBinaryUfuncsCPU.test_int_tensor_pow_neg_ints_cpu",  # known py38 fail
    "TestBinaryUfuncsCPU.test_shift_limits_cpu_uint8",  # known py38 fail
    "TestFXExperimental.test_optimize_for_inference_cpu",  # known py38 fail
    "TestForeachCPU.test_add_scalar_with_empty_list_and_empty_tensor_cpu_int32",  # known py38 fail
    "TestForeachCPU.test_add_scalar_with_empty_list_and_empty_tensor_cpu_int64",  # known py38 fail
    "TestForeachCPU.test_add_scalar_with_empty_list_and_empty_tensor_cpu_int16",  # known py38 fail
    "TestForeachCPU.test_add_scalar_with_empty_list_and_empty_tensor_cpu_int8",  # known py38 fail
    "TestForeachCPU.test_add_scalar_with_empty_list_and_empty_tensor_cpu_uint8",  # known py38 fail
    "TestProfiler.test_profiler_metadata",
    "TestSerializeCustomClass.test_custom_class",
    "TestTensorExprPyBind.test_kernel_with_custom_lowering",
    "TestFX.test_assert",  # known py38 fail
    "TestFX.test_annotations_empty_tuple",  # known py38 fail
    "TestLazyDynamicOps.test_nonzero_dynamic",  # known py311 fail
    "TestLazyTensor.test_view_mark_step_preserved",  # known py311 fail
    "TestRepackage.test_repackage_import_indirectly_via_parent_module",  # known py311 fail
    "TestPackageScript.test_load_shared_tensors_repackaged",  # known py311 fail
    "TestFXExperimental.test_optimize_for_inference_cpu_torchvision",
    "TestProfilerTree.test_profiler_experimental_tree",  # known py311 fail
    "TestExperiment.test_mark_strict_with_container_type",  # known py311 fail
    "TestAutograd.test_hook_closure_cycle_use_custom_function_True_use_tensor_hook_True",  # known py311 fail
    "TestReductionsCPU.test_logcumsumexp_complex_cpu_complex128",  # test_reductions.py
    "TestReductionsCPU.test_logcumsumexp_complex_cpu_complex64",  # test_reductions.py
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpyViewCopyCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpyCatCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_split_cpu_float32",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_split_list_args_cpu_float32",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpyNonzeroCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_mul_cpu_float32",  # known py38 fail
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpyMulCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
    "TestWrapperSubclassAliasingCPU.test_wrapper_subclass_aliasing_custom_NumpySortCustomOp_cpu_float32",  # known py38 fail  # noqa: B950
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
    "TestNnapiBackend.test_to",  # test_jit
    "TestNnapiBackend.test_prelu",  # test_jit
    "TestFreezing.test_freeze_module_with_fork2",  # test_jit
    "TestTorchbind.test_torchbind_getattr",  # test_jit
    "TestRecursiveScript.test_inner_traced_module",  # test_jit
    "TestNnapiBackend.test_pointwise_unary",  # test_jit
    "TestNnapiBackend.test_seblock_mul",  # test_jit
    "TestTorchbind.test_torchbind_return_instance",  # test_jit
    "TestModels.test_time_sequence_prediction",  # test_jit
    "TestMisc.test_parse_ir_single_element_tensor_negative",  # test_jit
    "TestNnapiBackend.test_avg_pool2d",  # test_jit
    "TestMisc.test_parse_ir_annotate",  # test_jit
    "TestTorchbind.test_torchbind_pickle_serialization",  # test_jit
    "TestNnapiBackend.test_slice",  # test_jit
    "TestSymbolicShapeAnalysis.test_convolution_backward",  # test_jit
    "TestTorchbind.test_default_args",  # test_jit
    "TestTorchbind.test_torchbind_return_tuple",  # test_jit
    "TestTorchbind.test_staticmethod",  # test_jit
    "TestTorchbind.test_torchbind_def_property_readwrite",  # test_jit
    "TestTorchbind.test_torchbind_attr_exception",  # test_jit
    "TestTorchbind.test_torchbind_class_attr_recursive",  # test_jit
    "TestNnapiBackend.test_mean",  # test_jit
    "TestNnapiBackend.test_reshape",  # test_jit
    "TestFrozenOptimizations.test_collapse_adjacent_conversions",  # test_jit
    "TestTorchbind.test_torchbind_python_deepcopy",  # test_jit
    "TestParametrization.test_scriptable",  # test_jit
    "TestMKLDNNReinplacing.test_always_alive_values",  # test_jit
    "TestNnapiBackend.test_upsample_nearest2d",  # test_jit
    "TestNnapiBackend.test_multi_output",  # test_jit
    "TestPeephole.test_peephole_int",  # test_jit
    "TestTorchbind.test_profiler_custom_op",  # test_jit
    "TestTorchbind.test_torchbind_class_attribute",  # test_jit
    "TestList.test_comprehension_iterable",  # test_jit
    "TestTorchbind.test_torchbind_optional_explicit_attr",  # test_jit
    "TestTorchbind.test_torchbind_pass_wrong_type",  # test_jit
    "TestLogging.test_trace_numeric_counter",  # test_jit
    "TestTorchbind.test_torchbind_def_property_getter_setter",  # test_jit
    "TestNnapiBackend.test_softmax",  # test_jit
    "TestMisc.test_broadcasting_list",  # test_jit
    "TestBackends.test_save_load",  # test_jit
    "TestNnapiBackend.test_qlinear",  # test_jit
    "TestNnapiBackend.test_quantize",  # test_jit
    "TestNnapiBackend.test_unsqueeze",  # test_jit
    "TestTorchbind.test_lambda_as_constructor",  # test_jit
    "TestTorchbind.test_torchbind_getstate",  # test_jit
    "TestAwait.test_await_python",  # test_jit
    "TestTorchbind.test_torchbind_take_as_arg",  # test_jit
    "TestNnapiBackend.test_qadd",  # test_jit
    "TestTypesAndAnnotation.test_pep585_type",  # test_jit
    "TestNnapiBackend.test_detach",  # test_jit
    "TestTorchbind.test_torchbind_deepcopy",  # test_jit
    "TestTorchbind.test_torchbind_instantiate_missing_class",  # test_jit
    "TestNnapiBackend.test_conv2d",  # test_jit
    "TestDtypeAnalysis.test_unary",  # test_jit
    "TestFrozenOptimizations.test_conv_add_folding",  # test_jit
    "TestParametrization.test_traceable",  # test_jit
    "TestNnapiBackend.test_cat",  # test_jit
    "TestTorchbind.test_torchbind_return_instance_from_method",  # test_jit
    "TestFreezing.test_freeze_module_with_fork_calling_module_method",  # test_jit
    "TestTorchbind.test_torchbind_lambda_method",  # test_jit
    "TestPythonBindings.test_cu_create_function",  # test_jit
    "TestTorchbind.test_torchbind_tracing",  # test_jit
    "TestSaveLoadForOpVersion.test_versioned_div_tensor_inplace",  # test_jit
    "TestTorchbind.test_torchbind_take_instance_as_method_arg",  # test_jit
    "TestTorchbind.test_torchbind_save_load",  # test_jit
    "TestNnapiBackend.test_flatten",  # test_jit
    "TestTorchbind.test_torchbind_no_init",  # test_jit
    "TestModels.test_vae_quantized",  # test_jit
    "TestNnapiBackend.test_dequantize",  # test_jit
    "TestPeephole.test_peephole_optional_refine",  # test_jit
    "TestTorchbind.test_torchbind",  # test_jit
    "TestNnapiBackend.test_conv2d_transpose",  # test_jit
    "TestNnapiBackend.test_max_pool2d",  # test_jit
    "TestTyping.test_optional_conversion",  # test_jit
    "TestNnapiBackend.test_linear",  # test_jit
    "TestNnapiBackend.test_compile_spec_santiy",  # test_jit
    "TestDtypeAnalysis.test_custom_rules",  # test_jit
    "TestModels.test_snli_quantized",  # test_jit
    "TestBackendsWithCompiler.test_execution",  # test_jit
    "TestTorchbind.test_torchbind_def_property_just_getter",  # test_jit
    "TestNnapiBackend.test_tensor_input",  # test_jit
    "TestBackends.test_execution",  # test_jit
    "TestMisc.test_parse_ir_single_element_tensor_positive",  # test_jit
    "TestNnapiBackend.test_log_softmax",  # test_jit
    "TestTorchbind.test_torchbind_tracing_nested",  # test_jit
    "TestNnapiBackend.test_hardtanh",  # test_jit
    "TestNnapiBackend.test_pointwise_binary_const",  # test_jit
    "TestSlice.test_tuple_slicing",  # test_jit
    "TestTensorBuiltins.test_scalar_to_num_conversions",  # test_jit
    "TestNnapiBackend.test_pointwise_binary",  # test_jit
    "TestLazyTensor.test_tensor_ctr",  # lazy/test_ts_opinfo
    "TestFrozenOptimizations.test_conv_bn_folding",  # test_jit.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_1_shape0_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_2_shape0_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_1_shape3_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_2_shape2_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_1_shape2_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_2_shape1_cpu",  # test_transformers.py
    "TestAttnBiasCPU.test_causal_variants_compile_causal_variant_1_shape1_cpu",  # test_transformers.py
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_Q_operation0",
    "TestScalarOpsMisc.test_scalar_integer_operation_divbyzero_dtype_Q_operation1",
    "TestArgmax.test_combinations_data61",  # torch_np/test_ndarray_methods.py
    "TestArgmax.test_combinations_data58",  # torch_np/test_ndarray_methods.py
    "TestPythonDispatch.test_list_ret",  # test_python_dispatch.py
    "TestCustomOpTestingCPU.test_opcheck_fails_basic_cpu",  # test_custom_ops.py
    "TestSaveLoadForOpVersion.test_versioned_div_tensor_out",  # test_jit.py
    "TestAutograd.test_post_accumulate_grad_hook_gets_cleaned_up",  # test_autograd
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

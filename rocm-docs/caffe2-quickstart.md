# rocm-caffe2 Quickstart Guide


## Running Core Tests
Before running the tests, make sure that the required environment variables are set:

```
export LD_LIBRARY_PATH=/pytorch/build_caffe2/lib:$LD_LIBRARY_PATH

```

Next, Navigate to /pytorch/caffe2_build/bin and run the binaries corresponding to the tests. The test binaries are:

* backend_cutting_test
* basic_test
* batch_matmul_op_hip_test
* batch_matmul_op_test
* binary_match_test
* blob_hip_test
* blob_test
* boolean_unmask_ops_test
* c10_utils_cpu_test
* c10_utils_gpu_test
* c10_utils_hip_test
* cast_test
* common_subexpression_elimination_test
* common_test
* context_hip_test
* context_test
* conv_to_nnpack_transform_test
* conv_transpose_op_mobile_test
* converter_nomigraph_test
* cpuid_test
* depthwise3x3_conv_op_test
* device_test
* dispatch_test
* dominator_tree_test
* elementwise_op_test
* event_hip_test
* event_test
* fatal_signal_asan_no_sig_test
* fixed_divisor_test
* generate_proposals_op_test
* generate_proposals_op_util_boxes_test
* generate_proposals_op_util_nms_test
* graph_test
* init_test
* logging_test
* match_test
* math_blas_hip_test
* math_hip_test
* math_test
* mobile_test
* module_test
* net_async_tracing_test
* net_test
* nnpack_test
* observer_test
* operator_fallback_hip_test
* operator_hip_test
* operator_schema_test
* operator_test
* parallel_net_test
* pattern_net_transform_test
* predictor_test
* proto_utils_test
* registry_test
* reshape_op_hip_test
* simple_queue_test
* smart_tensor_printer_test
* ssa_test
* stats_test
* string_ops_test
* tarjans_test
* text_file_reader_utils_test
* time_observer_test
* timer_test
* transform_test
* typeid_test
* utility_ops_hip_test
* utility_ops_test
* workspace_test



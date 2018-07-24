# rocm-caffe2 Quickstart Guide


## Running Core Tests
Before running the tests, make sure that the required environment variables are set:

```
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

```

Next, Navigate to /pytorch/caffe2_build and run the binaries corresponding to the tests. The test binaries are:

* blob_hip_test
* blob_test
* boolean_unmask_ops_test
* common_subexpression_elimination_test
* common_test
* context_hip_test
* context_test
* conv_to_nnpack_transform_test
* conv_transpose_op_mobile_test
* cpuid_test
* elementwise_op_hip_test
* elementwise_op_test
* event_hip_test
* event_test
* fatal_signal_asan_no_sig_test
* fixed_divisor_test
* fully_connected_op_hip_test
* fully_connected_op_test
* graph_test
* init_test
* logging_test
* math_hip_test
* math_test
* module_test
* mpi_test
* net_test
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
* stats_test
* string_ops_test
* text_file_reader_utils_test
* timer_test
* transform_test
* typeid_test
* utility_ops_hip_test
* utility_ops_test
* workspace_test


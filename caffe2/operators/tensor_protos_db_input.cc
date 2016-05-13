#include "caffe2/operators/tensor_protos_db_input.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(TensorProtosDBInput, TensorProtosDBInput<CPUContext>);
OPERATOR_SCHEMA(TensorProtosDBInput).NumInputs(1).NumOutputs(1, INT_MAX);
NO_GRADIENT(TensorProtosDBInput);
}  // namespace
}  // namespace caffe2

#include "caffe2/experiments/operators/fully_connected_op_sparse.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(FC_Sparse, FullyConnectedOp_SPARSE<float, CPUContext>);

OPERATOR_SCHEMA(FC_Sparse).NumInputs(5).NumOutputs(1);
}  // namespace
}  // namespace caffe2

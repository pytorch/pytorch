#include "cc_bmm_bg_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    ConcatBatchMatMulBatchGatherOp,
    ConcatBatchMatMulBatchGatherOp<CPUContext>);

OPERATOR_SCHEMA(ConcatBatchMatMulBatchGatherOp)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1);

} // namespace caffe2

#include "cc_bmm_bg_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    ConcatBatchMatMulBatchGatherOp,
    ConcatBatchMatMulBatchGatherOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ConcatBatchMatMulBatchGatherOp)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1);

} // namespace caffe2

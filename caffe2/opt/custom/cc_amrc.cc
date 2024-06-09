#include "caffe2/opt/custom/cc_amrc.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    ConcatAddMulReplaceNaNClip,
    ConcatAddMulReplaceNaNClipOp<CPUContext>);

OPERATOR_SCHEMA(ConcatAddMulReplaceNaNClip).NumInputs(3, INT_MAX).NumOutputs(2);

} // namespace caffe2

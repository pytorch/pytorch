#include "caffe2/quantization/server/group_norm_dnnlowp_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    GroupNorm,
    DNNLOWP,
    GroupNormDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8GroupNorm,
    DNNLOWP,
    GroupNormDNNLowPOp<uint8_t>);

OPERATOR_SCHEMA(Int8GroupNorm).NumInputs(3).NumOutputs({1, 3});

} // namespace caffe2

#include "caffe2/quantization/server/relu_dnnlowp_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR_WITH_ENGINE(Relu, DNNLOWP, ReluDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(Relu, DNNLOWP_16, ReluDNNLowPOp<uint16_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(Int8Relu, DNNLOWP, ReluDNNLowPOp<uint8_t>);

} // namespace caffe2

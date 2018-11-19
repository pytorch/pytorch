#include "utility_dnnlowp_ops.h"

#include "caffe2/utils/cpuid.h"

namespace caffe2 {

OPERATOR_SCHEMA(SumRelu)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .InputsCanCrossDevices()
    .IdenticalTypeAndShapeOfInput(0)
    .Input(0, "data_0", "First of the input tensors. Can be inplace.")
    .Output(0, "sum", "Output tensor. Same dimension as inputs.");

REGISTER_CPU_OPERATOR_WITH_ENGINE(Sum, DNNLOWP, SumDNNLowPOp<uint8_t, false>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SumRelu,
    DNNLOWP,
    SumDNNLowPOp<uint8_t, true>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8Sum,
    DNNLOWP,
    SumDNNLowPOp<uint8_t, false>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8SumRelu,
    DNNLOWP,
    SumDNNLowPOp<uint8_t, true>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Sum,
    DNNLOWP_16,
    SumDNNLowPOp<uint16_t, false>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SumRelu,
    DNNLOWP_16,
    SumDNNLowPOp<uint16_t, true>);

} // namespace caffe2

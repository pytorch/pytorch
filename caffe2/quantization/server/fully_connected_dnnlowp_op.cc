#include "fully_connected_dnnlowp_op.h"

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif

#include "caffe2/core/flags.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/fc_inference.h"
#include "caffe2/utils/cpuid.h"

C10_DEFINE_bool(
    caffe2_dnnlowp_enforce_default_operators,
    false,
    "When true, enforce to use the default Caffe2 operators inside DNNLOWP"
    "instead of using its own implementation that uses AVX2 instructions"
    "(currently only honored by FC)");

namespace caffe2 {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    DNNLOWP,
    FullyConnectedDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    DNNLOWP_16,
    FullyConnectedDNNLowPOp<uint16_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8FC,
    DNNLOWP,
    FullyConnectedDNNLowPOp<uint8_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    DNNLOWP_ROWWISE,
    FullyConnectedDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    DNNLOWP_ROWWISE_16,
    FullyConnectedDNNLowPOp<uint16_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8FC,
    DNNLOWP_ROWWISE,
    FullyConnectedDNNLowPOp<uint8_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8FCRelu,
    DNNLOWP,
    FullyConnectedDNNLowPOp<uint8_t, true>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8FCRelu,
    DNNLOWP_ROWWISE,
    FullyConnectedDNNLowPOp<uint8_t, true>);

using namespace std::placeholders;
OPERATOR_SCHEMA(Int8FCRelu)
    .NumInputs(3)
    .NumOutputs(1)
    .TensorInferenceFunction(std::bind(FCShapeInference, _1, _2, false))
    .CostInferenceFunction(std::bind(CostInferenceForFC, _1, _2, false));

} // namespace caffe2

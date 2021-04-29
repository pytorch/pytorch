#include "caffe2/operators/quantized/int8_dequantize_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Int8Dequantize, int8::Int8DequantizeOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Int8Dequantize)
    .IdenticalTypeAndShape()
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "qX", "Int8 Tensor qX.")
    .Output(0, "Y", "FP32 Tensor that represents mapped real value of qX.");

} // namespace caffe2

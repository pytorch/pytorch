#include "caffe2/contrib/fakelowp/int8_dequantize_op_nnpi.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8DequantizeNNPI, int8::Int8DequantizeNNPIOp);

OPERATOR_SCHEMA(Int8DequantizeNNPI)
    .IdenticalTypeAndShape()
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "qX", "Int8 Tensor qX.")
    .Output(0, "Y", "FP32 Tensor that represents mapped real value of qX.");

} // namespace caffe2

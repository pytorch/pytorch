#include "caffe2/operators/quantized/int8_quantize_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8Quantize, int8::Int8QuantizeOp);

OPERATOR_SCHEMA(Int8Quantize)
    .IdenticalTypeAndShape()
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "X", "FP32 Tensor X.")
    .Output(0, "Y", "Int8 Tensor qX representing X with linear quantization.");

} // namespace caffe2

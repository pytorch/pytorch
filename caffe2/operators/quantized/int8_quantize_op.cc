#include "caffe2/operators/quantized/int8_quantize_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8Quantize, int8::Int8QuantizeOp);

OPERATOR_SCHEMA(Int8Quantize)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .NumInputs(1, 2)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      out.emplace_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .Input(0, "X", "FP32 Tensor X.")
    .Input(
        1,
        "Qparam",
        "Optional Qparam blob that contains quant param computed on activation histogram data"
        "Will overwrite Y_scale and Y_zero_point argument if specified")
    .Output(0, "Y", "Int8 Tensor qX representing X with linear quantization.");

} // namespace caffe2

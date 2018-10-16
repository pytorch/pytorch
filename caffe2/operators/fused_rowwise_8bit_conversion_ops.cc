#include "caffe2/operators/fused_rowwise_8bit_conversion_ops.h"
#include <fp16.h>
#include "c10/util/Registry.h"

namespace caffe2 {

namespace {
void convertfp32fp32(float* dst, const float* src, size_t N) {
  memcpy(dst, src, sizeof(float) * N);
}

void convertfp16fp32(float* dst, const at::Half* src, size_t N) {
  for (size_t i = 0; i < N; i++) {
    dst[i] = fp16_ieee_to_fp32_value(src[i].x);
  }
}

void convertfp32fp16(at::Half* dst, const float* src, size_t N) {
  for (size_t i = 0; i < N; i++) {
    uint16_t out = fp16_ieee_from_fp32_value(src[i]);
    memcpy(dst + i, &out, sizeof(uint16_t));
  }
}
} // namespace

REGISTER_CPU_OPERATOR(
    FloatToFused8BitRowwiseQuantized,
    FloatToFused8BitRowwiseQuantizedOp<float, convertfp32fp32, CPUContext>);
OPERATOR_SCHEMA(FloatToFused8BitRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) + 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 8-bit row-wise quantization by determining the range
(maximum - minimum) and offset (minimum value) of each row in the input
matrix, and then scaling each element to an 8-bit number between 0 and
255. To later de-quantize values, the scale (range / 255) and offset
(bias) are stored alongside the data. More precisely, the first 4 bytes
of each row in the output matrix are a 32-bit float storing the scale,
the next 4 bytes store the bias as a 32-bit float, and all remaining
bytes in the row encode single quantized values.)
)DOC")
    .Input(0, "input", "Float32 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
NO_GRADIENT(FloatToFused8BitRowwiseQuantized);

REGISTER_CPU_OPERATOR(
    HalfFloatToFused8BitRowwiseQuantized,
    FloatToFused8BitRowwiseQuantizedOp<at::Half, convertfp16fp32, CPUContext>);
OPERATOR_SCHEMA(HalfFloatToFused8BitRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) + 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 8-bit row-wise quantization by determining the range
(maximum - minimum) and offset (minimum value) of each row in the input
matrix, and then scaling each element to an 8-bit number between 0 and
255. To later de-quantize values, the scale (range / 255) and offset
(bias) are stored alongside the data. More precisely, the first 4 bytes
of each row in the output matrix are a 32-bit float storing the scale,
the next 4 bytes store the bias as a 32-bit float, and all remaining
bytes in the row encode single quantized values.)
)DOC")
    .Input(0, "input", "Float16 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
NO_GRADIENT(HalfFloatToFused8BitRowwiseQuantized);

REGISTER_CPU_OPERATOR(
    Fused8BitRowwiseQuantizedToFloat,
    Fused8BitRowwiseQuantizedToFloatOp<float, convertfp32fp32, CPUContext>);
OPERATOR_SCHEMA(Fused8BitRowwiseQuantizedToFloat)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) - 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
De-quantizes the result of the
FloatToFused8BitRowwiseQuantized operator. The input is expected to
encode the scale as a 32-bit float in the second to the last 4 bytes of each
row, followed by the bias as a 32-bit float in the next 4 bytes, and the
quantized values in the preceding bytes of the row. The output is a
matrix containing only the values, but de-quantized. De-quantization is
performed by multiplying each value by its row's scale and bias
parameters. The de-quantized values will thus not be exactly equal to
the original, un-quantized floating point values.
)DOC")
    .Input(
        0,
        "scale_bias_quantized_input",
        "Fused scale, bias and quantized data")
    .Output(0, "float_output", "Float32 data");
NO_GRADIENT(Fused8BitRowwiseQuantizedToFloat);

REGISTER_CPU_OPERATOR(
    Fused8BitRowwiseQuantizedToHalfFloat,
    Fused8BitRowwiseQuantizedToFloatOp<at::Half, convertfp32fp16, CPUContext>);
OPERATOR_SCHEMA(Fused8BitRowwiseQuantizedToHalfFloat)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) - 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT16);
      return out;
    })
    .SetDoc(R"DOC(
De-quantizes the result of the
HalfFloatToFused8BitRowwiseQuantized operator. The input is expected to
encode the scale as a 32-bit float in the second to the last 4 bytes of each
row, followed by the bias as a 32-bit float in the next 4 bytes, and the
quantized values in the preceding bytes of the row. The output is a
matrix containing only the values, but de-quantized. De-quantization is
performed by multiplying each value by its row's scale and bias
parameters. The de-quantized values will thus not be exactly equal to
the original, un-quantized floating point values.
)DOC")
    .Input(
        0,
        "scale_bias_quantized_input",
        "Fused scale, bias and quantized data")
    .Output(0, "float16_output", "Float16 data");
NO_GRADIENT(Fused8BitRowwiseQuantizedToHalfFloat);

} // namespace caffe2

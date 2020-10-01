#include "caffe2/operators/fused_rowwise_nbit_conversion_ops.h"
#include "c10/util/Registry.h"

namespace caffe2 {

using std::uint16_t;
using std::vector;

namespace internal {
void convertfp32fp16(at::Half* dst, const float* src, size_t N) {
  for (size_t i = 0; i < N; i++) {
    dst[i] = src[i];
  }
}

} // namespace internal

REGISTER_CPU_OPERATOR(
    FloatToFused4BitRowwiseQuantized,
    FloatToFusedNBitRowwiseQuantizedOp<4, float, internal::convertfp32fp32>);
OPERATOR_SCHEMA(FloatToFused4BitRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      // divide over 2 and round up, add 4 for the extra scale and bias
      X.set_dims(
          X.dims().size() - 1,
          (X.dims(X.dims().size() - 1) + 1) / 2 + 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 4-bit row-wise quantization by determining the range
(maximum - minimum) and offset (minimum value) of each row in the input
matrix, and then scaling each element to an 4-bit number between 0 and
15. To later de-quantize values, the scale (range / 15) and zero_point
are stored alongside the data. More precisely, each row first has quantized
values, and then 2-byte fp16 scale and 2-byte zero_offset.)
)DOC")
    .Input(0, "input", "Float32 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
NO_GRADIENT(FloatToFused4BitRowwiseQuantized);

REGISTER_CPU_OPERATOR(
    HalfToFused4BitRowwiseQuantized,
    FloatToFusedNBitRowwiseQuantizedOp<4, at::Half, internal::convertfp16fp32>);
OPERATOR_SCHEMA(HalfToFused4BitRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          (X.dims(X.dims().size() - 1) + 1) / 2 + 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 4-bit row-wise quantization by determining the range
(maximum - minimum) and offset (minimum value) of each row in the input
matrix, and then scaling each element to an 4-bit number between 0 and
15. To later de-quantize values, the scale (range / 15) and zero_point
are stored alongside the data. More precisely, each row first has quantized
values, and then 2-byte fp16 scale and 2-byte zero_offset.)
)DOC")
    .Input(0, "input", "Float16 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
NO_GRADIENT(HalfToFused4BitRowwiseQuantized);

REGISTER_CPU_OPERATOR(
    Fused4BitRowwiseQuantizedToFloat,
    FusedNBitRowwiseQuantizedToFloatOp<4, float, internal::convertfp32fp32>);
OPERATOR_SCHEMA(Fused4BitRowwiseQuantizedToFloat)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          (X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half)) * 2);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
De-quantizes the result of the
FloatToFused4BitRowwiseQuantized operator. The input is expected to first have
quantized values, then 2-byte fp16 scale and 1-byte zero_offset. The output is a
matrix containing only the values, but de-quantized. De-quantization is
performed by multiplying each value by its row's scale and zero_point
parameters. The de-quantized values will thus not be exactly equal to
the original, un-quantized floating point values.
)DOC")
    .Input(
        0,
        "scale_bias_quantized_input",
        "Fused scale, bias and quantized data")
    .Output(0, "float_output", "Float32 data");
NO_GRADIENT(Fused4BitRowwiseQuantizedToFloat);

REGISTER_CPU_OPERATOR(
    Fused4BitRowwiseQuantizedToHalf,
    FusedNBitRowwiseQuantizedToFloatOp<4, at::Half, internal::convertfp32fp16>);
OPERATOR_SCHEMA(Fused4BitRowwiseQuantizedToHalf)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          (X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half)) * 2);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT16);
      return out;
    })
    .SetDoc(R"DOC(
De-quantizes the result of the
FloatToFused4BitRowwiseQuantized operator. The input is expected to first have
quantized values, then 2-byte fp16 scale and 1-byte zero_offset. The output is a
matrix containing only the values, but de-quantized. De-quantization is
performed by multiplying each value by its row's scale and zero_point
parameters. The de-quantized values will thus not be exactly equal to
the original, un-quantized floating point values.
)DOC")
    .Input(
        0,
        "scale_bias_quantized_input",
        "Fused scale, bias and quantized data")
    .Output(0, "float16_output", "Float16 data");
NO_GRADIENT(Fused4BitRowwiseQuantizedToHalf);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FloatToFused4BitRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitRowwiseQuantizedOp<
        4,
        float,
        internal::convertfp32fp32,
        true /*GREEDY*/>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    HalfToFused4BitRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitRowwiseQuantizedOp<
        4,
        at::Half,
        internal::convertfp16fp32,
        true /*GREEDY*/>);

REGISTER_CPU_OPERATOR(
    FloatToFused2BitRowwiseQuantized,
    FloatToFusedNBitRowwiseQuantizedOp<2, float, internal::convertfp32fp32>);
OPERATOR_SCHEMA(FloatToFused2BitRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      // divide over 4 and round up, add 4 for the extra scale and bias
      X.set_dims(
          X.dims().size() - 1,
          (X.dims(X.dims().size() - 1) + 3) / 4 + 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 2-bit row-wise quantization by determining the range
(maximum - minimum) and offset (minimum value) of each row in the input
matrix, and then scaling each element to an 2-bit number between 0 and
3. To later de-quantize values, the scale (range / 3) and zero_point
are stored alongside the data. More precisely, each row first has quantized
values, and then 2-byte fp16 scale and 2-byte zero_offset.)
)DOC")
    .Input(0, "input", "Float32 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
NO_GRADIENT(FloatToFused2BitRowwiseQuantized);

REGISTER_CPU_OPERATOR(
    HalfToFused2BitRowwiseQuantized,
    FloatToFusedNBitRowwiseQuantizedOp<2, at::Half, internal::convertfp16fp32>);
OPERATOR_SCHEMA(HalfToFused2BitRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          (X.dims(X.dims().size() - 1) + 3) / 4 + 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 2-bit row-wise quantization by determining the range
(maximum - minimum) and offset (minimum value) of each row in the input
matrix, and then scaling each element to an 2-bit number between 0 and
3. To later de-quantize values, the scale (range / 3) and zero_point
are stored alongside the data. More precisely, each row first has quantized
values, and then 2-byte fp16 scale and 2-byte zero_offset.)
)DOC")
    .Input(0, "input", "Float16 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
NO_GRADIENT(HalfToFused2BitRowwiseQuantized);

REGISTER_CPU_OPERATOR(
    Fused2BitRowwiseQuantizedToFloat,
    FusedNBitRowwiseQuantizedToFloatOp<2, float, internal::convertfp32fp32>);
OPERATOR_SCHEMA(Fused2BitRowwiseQuantizedToFloat)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          (X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half)) * 4);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
De-quantizes the result of the
FloatToFused2BitRowwiseQuantized operator. The input is expected to first have
quantized values, then 2-byte fp16 scale and 1-byte zero_offset. The output is a
matrix containing only the values, but de-quantized. De-quantization is
performed by multiplying each value by its row's scale and zero_point
parameters. The de-quantized values will thus not be exactly equal to
the original, un-quantized floating point values.
)DOC")
    .Input(
        0,
        "scale_bias_quantized_input",
        "Fused scale, bias and quantized data")
    .Output(0, "float_output", "Float32 data");
NO_GRADIENT(Fused2BitRowwiseQuantizedToFloat);

REGISTER_CPU_OPERATOR(
    Fused2BitRowwiseQuantizedToHalf,
    FusedNBitRowwiseQuantizedToFloatOp<2, at::Half, internal::convertfp32fp16>);
OPERATOR_SCHEMA(Fused2BitRowwiseQuantizedToHalf)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          (X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half)) * 4);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT16);
      return out;
    })
    .SetDoc(R"DOC(
De-quantizes the result of the
FloatToFused2BitRowwiseQuantized operator. The input is expected to first have
quantized values, then 2-byte fp16 scale and 1-byte zero_offset. The output is a
matrix containing only the values, but de-quantized. De-quantization is
performed by multiplying each value by its row's scale and zero_point
parameters. The de-quantized values will thus not be exactly equal to
the original, un-quantized floating point values.
)DOC")
    .Input(
        0,
        "scale_bias_quantized_input",
        "Fused scale, bias and quantized data")
    .Output(0, "float16_output", "Float16 data");
NO_GRADIENT(Fused2BitRowwiseQuantizedToHalf);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FloatToFused2BitRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitRowwiseQuantizedOp<
        2,
        float,
        internal::convertfp32fp32,
        true /*GREEDY*/>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    HalfToFused2BitRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitRowwiseQuantizedOp<
        2,
        at::Half,
        internal::convertfp16fp32,
        true /*GREEDY*/>);

} // namespace caffe2

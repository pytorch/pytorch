#include "caffe2/operators/fused_rowwise_8bit_conversion_ops.h"
#include "c10/util/Registry.h"

namespace caffe2 {

namespace {
void convertfp16fp32(float* dst, const at::Half* src, size_t N) {
  for (size_t i = 0; i < N; i++) {
    dst[i] = src[i];
  }
}

void convertfp32fp16(at::Half* dst, const float* src, size_t N) {
  for (size_t i = 0; i < N; i++) {
    dst[i] = src[i];
  }
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    FloatToFused8BitRowwiseQuantized,
    FloatToFused8BitRowwiseQuantizedOp<
        float,
        float,
        nullptr,
        false,
        CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(FloatToFused8BitRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1, X.dims(X.dims().size() - 1) + 2 * sizeof(float));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 8-bit row-wise quantization by determining the range
(maximum - minimum) and offset (minimum value) of each row in the input
matrix, and then scaling each element to an 8-bit number between 0 and
255. To later de-quantize values, the scale (range / 255) and offset
(bias) are stored alongside the data. More precisely, each row contains
int8 elements for each quantized element, and the last 8 bytes
of each row in the output matrix are a float storing the scale
followed by another float containing the scale.
For N-dimensional input tensor, the first N-1 dimensions are interpreted as
rows and the last dimension is interpreted as a column. For example, an
input tensor with dimension 5x2x4 is interpreted as 10 rows and 4 columns.
)
)DOC")
    .Input(0, "input", "Float32 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(FloatToFused8BitRowwiseQuantized);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    FloatToFused8BitRowwiseQuantizedHalfScaleBias,
    FloatToFused8BitRowwiseQuantizedOp<
        float,
        at::Half,
        nullptr,
        false,
        CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(FloatToFused8BitRowwiseQuantizedHalfScaleBias)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          X.dims(X.dims().size() - 1) + 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 8-bit row-wise quantization by determining the range
(maximum - minimum) and offset (minimum value) of each row in the input
matrix, and then scaling each element to an 8-bit number between 0 and
255. To later de-quantize values, the scale (range / 255) and offset
(bias) are stored alongside the data. More precisely, each row contains
int8 elements for each quantized element, and the last 4 bytes
of each row in the output matrix are a half float storing the scale
followed by another half float containing the scale.)
)DOC")
    .Input(0, "input", "Float32 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(FloatToFused8BitRowwiseQuantizedHalfScaleBias);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    HalfFloatToFused8BitRowwiseQuantized,
    FloatToFused8BitRowwiseQuantizedOp<
        at::Half,
        float,
        convertfp16fp32,
        true,
        CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(HalfFloatToFused8BitRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1, X.dims(X.dims().size() - 1) + 2 * sizeof(float));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 8-bit row-wise quantization by determining the range
(maximum - minimum) and offset (minimum value) of each row in the input
matrix, and then scaling each element to an 8-bit number between 0 and
255. To later de-quantize values, the scale (range / 255) and offset
(bias) are stored alongside the data. More precisely, each row contains
int8 elements for each quantized element, and the last 8 bytes
of each row in the output matrix are a float storing the scale
followed by another float containing the scale.)
)DOC")
    .Input(0, "input", "Float16 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(HalfFloatToFused8BitRowwiseQuantized);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    HalfFloatToFused8BitRowwiseQuantizedHalfScaleBias,
    FloatToFused8BitRowwiseQuantizedOp<
        at::Half,
        at::Half,
        convertfp16fp32,
        true,
        CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(HalfFloatToFused8BitRowwiseQuantizedHalfScaleBias)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          X.dims(X.dims().size() - 1) + 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 8-bit row-wise quantization by determining the range
(maximum - minimum) and offset (minimum value) of each row in the input
matrix, and then scaling each element to an 8-bit number between 0 and
255. To later de-quantize values, the scale (range / 255) and offset
(bias) are stored alongside the data. More precisely, each row contains
int8 elements for each quantized element, and the last 4 bytes
of each row in the output matrix are a float storing the scale
followed by another float containing the scale.)
)DOC")
    .Input(0, "input", "Float16 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(HalfFloatToFused8BitRowwiseQuantizedHalfScaleBias);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    Fused8BitRowwiseQuantizedToFloat,
    Fused8BitRowwiseQuantizedToFloatOp<
        float,
        float,
        nullptr,
        false,
        CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Fused8BitRowwiseQuantizedToFloat)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1, X.dims(X.dims().size() - 1) - 2 * sizeof(float));
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(Fused8BitRowwiseQuantizedToFloat);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    Fused8BitRowwiseQuantizedHalfScaleBiasToFloat,
    Fused8BitRowwiseQuantizedToFloatOp<
        float,
        at::Half,
        nullptr,
        false,
        CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Fused8BitRowwiseQuantizedHalfScaleBiasToFloat)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
De-quantizes the result of the
FloatToFused8BitRowwiseQuantized operator. The input is expected to
encode the scale as a 16-bit float in the second to the last 2 bytes of each
row, followed by the bias as a 16-bit float in the next 2 bytes, and the
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(Fused8BitRowwiseQuantizedHalfScaleBiasToFloat);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    Fused8BitRowwiseQuantizedToHalfFloat,
    Fused8BitRowwiseQuantizedToFloatOp<
        at::Half,
        float,
        convertfp32fp16,
        true,
        CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Fused8BitRowwiseQuantizedToHalfFloat)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1, X.dims(X.dims().size() - 1) - 2 * sizeof(float));
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(Fused8BitRowwiseQuantizedToHalfFloat);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    Fused8BitRowwiseQuantizedHalfScaleBiasToHalfFloat,
    Fused8BitRowwiseQuantizedToFloatOp<
        float,
        at::Half,
        nullptr,
        false,
        CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Fused8BitRowwiseQuantizedHalfScaleBiasToHalfFloat)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
De-quantizes the result of the
FloatToFused8BitRowwiseQuantized operator. The input is expected to
encode the scale as a 16-bit float in the second to the last 2 bytes of each
row, followed by the bias as a 16-bit float in the next 2 bytes, and the
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(Fused8BitRowwiseQuantizedHalfScaleBiasToHalfFloat);

} // namespace caffe2

// To workaround comma

using Fused8BitRowwiseQuantizedToFloatCPUOp =
    caffe2::Fused8BitRowwiseQuantizedToFloatOp<
        float,
        float,
        nullptr,
        false,
        caffe2::CPUContext>;

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    Fused8BitRowwiseQuantizedToFloat,
    "_caffe2::Fused8BitRowwiseQuantizedToFloat(Tensor scale_bias_quantized_input) -> Tensor",
    Fused8BitRowwiseQuantizedToFloatCPUOp);

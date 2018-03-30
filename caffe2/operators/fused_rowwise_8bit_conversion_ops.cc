#include "caffe2/operators/fused_rowwise_8bit_conversion_ops.h"
#include "caffe2/core/registry.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(
    FloatToFused8BitRowwiseQuantized,
    FloatToFused8BitRowwiseQuantizedOp<CPUContext>);
OPERATOR_SCHEMA(FloatToFused8BitRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
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
    Fused8BitRowwiseQuantizedToFloat,
    Fused8BitRowwiseQuantizedToFloatOp<CPUContext>);
OPERATOR_SCHEMA(Fused8BitRowwiseQuantizedToFloat)
    .NumInputs(1)
    .NumOutputs(1)
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
    .Output(0, "float_input", "Float32 data");
NO_GRADIENT(Fused8BitRowwiseQuantizedToFloat);
} // namespace caffe2

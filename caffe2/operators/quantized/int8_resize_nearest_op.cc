#include "caffe2/operators/quantized/int8_resize_nearest_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Int8ResizeNearest, int8::Int8ResizeNearestOp);

// Input: X, output: Y
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Int8ResizeNearest)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension")
    .Arg("output_size", "Output dimensions (HxW). If specified this takes precedence over scale values.")
    .SetDoc(R"DOC(
Resizes the spatial dimensions of the input using nearest neighbor
interpolation. The `width_scale` and `height_scale` arguments
control the size of the output, which is given by:
output_width = floor(input_width * width_scale)
output_height = floor(output_height * height_scale)
)DOC")
    .Input(0, "X", "Input Int8 tensor")
    .Output(0, "Y", "Output Int8 tensor");

} // namespace caffe2

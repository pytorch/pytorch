#include "caffe2/operators/quantized/int8_conv_transpose_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Int8ConvTranspose, int8::Int8ConvTransposeOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Int8ConvTranspose)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .SetDoc(R"DOC(
The transposed convolution consumes an input vector, the filter blob, and
the bias blob, and computes the output. Note that other parameters, such as
the stride and kernel size, or the pads' sizes in each direction are not
necessary for input because they are provided by the
ConvTransposeUnpoolOpBase operator. Various dimension checks are done
implicitly, and the sizes are specified in the Input docs for this operator.
As is expected, the filter is deconvolved with a subset of the
image and the bias is added; this is done throughout the image data and the
output is computed. As a side note on the implementation layout:
conv_transpose_op_impl.h is the templated implementation of the
conv_transpose_op.h file, which is why they are separate files.
  )DOC")
    .Input(
        0,
        "X",
        "Input data blob from previous layer; has size "
        "(N x H x W x C), where N is the batch size, C is the number of channels, and"
        " H and W are the height and width. Note that NHWC is supported now")
    .Input(
        1,
        "filter",
        "The filter blob that will be used in the transposed "
        "convolution; has size (M x kH x kW x C), where C is the number of channels,"
        " and kH and kW are the height and width of the kernel.")
    .Input(
        2,
        "bias",
        "The 1D bias blob that is added through the convolution;"
        "has size (C). Optional, if not passed, will treat it as all 0.")
    .Output(
        0,
        "Y",
        "Output data blob that contains the result of the "
        "transposed convolution. The output dimensions are functions of the kernel"
        " size, stride size, and pad lengths.");

} // namespace caffe2

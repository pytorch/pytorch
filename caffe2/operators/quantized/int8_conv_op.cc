#include "caffe2/operators/quantized/int8_conv_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Int8Conv, int8::Int8ConvOp<int8::Activation::NONE>);

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
const char kConvDoc_int8[] = R"DOC(
[Only NHWC order is supported now]Note that other parameters, such as the stride and
kernel size, or the pads' sizes in each direction are not necessary for input
because they are provided by the ConvPoolOpBase operator. Various dimension
checks are done implicitly, and the sizes are specified in the Input docs for
this operator. As is expected, the filter is convolved with a subset of the
image and the bias is added; this is done throughout the image data and the
output is computed. As a side note on the implementation layout:
conv_op_impl.h is the templated implementation of the conv_op.h file, which is
why they are separate files.
)DOC";

std::function<void(OpSchema&)> ConvDocGenerator(
    const char* dim,
    bool relu_fused = false) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
The convolution operator consumes an input vector, a {dim}filter blob
and a bias blob and computes the output. {conv_doc})DOC";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{conv_doc}", kConvDoc_int8);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data blob from previous layer; has size (N x C x H x W), "
        "where N is the batch size, C is the number of channels, "
        "and H and W are the height and width. Note that this is for the NCHW "
        "usage. On the other hand, the NHWC Op has a different set of "
        "dimension constraints. ");
    schema.Input(
        1,
        "filter",
        "The filter blob that will be used in the "
        "convolutions; has size (M x C x kH x kW), where C is the number of "
        "channels, and kH and kW are the height and width of the kernel.");
    schema.Input(
        2,
        "bias",
        "The 1D bias blob that is added through the "
        "convolution; has size (M).");
    schema.Output(0, "Y", relu_fused ?
        "Output data blob that contains the result of the "
        "convolution. The output dimensions are functions of the kernel size, "
        "stride size, and pad lengths. Output will go through rectified linear "
        "function, where y = max(0, x)." :
        "Output data blob that contains the result of the "
        "convolution. The output dimensions are functions of the kernel size, "
        "stride size, and pad lengths.");
  };
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Int8Conv)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .FillUsing(ConvDocGenerator(""));

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Int8ConvRelu)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .FillUsing(ConvDocGenerator("", true));

} // namespace caffe2

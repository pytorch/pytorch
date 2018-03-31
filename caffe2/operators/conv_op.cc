#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_impl.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

const char* kConvDoc = R"DOC(
Note that other parameters, such as the stride and
kernel size, or the pads' sizes in each direction are not necessary for input
because they are provided by the ConvPoolOpBase operator. Various dimension
checks are done implicitly, and the sizes are specified in the Input docs for
this operator. As is expected, the filter is convolved with a subset of the
image and the bias is added; this is done throughout the image data and the
output is computed. As a side note on the implementation layout:
conv_op_impl.h is the templated implementation of the conv_op.h file, which is
why they are separate files.
)DOC";

std::function<void(OpSchema&)> ConvDocGenerator(const char* dim) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
The convolution operator consumes an input vector, a {dim}filter blob
and a bias blob and computes the output. {conv_doc})DOC";
    ReplaceAll(doc, "{dim}", dim);
    ReplaceAll(doc, "{conv_doc}", kConvDoc);
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
    schema.Output(
        0,
        "Y",
        "Output data blob that contains the result of the "
        "convolution. The output dimensions are functions of the kernel size, "
        "stride size, and pad lengths."
        "");
  };
}
REGISTER_CPU_OPERATOR(Conv, ConvOp<float, CPUContext>);

OPERATOR_SCHEMA(Conv)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .FillUsing(ConvDocGenerator(""))
    .InheritOnnxSchema("Conv");

REGISTER_CPU_OPERATOR(Conv1D, ConvOp<float, CPUContext>);

OPERATOR_SCHEMA(Conv1D)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .FillUsing(ConvDocGenerator("1D "))
    .InheritOnnxSchema("Conv");

REGISTER_CPU_OPERATOR(Conv2D, ConvOp<float, CPUContext>);

OPERATOR_SCHEMA(Conv2D)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .FillUsing(ConvDocGenerator("2D "))
    .InheritOnnxSchema("Conv");

REGISTER_CPU_OPERATOR(Conv3D, ConvOp<float, CPUContext>);

OPERATOR_SCHEMA(Conv3D)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .FillUsing(ConvDocGenerator("3D "))
    .InheritOnnxSchema("Conv");

} // namespace caffe2

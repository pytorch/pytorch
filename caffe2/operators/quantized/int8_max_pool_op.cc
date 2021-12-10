#include "caffe2/operators/quantized/int8_max_pool_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8MaxPool, int8::Int8MaxPoolOp<int8::Activation::NONE>);
REGISTER_CPU_OPERATOR(
    Int8MaxPoolRelu,
    int8::Int8MaxPoolOp<int8::Activation::RELU>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
const char kMaxPoolDoc_int8[] = R"DOC(
consumes an input blob X and applies max pooling across the
the blob according to kernel sizes, stride sizes, and pad lengths defined by the
ConvPoolOpBase operator. Max pooling consisting of taking the maximum value of a
subset of the input tensor according to the kernel size and downsampling the
data into the output blob Y for further processing.
)DOC";

std::function<void(OpSchema&)> MaxPoolDocGenerator(
    const char* dim,
    bool relu_fused = false) {
  return [=](OpSchema& schema) {
    string doc = "MaxPool{dim} {pool_doc}";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{pool_doc}", kMaxPoolDoc_int8);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; dimensions depend on "
        "whether the NCHW or NHWC operators are being used. For example, in "
        "the former, the input has size (N x C x H x W), where N is the batch "
        "size, C is the number of channels, and H and W are the height and the "
        "width of the data. The corresponding permutation of dimensions is "
        "used in the latter case.");
    schema.Output(0, "Y", relu_fused ?
        "Output data tensor from max pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes. Output will go through rectified linear"
        "function, where y = max(0, x)." :
        "Output data tensor from max pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes. Output will go through rectified linear");
  };
}

OPERATOR_SCHEMA(Int8MaxPool)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator(""));

OPERATOR_SCHEMA(Int8MaxPoolRelu)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator("", true));

} // namespace caffe2

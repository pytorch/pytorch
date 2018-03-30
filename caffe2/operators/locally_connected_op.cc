#include <functional>
#include <vector>

#include "caffe2/operators/locally_connected_op.h"
#include "caffe2/operators/locally_connected_op_impl.h"

namespace caffe2 {

namespace {

constexpr char kLCDoc[] = R"DOC(
Note that other parameters, such as the stride and
kernel size, or the pads' sizes in each direction are not necessary for input
because they are provided by the ConvPoolOpBase operator. Various dimension
checks are done implicitly, and the sizes are specified in the Input docs for
this operator. As is expected, the filter is locally connected with a subset of
the image and the bias is added; this is done throughout the image data and the
output is computed. As a side note on the implementation layout:
locally_connected_op_impl.h is the templated implementation of the
locally_connected_op.h file, which is why they are separate files.
)DOC";

std::function<void(OpSchema&)> LCDocGenerator(const char* dim) {
  return [dim](OpSchema& schema) {
    string doc = R"DOC(
The locally connected operator consumes an input vector, a {dim}filter blob
and a bias blob and computes the output. {lc_doc})DOC";
    ReplaceAll(doc, "{dim}", dim);
    ReplaceAll(doc, "{lc_doc}", kLCDoc);
    schema.SetDoc(doc);
    schema.Input(
        1,
        "filter",
        "The filter blob that will be used in the locally connected op; "
        "has size (YH * YW * M x C x kH x kW), where YH and YW are the height "
        "and width of the output image, C is the number of channels, and kH "
        "and kW are the height and width of the kernel.");
    schema.Input(
        2,
        "bias",
        "The 1D bias blob that is added through the locally connected op; "
        "has size (YH * YW * M).");
    schema.Output(
        0,
        "Y",
        "Output data blob that contains the result of the locally connected op."
        "The output dimensions are functions of the kernel size, stride size, "
        "and pad lengths."
        "");
  };
}

} // namespace

REGISTER_CPU_OPERATOR(LC, LocallyConnectedOp<float, CPUContext>);

OPERATOR_SCHEMA(LC)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .FillUsing(LCDocGenerator(""));

REGISTER_CPU_OPERATOR(LC1D, LocallyConnectedOp<float, CPUContext>);

OPERATOR_SCHEMA(LC1D)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .FillUsing(LCDocGenerator("1D "));

REGISTER_CPU_OPERATOR(LC2D, LocallyConnectedOp<float, CPUContext>);

OPERATOR_SCHEMA(LC2D)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .FillUsing(LCDocGenerator("2D "));

REGISTER_CPU_OPERATOR(LC3D, LocallyConnectedOp<float, CPUContext>);

OPERATOR_SCHEMA(LC3D)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .FillUsing(LCDocGenerator("3D "));

REGISTER_CPU_OPERATOR(
    LCGradient,
    LocallyConnectedGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LCGradient).NumInputs(2, 3).NumOutputs(1, 3);

REGISTER_CPU_OPERATOR(
    LC1DGradient,
    LocallyConnectedGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LC1DGradient).NumInputs(2, 3).NumOutputs(1, 3);

REGISTER_CPU_OPERATOR(
    LC2DGradient,
    LocallyConnectedGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LC2DGradient).NumInputs(2, 3).NumOutputs(1, 3);

REGISTER_CPU_OPERATOR(
    LC3DGradient,
    LocallyConnectedGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LC3DGradient).NumInputs(2, 3).NumOutputs(1, 3);

namespace {

class GetLocallyConnectedGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(def_.input_size() == 3 || def_.input_size() == 2);
    ArgumentHelper argsHelper(def_);
    const bool compute_dX =
        !argsHelper.GetSingleArgument<bool>("no_gradient_to_input", 0);

    if (def_.input_size() == 3) {
      if (compute_dX) {
        return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            std::vector<string>{I(0), I(1), GO(0)},
            std::vector<string>{GI(1), GI(2), GI(0)});
      } else {
        return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            std::vector<string>{I(0), I(1), GO(0)},
            std::vector<string>{GI(1), GI(2)});
      }
    } else {
      if (compute_dX) {
        return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            std::vector<string>{I(0), I(1), GO(0)},
            std::vector<string>{GI(1), GI(0)},
            std::vector<Argument>{MakeArgument<int>("no_bias", 1)});
      } else {
        return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            std::vector<string>{I(0), I(1), GO(0)},
            std::vector<string>{GI(1)},
            std::vector<Argument>{MakeArgument<int>("no_bias", 1)});
      }
    }
  }
};

} // namespace

REGISTER_GRADIENT(LC, GetLocallyConnectedGradient);
REGISTER_GRADIENT(LC1D, GetLocallyConnectedGradient);
REGISTER_GRADIENT(LC2D, GetLocallyConnectedGradient);
REGISTER_GRADIENT(LC3D, GetLocallyConnectedGradient);

} // namespace caffe2

#include "caffe2/operators/kmax_pooling_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(kMaxPooling, kMaxPoolingOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    kMaxPoolingGradient,
    kMaxPoolingGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(kMaxPooling)
    .NumInputs(2)
    .NumOutputs(2)
    .SetDoc(R"DOC(This operator calculate the k max pooling of input.)DOC")
    .Input(0, "X", "Input sparse segments")
    .Input(1, "Y", "Length of input sparse segment")
    .Output(0, "kMaxValue", "Output k max values")
    .Output(1, "kMaxIndices", "Output indices corresponding to k max values")
    .Arg(
        "k",
        "the number of top values to return, if the number of values"
        " is smaller than k, the values would be padded with 0 and indices"
        " would be padded with -1.");

struct GetkMaxPoolingGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 2);
    CAFFE_ENFORCE_EQ(def_.output_size(), 2);
    int k = -1;
    if (ArgumentHelper::HasArgument(Def(), "k")) {
      k = GetArgument(Def(), "k").i();
    }
    const auto kmax_pooling_arg = vector<Argument>{MakeArgument<int>("k", k)};
    return SingleGradientDef(
        "kMaxPoolingGradient",
        "",
        vector<string>{I(1), O(1), GO(0)},
        vector<string>{GI(0)},
        kmax_pooling_arg);
  }
};
REGISTER_GRADIENT(kMaxPooling, GetkMaxPoolingGradient);
} // namespace caffe2

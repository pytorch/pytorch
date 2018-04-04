#include "caffe2/operators/dropout_op.h"

namespace caffe2 {

template <>
bool DropoutOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->Resize(X.dims());
  if (is_test_) {
    if (Y != &X) {
      context_.Copy<float, CPUContext, CPUContext>(
          X.size(), X.data<float>(), Y->mutable_data<float>());
    }
    return true;
  } else {
    float scale = 1. / (1. - ratio_);
    // mask=true means keep, and mask=false means not keep, so we will
    // generate probability depending on 1-ratio.
    std::bernoulli_distribution dist(1. - ratio_);
    const float* Xdata = X.data<float>();
    float* Ydata = Y->mutable_data<float>();
    auto mask = Output(1);
    mask->Resize(X.dims());
    bool* mask_data = mask->mutable_data<bool>();
    auto& gen = context_.RandGenerator();
    for (int i = 0; i < X.size(); ++i) {
      mask_data[i] = dist(gen);
      Ydata[i] = Xdata[i] * scale * mask_data[i];
    }
    return true;
  }
}

template <>
bool DropoutGradientOp<float, CPUContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto* dX = Output(0);
  dX->Resize(dY.dims());
  if (is_test_) {
    if (dX != &dY) {
      context_.Copy<float, CPUContext, CPUContext>(
          dY.size(), dY.data<float>(), dX->mutable_data<float>());
    }
    return true;
  } else {
    auto& mask = Input(1);
    CAFFE_ENFORCE_EQ(dY.size(), mask.size());
    const float* dYdata = dY.data<float>();
    const bool* mask_data = mask.data<bool>();
    float* dXdata = dX->mutable_data<float>();
    float scale = 1. / (1. - ratio_);
    for (int i = 0; i < dY.size(); ++i) {
      dXdata[i] = dYdata[i] * mask_data[i] * scale;
    }
    return true;
  }
}

REGISTER_CPU_OPERATOR(Dropout, DropoutOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(DropoutGrad, DropoutGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Dropout)
    .NumInputs(1)
    .NumOutputs(1, 2)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      CAFFE_ENFORCE_EQ(1, in.size());
      vector<TensorShape> out;
      ArgumentHelper argsHelper(def);
      out.push_back(in[0]);
      auto output_mask = !argsHelper.GetSingleArgument<bool>("is_test", 0);
      if (output_mask) {
        out.push_back(in[0]);
        out[1].set_data_type(TensorProto_DataType_BOOL);
      }
      return out;
    })
    .SetDoc(R"DOC(
Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
)DOC")
    .Arg("ratio", "(float, default 0.5) the ratio of random dropout")
    .ArgIsTest(
        "(int) if nonzero, run dropout in test mode where "
        "the output is simply Y = X.")
    .Input(0, "data", "The input data as Tensor.")
    .Output(0, "output", "The output.")
    .Output(
        1,
        "mask",
        "The output mask. If is_test is nonzero, this output is not filled.")
    .InheritOnnxSchema("Dropout");

OPERATOR_SCHEMA(DropoutGrad)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});

class GetDropoutGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argshelper(def_);
    auto is_test = argshelper.GetSingleArgument<bool>("is_test", 0);
    if (is_test) {
      return SingleGradientDef(
          "DropoutGrad", "", vector<string>{GO(0)}, vector<string>{GI(0)});
    } else {
      return SingleGradientDef(
          "DropoutGrad",
          "",
          vector<string>{GO(0), O(1)},
          vector<string>{GI(0)});
    }
  }
};
REGISTER_GRADIENT(Dropout, GetDropoutGradient);
} // namespace caffe2

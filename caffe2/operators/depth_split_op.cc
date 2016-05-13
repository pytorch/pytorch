#include "caffe2/operators/depth_split_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(DepthSplit, DepthSplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(DepthConcat, DepthConcatOp<CPUContext>);

OPERATOR_SCHEMA(DepthSplit)
    .NumInputs(1, 2).NumOutputs(1, INT_MAX);

OPERATOR_SCHEMA(DepthConcat)
    .NumInputs(1, INT_MAX).NumOutputs(2);

class GetDepthSplitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> output_grads;
    for (int i = 0; i < def_.output_size(); ++i) {
      output_grads.push_back(GO(i));
    }
    return SingleGradientDef(
        "DepthConcat", "", output_grads,
        vector<string>{GI(0), "_" + GI(0) + "_dims"});
  }
};
REGISTER_GRADIENT(DepthSplit, GetDepthSplitGradient);

class GetDepthConcatGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grads;
    for (int i = 0; i < def_.input_size(); ++i) {
      grads.push_back(GI(i));
    }
    return SingleGradientDef(
        "DepthSplit", "", vector<string>{GO(0), O(1)}, grads);
  }
};
REGISTER_GRADIENT(DepthConcat, GetDepthConcatGradient);
}  // namespace
}  // namespace caffe2


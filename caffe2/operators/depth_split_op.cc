#include "caffe2/operators/depth_split_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(DepthSplit, DepthSplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(DepthConcat, DepthConcatOp<CPUContext>);

struct GetDepthSplitGradient : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override {
    vector<string> grad_inputs;
    for (const string& out : def.output()) {
      grad_inputs.push_back(GradientName(out));
    }
    return SingleGradientDef(
        "DepthConcat", "", grad_inputs,
        vector<string>{GI(def, 0), "_" + GI(def, 0) + "_dims"});
  }
};
REGISTER_GRADIENT(DepthSplit, GetDepthSplitGradient);

struct GetDepthConcatGradient : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override {
    vector<string> grad_outputs;
    for (const string& in : def.input()) {
      grad_outputs.push_back(GradientName(in));
    }
    return SingleGradientDef(
        "DepthSplit", "", vector<string>{GO(def, 0), O(def, 1)}, grad_outputs);
  }
};
REGISTER_GRADIENT(DepthConcat, GetDepthConcatGradient);
}  // namespace
}  // namespace caffe2


#include "caffe2/operators/depth_split_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(DepthSplit, DepthSplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(DepthConcat, DepthConcatOp<CPUContext>);

struct GetDepthSplitGradient : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    vector<string> grad_inputs;
    for (const string& out : def.output()) {
      grad_inputs.push_back(GradientName(out));
    }
    return new vector<OperatorDef>{
        CreateOperatorDef(
            "DepthConcat", "", grad_inputs,
            std::vector<string>{
                GradientName(def.input(0)),
                "_" + GradientName(def.input(0)) + "_dims"})};
  }
};
REGISTER_GRADIENT(DepthSplit, GetDepthSplitGradient);

struct GetDepthConcatGradient : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    vector<string> grad_outputs;
    for (const string& in : def.input()) {
      grad_outputs.push_back(GradientName(in));
    }
    return new vector<OperatorDef>{
        CreateOperatorDef(
            "DepthSplit", "",
            std::vector<string>{GradientName(def.output(0)), def.output(1)},
            grad_outputs)};
  }
};
REGISTER_GRADIENT(DepthConcat, GetDepthConcatGradient);
}  // namespace
}  // namespace caffe2


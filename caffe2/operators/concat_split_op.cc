#include "caffe2/operators/concat_split_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(Split, SplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(Concat, ConcatOp<CPUContext>);
OPERATOR_SCHEMA(Split)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .Input(0, "input", "The tensor to split")
    .Input(1, "split", "Optional list of output lengths (see also arg 'split')")
    .Arg("axis", "Which axis to split on")
    .Arg("split", "length of each output")
    .Arg("order", "Either NHWC or NCWH, will split on C axis")
    .SetDoc(R"DOC(Split a tensor into a list of tensors, along the specified
    'axis'. The lengths of the split can be specified using argument 'axis' or
    optional second input blob to the operator. Otherwise, the tensor is split
    to equal sized parts.
    )DOC");
OPERATOR_SCHEMA(Concat)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2)
    .Arg("axis", "Which axis to concat on")
    .Arg("order", "Either NHWC or HCWH, will concat on C axis")
    .Arg(
        "add_axis",
        "Pass 1 to add the axis specified in arg 'axis' to all "
        "input tensors")
    .SetDoc("Concatenate a list of tensors into a single tensor")
    .Output(0, "concat_result", "Concatenated tensor")
    .Output(1, "split_info", "The dimensions of the inputs.");

// Backward compatibility names.
REGISTER_CPU_OPERATOR(DepthSplit, SplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(DepthConcat, ConcatOp<CPUContext>);
OPERATOR_SCHEMA(DepthSplit)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .SetDoc("Backward compatible operator name for Split.");
OPERATOR_SCHEMA(DepthConcat)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2)
    .SetDoc("Backward compatible operator name for Concat.");

class GetSplitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> output_grads;
    for (int i = 0; i < def_.output_size(); ++i) {
      if (!GradOut(i).IsEmpty()) {
        output_grads.push_back(GO(i));
      }
    }
    if (output_grads.empty()) {
      return {};
    }
    return SingleGradientDef(
        "Concat", "", output_grads,
        vector<string>{GI(0), "_" + GI(0) + "_dims"});
  }
};
REGISTER_GRADIENT(Split, GetSplitGradient);
REGISTER_GRADIENT(DepthSplit, GetSplitGradient);

class GetConcatGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (GradOut(0).IsEmpty()) {
      return {};
    }
    vector<string> grads;
    for (int i = 0; i < def_.input_size(); ++i) {
      grads.push_back(GI(i));
    }
    return SingleGradientDef(
        "Split", "", vector<string>{GO(0), O(1)}, grads);
  }
};
REGISTER_GRADIENT(Concat, GetConcatGradient);
REGISTER_GRADIENT(DepthConcat, GetConcatGradient);
}  // namespace caffe2

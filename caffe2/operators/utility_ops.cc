#include "caffe2/operators/utility_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(Free, FreeOp);
REGISTER_CPU_OPERATOR(Print, PrintOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(PrintInt, PrintOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(Flatten, FlattenOp<CPUContext>);
REGISTER_CPU_OPERATOR(Alias, AliasOp<CPUContext>);
REGISTER_CPU_OPERATOR(ReshapeLike, ReshapeLikeOp<CPUContext>);
REGISTER_CPU_OPERATOR(Split, SplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(Sum, SumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(WeightedSum, WeightedSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    ScatterWeightedSum,
    ScatterWeightedSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Copy, CopyOp<CPUContext, CPUContext, CPUContext>);
REGISTER_CPU_OPERATOR(RecordShape, RecordShapeOp<CPUContext>);
REGISTER_CPU_OPERATOR(Gather, GatherOp<float, CPUContext>);


// FreeOp frees the content of the output blob. We allow it to take in input
// blobs purely for the reason that it can "wait" on the input blobs to be
// produced by some of the earlier operators before it is used.
OPERATOR_SCHEMA(Free).NumInputs(0, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(Print).NumInputs(1, INT_MAX).NumOutputs(0);
OPERATOR_SCHEMA(PrintInt).NumInputs(1, INT_MAX).NumOutputs(0);
OPERATOR_SCHEMA(Flatten).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Alias).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ReshapeLike).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Split).NumInputs(1).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(Sum).NumInputs(1, INT_MAX).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(WeightedSum)
    .NumInputs([](int n) { return (n > 0 && n % 2 == 0); })
    .NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(ScatterWeightedSum)
    .NumInputs([](int n) { return (n > 3 && (n - 3) % 2 == 0); })
    .NumOutputs(1).EnforceInplace({{0, 0}});
OPERATOR_SCHEMA(Copy).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(RecordShape).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Gather).NumInputs(2).NumOutputs(1);


SHOULD_NOT_DO_GRADIENT(Free);
SHOULD_NOT_DO_GRADIENT(Print);
SHOULD_NOT_DO_GRADIENT(PrintInt);
SHOULD_NOT_DO_GRADIENT(RecordShape);

class GetFlattenGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ReshapeLike", "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Flatten, GetFlattenGradient);

class GetAliasGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // We will simply pass-along the gradient. Nothing needs to
    // be calculated.
    SetDense(0, GO(0));
    return vector<OperatorDef>();
  }
};
REGISTER_GRADIENT(Alias, GetAliasGradient);

SHOULD_NOT_DO_GRADIENT(ReshapeLike);

class GetSplitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_output;
    for (int i = 0; i < def_.output_size(); ++i) {
      grad_output.push_back(GO(i));
    }
    return SingleGradientDef(
        "Sum", "", grad_output,
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Split, GetSplitGradient);

class GetSumGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    for (auto i = 0; i < def_.input_size(); ++i) {
      SetDense(i, GO(0));
    }
    return vector<OperatorDef>();
  }
};
REGISTER_GRADIENT(Sum, GetSumGradient);

// TODO(jiayq): Weighted sum is originally intended to be used in SGD, but in
// theory, its gradient DOES exist. Should we enable the gradient?
SHOULD_NOT_DO_GRADIENT(WeightedSum);
SHOULD_NOT_DO_GRADIENT(ScatterWeightedSum);

// TODO(jiayq): Copy is a bit tricky because one need to figure out correctly
// where the input lies (e.g. for muji, which gpu). Right now I am marking it
// as not gradient ready.
SHOULD_NOT_DO_GRADIENT(Copy);

class GetGatherGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // For now we don't do any reshaping as the consumer of this op would
    // probably be ScatterUpdate which is intenionally ignores shapes. We might
    // need to revisit it in the future for correctness purposes. The right
    // shape for the output woild be to flatten INDICES and collapse first X
    // dims of GRAD
    using Op = GatherOp<float, CPUContext>;
    SetSparse(Op::DATA, I(Op::INDICES), GO(0));
    return vector<OperatorDef>();
  }
};
REGISTER_GRADIENT(Gather, GetGatherGradient);

}  // namespace
}  // namespace caffe2



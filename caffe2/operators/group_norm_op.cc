// Copyright 2004-present Facebook. All Rights Reserved.

#include "group_norm_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(GroupNorm, GroupNormOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    GroupNormGradient,
    GroupNormGradientOp<float, CPUContext>);

/* Warning: mu and sig are for backward usage or reference. They should NOT be
used as forward activations as they have no direct gradients computed */

// Input: X, gamma, beta; Output: Y, mu, sig
OPERATOR_SCHEMA(GroupNorm)
    .NumInputs(3)
    .NumOutputs(3)
    .SetDoc(R"DOC(
Group Normalization (GN) operation: https://arxiv.org/abs/1803.08494
)DOC")
    .Arg("num_groups", "(int) default 32; number of groups used by GN.")
    .Arg("epsilon", "(float) default 1e-5; small constant added to var.")
    .Input(
        0,
        "X",
        ">=4D feature map input of shape (N, C, H, W) or (N, C, T, H, W)")
    .Input(
        1,
        "gamma",
        "The scale as a 1-dimensional tensor of size C to be applied to the "
        "output.")
    .Input(
        2,
        "beta",
        "The bias as a 1-dimensional tensor of size C to be applied to the "
        "output.")
    .Output(0, "Y", "The output >=4-dimensional tensor of the same shape as X.")
    .Output(
        1,
        "mean",
        "The mean of shape (N, G). "
        "For backward usage or reference. "
        "Cannot be used as activations.")
    .Output(
        2,
        "std",
        "The std of shape (N, G). "
        "For backward usage or reference. "
        "Cannot be used as activations.");

// Input: dY, X, gamma, beta, mu, sig; Output: dX, dgamma, dbeta
OPERATOR_SCHEMA(GroupNormGradient).NumInputs(6).NumOutputs(3);

class GetGroupNormGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "GroupNormGradient",
        "",
        vector<string>{GO(0), I(0), I(1), I(2), O(1), O(2)},
        vector<string>{GI(0), GI(1), GI(2)});
  }
};

REGISTER_GRADIENT(GroupNorm, GetGroupNormGradient);

} // namespace caffe2

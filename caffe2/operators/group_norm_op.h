// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef GROUP_NORM_OP_H_
#define GROUP_NORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class GroupNormOp final : public Operator<Context> {
 public:
  GroupNormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_groups_(OperatorBase::GetSingleArgument<int32_t>("num_groups", 32)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
    // return true;
  }

 protected:
  int num_groups_;
  float epsilon_;
};

template <typename T, class Context>
class GroupNormGradientOp final : public Operator<Context> {
 public:
  GroupNormGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        num_groups_(OperatorBase::GetSingleArgument<int32_t>("num_groups", 32)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
    // return true;
  }

 protected:
  int num_groups_;
  float epsilon_;
  Tensor<Context> buffer_; // NxC size
  Tensor<Context> buffer1_; // NxC size

  Tensor<Context> buffer2_; // NxG size
  Tensor<Context> buffer3_; // NxG size

  Tensor<Context> sum_multiplier_; // max(HxW, dim_per_gp, N)
};

} // namespace caffe2

#endif // GROUP_NORM_OP_H_

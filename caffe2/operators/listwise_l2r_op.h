// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class LambdaRankNdcgOp final : public Operator<Context> {
 public:
  LambdaRankNdcgOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        use_ndcg_as_loss_(
            this->template GetSingleArgument<bool>("use_ndcg_as_loss", false)),
        use_exp_gain_(
            this->template GetSingleArgument<bool>("use_exp_gain", true)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 private:
  INPUT_TAGS(PRED, REL, SESSION_LENS);
  OUTPUT_TAGS(LOSS, DPRED);

  void ResizeInvLogITensor(int);
  void ComputeDiscounts(int*, int);
  float LambdaRankNdcgSession(
      int start_index,
      int end_index,
      const Tensor& y,
      const Tensor& r,
      Tensor** dy);
  bool use_ndcg_as_loss_;
  bool use_exp_gain_;
  Tensor gain_{Context::GetDeviceType()};
  Tensor discount_{Context::GetDeviceType()};
  Tensor rank_idx_{Context::GetDeviceType()};
  Tensor ideal_idx_{Context::GetDeviceType()};
  Tensor lambda_{Context::GetDeviceType()};
  Tensor inv_log_i_{Context::GetDeviceType()};
};

template <typename T, class Context>
class LambdaRankNdcgGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(LambdaRankNdcgGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 private:
  INPUT_TAGS(Y, SESSION_LENS, DY_CACHE, DLOSS);
  OUTPUT_TAGS(DY);
};

} // namespace caffe2

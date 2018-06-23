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
        use_ndcg_as_loss_(OperatorBase::template GetSingleArgument<bool>(
            "use_ndcg_as_loss",
            false)) {}
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
      const Tensor<CPUContext>& y,
      const Tensor<CPUContext>& r,
      Tensor<CPUContext>** dy);
  bool use_ndcg_as_loss_;
  Tensor<Context> gain_;
  Tensor<Context> discount_;
  Tensor<Context> rank_idx_;
  Tensor<Context> ideal_idx_;
  Tensor<Context> lambda_;
  Tensor<Context> inv_log_i_;
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

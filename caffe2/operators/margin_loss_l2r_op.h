// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SessionMarginLossOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit SessionMarginLossOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        margin_(this->template GetSingleArgument<float>("margin", 1.0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 private:
  INPUT_TAGS(PRED, LABEL, SESSION_LENS);
  OUTPUT_TAGS(LOSS, DPRED);

  void ResizeInvLogITensor(int);
  void ComputeDiscounts(int*, int);
  float SessionMarginLoss(
      int start_index,
      int end_index,
      const Tensor& pred,
      const Tensor& label,
      Tensor** dpred);
  float margin_;
  Tensor label_relation_sign_;
  Tensor margin_diff_;
};

template <typename T, class Context>
class SessionMarginLossGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SessionMarginLossGradientOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 private:
  INPUT_TAGS(PRED, SESSION_LENS, PRECOMPUTED_DPRED, DLOSS);
  OUTPUT_TAGS(DPRED);
};

} // namespace caffe2

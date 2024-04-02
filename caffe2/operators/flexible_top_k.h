// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_FLEXIBLE_TOP_K_H_
#define CAFFE2_OPERATORS_FLEXIBLE_TOP_K_H_

#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class FlexibleTopKOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit FlexibleTopKOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override;
};

template <typename T, class Context>
class FlexibleTopKGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit FlexibleTopKGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FLEXIBLE_TOP_K_H_

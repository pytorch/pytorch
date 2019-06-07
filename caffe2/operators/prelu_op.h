#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class PReluOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit PReluOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  StorageOrder order_;
};

template <typename T, class Context>
class PReluGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit PReluGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  StorageOrder order_;
};

} // namespace caffe2

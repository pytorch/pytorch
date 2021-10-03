#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class LeakyReluOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit LeakyReluOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...), alpha_(0.01) {
    if (HasArgument("alpha")) {
      alpha_ = static_cast<T>(
          this->template GetSingleArgument<float>("alpha", 0.01));
    }
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  T alpha_;
};

template <typename T, class Context>
class LeakyReluGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit LeakyReluGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...), alpha_(0.01) {
    if (HasArgument("alpha")) {
      alpha_ = static_cast<T>(
          this->template GetSingleArgument<float>("alpha", 0.01));
    }
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  T alpha_;
};

} // namespace caffe2

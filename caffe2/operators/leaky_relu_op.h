#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class LeakyReluOp : public Operator<Context> {
 public:
  LeakyReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), alpha_(0) {
    if (HasArgument("alpha")) {
      alpha_ =
          static_cast<T>(OperatorBase::GetSingleArgument<float>("alpha", 0));
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
  LeakyReluGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), alpha_(0) {
    if (HasArgument("alpha")) {
      alpha_ =
          static_cast<T>(OperatorBase::GetSingleArgument<float>("alpha", 0));
    }
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  T alpha_;
};

} // namespace caffe2

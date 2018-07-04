#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class EluOp final : public Operator<Context> {
 public:
  EluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1.0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  T alpha_;
};

template <typename T, class Context>
class EluGradientOp final : public Operator<Context> {
 public:
  EluGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 1.0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  T alpha_;
};

} // namespace caffe2

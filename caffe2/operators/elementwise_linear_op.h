#ifndef CAFFE2_OPERATORS_ELEMENTWISE_LINEAR_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_LINEAR_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
template <typename T, class Context, class Engine = DefaultEngine>
class ElementwiseLinearOp final : public Operator<Context> {
 public:
  ElementwiseLinearOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int axis_;
};

template <typename T, class Context, class Engine = DefaultEngine>
class ElementwiseLinearGradientOp final : public Operator<Context> {
 public:
  ElementwiseLinearGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int axis_;
};

} // namespace caffe2

#endif  // CAFFE2_OPERATORS_ELEMENTWISE_LINEAR_OP_H_

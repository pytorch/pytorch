#ifndef CAFFE2_OPERATORS_SOFTMAX_OP_H_
#define CAFFE2_OPERATORS_SOFTMAX_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SoftmaxOp final : public Operator<Context> {
 public:
  SoftmaxOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
      axis_(OperatorBase::GetSingleArgument<int>("axis", 1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int axis_;
  Tensor<Context> scale_;
  Tensor<Context> rowmax_;
  Tensor<Context> sum_multiplier_;
};

template <typename T, class Context>
class SoftmaxGradientOp final : public Operator<Context> {
 public:
  SoftmaxGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int axis_;
  Tensor<Context> scale_;
  Tensor<Context> sum_multiplier_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SOFTMAX_OP_H_

#ifndef CAFFE2_OPERATORS_ELEMENTWISE_LINEAR_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_LINEAR_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
template <typename T, class Context, class Engine = DefaultEngine>
class ElementwiseLinearOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit ElementwiseLinearOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        axis_(this->template GetSingleArgument<int>("axis", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int axis_;
};

template <typename T, class Context, class Engine = DefaultEngine>
class ElementwiseLinearGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit ElementwiseLinearGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        axis_(this->template GetSingleArgument<int>("axis", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int axis_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_LINEAR_OP_H_

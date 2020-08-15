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
  template <class... Args>
  explicit SoftmaxOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        axis_(this->template GetSingleArgument<int>("axis", 1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int axis_;
  Tensor scale_;
  Tensor rowmax_;
  Tensor sum_multiplier_;
};

template <typename T, class Context>
class SoftmaxGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit SoftmaxGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        axis_(this->template GetSingleArgument<int>("axis", 1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int axis_;
  Tensor scale_;
  Tensor sum_multiplier_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SOFTMAX_OP_H_

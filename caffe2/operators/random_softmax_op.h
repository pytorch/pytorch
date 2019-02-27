#ifndef CAFFE2_OPERATORS_RANDOM_SOFTMAX_OP_H_
#define CAFFE2_OPERATORS_RANDOM_SOFTMAX_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class RandomSoftmaxOp final : public Operator<Context> {
 public:
  RandomSoftmaxOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(this->template GetSingleArgument<int>("axis", 1)),
        num_sampled_(this->template GetSingleArgument<int>("num_sampled", 1)) {
    CAFFE_ENFORCE_GT(num_sampled_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int axis_;
  int num_sampled_;
  Tensor scale_;
  Tensor rowmax_;
  Tensor sum_multiplier_;
  Tensor rand_;
};

template <typename T, class Context>
class RandomSoftmaxGradientOp final : public Operator<Context> {
 public:
  RandomSoftmaxGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(this->template GetSingleArgument<int>("axis", 1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int axis_;
  Tensor sampled_dX_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RANDOM_SOFTMAX_OP_H_

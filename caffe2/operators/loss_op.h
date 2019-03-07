#ifndef CAFFE2_OPERATORS_LOSS_OP_H_
#define CAFFE2_OPERATORS_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/reduction_ops.h"
#include "caffe2/operators/utility_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// AveragedLoss takes in the input and produces the output loss value as
// the average of the input.
template <typename T, class Context>
class AveragedLoss final : public SumElementsOp<T, Context> {
 public:
  template <class... Args>
  explicit AveragedLoss(Args&&... args)
      : SumElementsOp<T, Context>(std::forward<Args>(args)..., true) {}
  ~AveragedLoss() {}
};

template <typename T, class Context>
class AveragedLossGradient final : public SumElementsGradientOp<T, Context> {
 public:
  template <class... Args>
  explicit AveragedLossGradient(Args&&... args)
      : SumElementsGradientOp<T, Context>(std::forward<Args>(args)..., true) {}
  ~AveragedLossGradient() {}
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOSS_OP_H_

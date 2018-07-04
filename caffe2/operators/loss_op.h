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
  AveragedLoss(const OperatorDef& operator_def, Workspace* ws)
      : SumElementsOp<T, Context>(operator_def, ws, true) {}
  ~AveragedLoss() {}
};

template <typename T, class Context>
class AveragedLossGradient final : public SumElementsGradientOp<T, Context> {
 public:
  AveragedLossGradient(const OperatorDef& operator_def, Workspace* ws)
      : SumElementsGradientOp<T, Context>(operator_def, ws, true) {}
  ~AveragedLossGradient() {}
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOSS_OP_H_

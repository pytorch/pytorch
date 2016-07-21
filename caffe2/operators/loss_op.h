#ifndef CAFFE2_OPERATORS_LOSS_OP_H_
#define CAFFE2_OPERATORS_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

// AveragedLoss takes in the input and produces the output loss value as.
// the average of the input.
template <typename T, class Context>
class AveragedLoss final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(AveragedLoss);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Loss = Output(0);
    Loss->Resize(vector<TIndex>());
    T* loss_data = Loss->template mutable_data<T>();
    // Well... technically we won't need to sum and scale, but I am too lazy
    // to write an average function.
    math::Sum<T, Context>(
        X.size(), X.template data<T>(), loss_data, &context_);
    math::Scale<T, Context>(
        1, static_cast<T>(1.) / X.size(), loss_data, loss_data,
        &context_);
    return true;
  }

 protected:
  // Input: X, output: Loss
};

template <typename T, class Context>
class AveragedLossGradient final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(AveragedLossGradient);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    TensorCPU loss_grad = TensorCPU(Input(1));
    auto* dX = Output(0);
    dX->ResizeLike(X);
    DCHECK_EQ(loss_grad.size(), 1);
    math::Set<T, Context>(
        dX->size(), static_cast<T>(loss_grad.data<T>()[0]) / X.size(),
        dX->template mutable_data<T>(),
        &context_);
    return true;
  }
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_LOSS_OP_H_

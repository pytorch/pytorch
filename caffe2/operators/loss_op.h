#ifndef CAFFE2_OPERATORS_LOSS_OP_H_
#define CAFFE2_OPERATORS_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

// AveragedLoss takes in the input and produces two outputs: one being the loss
// value, and one being the gradient.
template <typename T, class Context>
class AveragedLoss final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(AveragedLoss);
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Loss = Output(0);
    auto* dX = Output(1);
    Loss->Reshape(std::vector<int>());
    dX->ReshapeLike(X);
    math::Set<T, Context>(
        dX->size(), static_cast<T>(1.) / X.size(), dX->template mutable_data<T>(),
        &device_context_);
    math::Dot<T, Context>(
        X.size(), X.template data<T>(), dX->template data<T>(),
        Loss->template mutable_data<T>(), &device_context_);
    return true;
  }

 protected:
  INPUT_OUTPUT_STATS(1, 1, 2, 2);
  DISABLE_COPY_AND_ASSIGN(AveragedLoss);
};

template <typename T, class Context>
class WeightedSumLoss final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(WeightedSumLoss);
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& W = Input(1);
    CAFFE_DCHECK_EQ(X.size(), W.size());
    auto* Loss = Output(0);
    auto* dX = Output(1);
    Loss->Reshape(std::vector<int>());
    math::Dot<T, Context>(
        X.size(), X.template data<T>(), W.template data<T>(),
        Loss->template mutable_data<T>(), &device_context_);
    dX->ReshapeLike(X);
    dX->ShareData(W);
    return true;
  }

 protected:
  INPUT_OUTPUT_STATS(2, 2, 2, 2);
  DISABLE_COPY_AND_ASSIGN(WeightedSumLoss);
};


}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_LOSS_OP_H_

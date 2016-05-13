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
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Loss = Output(0);
    Loss->Reshape(vector<TIndex>());
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
  DISABLE_COPY_AND_ASSIGN(AveragedLoss);
};

template <typename T, class Context>
class AveragedLossGradient final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(AveragedLossGradient);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* dX = Output(0);
    dX->ReshapeLike(X);
    math::Set<T, Context>(
        dX->size(), static_cast<T>(1.) / X.size(),
        dX->template mutable_data<T>(),
        &context_);
    return true;
  }

 protected:
  // Input: X, output: dX
  DISABLE_COPY_AND_ASSIGN(AveragedLossGradient);
};


template <typename T, class Context>
class WeightedSumLoss final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(WeightedSumLoss);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& W = Input(1);
    CAFFE_DCHECK_EQ(X.size(), W.size());
    auto* Loss = Output(0);
    Loss->Reshape(vector<TIndex>());
    math::Dot<T, Context>(
        X.size(), X.template data<T>(), W.template data<T>(),
        Loss->template mutable_data<T>(), &context_);
    return true;
  }

 protected:
  DISABLE_COPY_AND_ASSIGN(WeightedSumLoss);
};


template <typename T, class Context>
class WeightedSumLossGradient final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(WeightedSumLossGradient);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& W = Input(0);
    auto* dX = Output(0);
    dX->ReshapeLike(W);
    dX->ShareData(W);
    return true;
  }

 protected:
  DISABLE_COPY_AND_ASSIGN(WeightedSumLossGradient);
};


}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_LOSS_OP_H_

#ifndef CAFFE2_OPERATORS_LOSS_OP_H_
#define CAFFE2_OPERATORS_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "glog/logging.h"

namespace caffe2 {

// AveragedLoss takes in the input and produces two outputs: one being the loss
// value, and one being the gradient.
template <typename dtype, class DeviceContext>
class AveragedLoss final : public Operator<dtype, DeviceContext> {
 public:
  USE_SIMPLE_CTOR_DTOR(AveragedLoss);
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Loss = Output(0);
    auto* dX = Output(1);
    Loss->Reshape(std::vector<int>());
    dX->ReshapeLike(X);
    math::Set<dtype, DeviceContext>(
        dX->size(), static_cast<dtype>(1.) / X.size(), dX->mutable_data(),
        &device_context_);
    math::Dot<dtype, DeviceContext>(
        X.size(), X.data(), dX->data(), Loss->mutable_data(), &device_context_);
    return true;
  }

 protected:
  INPUT_OUTPUT_STATS(1, 1, 2, 2);
  DISABLE_COPY_AND_ASSIGN(AveragedLoss);
};

template <typename dtype, class DeviceContext>
class WeightedSumLoss final : public Operator<dtype, DeviceContext> {
 public:
  USE_SIMPLE_CTOR_DTOR(WeightedSumLoss);
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& W = Input(1);
    DCHECK_EQ(X.size(), W.size());
    auto* Loss = Output(0);
    auto* dX = Output(1);
    Loss->Reshape(std::vector<int>());
    math::Dot<dtype, DeviceContext>(
        X.size(), X.data(), W.data(), Loss->mutable_data(), &device_context_);
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

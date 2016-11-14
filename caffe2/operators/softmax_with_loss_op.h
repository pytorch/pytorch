#ifndef SOFTMAX_WITH_LOSS_OP_H_
#define SOFTMAX_WITH_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SoftmaxWithLossOp final : public Operator<Context> {
 public:
  SoftmaxWithLossOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        spatial_mode_(OperatorBase::GetSingleArgument<int>("spatial", 0)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float scale_;
  int spatial_mode_;
  StorageOrder order_;

  Tensor<Context> losses_; // Per example loss
  Tensor<Context> sum_multiplier_; // Vector of ones for summing via dot prod
};

template <typename T, class Context>
class SoftmaxWithLossGradientOp final : public Operator<Context> {
 public:
  SoftmaxWithLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        spatial_mode_(OperatorBase::GetSingleArgument<int>("spatial", 0)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float scale_;
  int spatial_mode_;
  Tensor<Context> sum_multiplier_;
  StorageOrder order_;
};

} // namespace caffe2

#endif // SOFTMAX_WITH_LOSS_OP_H_

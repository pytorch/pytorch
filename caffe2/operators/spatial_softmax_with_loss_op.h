#ifndef SPATIAL_SOFTMAX_WITH_LOSS_OP_H_
#define SPATIAL_SOFTMAX_WITH_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SpatialSoftmaxWithLossOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit SpatialSoftmaxWithLossOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float scale_;
  StorageOrder order_;

  Tensor losses_; // Per example loss
  Tensor rowmax_; // per example row max
  Tensor weights_; // unignored weights
  Tensor sum_multiplier_; // Vector of ones for summing via dot prod
  Tensor total_weight_ptr_;
  Tensor scratch_{Context::GetDeviceType()};
};

template <typename T, class Context>
class SpatialSoftmaxWithLossGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit SpatialSoftmaxWithLossGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))),
        only_loss_(this->template GetSingleArgument<bool>("only_loss", false)) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float scale_;
  Tensor sum_multiplier_;
  Tensor weights_; // unignored weights
  Tensor total_weight_ptr_;
  StorageOrder order_;
  bool only_loss_;
  Tensor scratch_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // SOFTMAX_WITH_LOSS_OP_H_

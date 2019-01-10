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
  SpatialSoftmaxWithLossOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
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

  Tensor losses_{Context::GetDeviceType()}; // Per example loss
  Tensor rowmax_{Context::GetDeviceType()}; // per example row max
  Tensor weights_{Context::GetDeviceType()}; // unignored weights
  Tensor sum_multiplier_{
      Context::GetDeviceType()}; // Vector of ones for summing via dot prod
  Tensor total_weight_ptr_{Context::GetDeviceType()};
  Tensor scratch_{Context::GetDeviceType()};
};

template <typename T, class Context>
class SpatialSoftmaxWithLossGradientOp final : public Operator<Context> {
 public:
  SpatialSoftmaxWithLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
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
  Tensor sum_multiplier_{Context::GetDeviceType()};
  Tensor weights_{Context::GetDeviceType()}; // unignored weights
  Tensor total_weight_ptr_{Context::GetDeviceType()};
  StorageOrder order_;
  bool only_loss_;
  Tensor scratch_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // SOFTMAX_WITH_LOSS_OP_H_

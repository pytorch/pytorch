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
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        label_prob_mode_(this->template GetSingleArgument<int>("label_prob", 0)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))),
        axis_(this->template GetSingleArgument<int>("axis", 1)) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float scale_;
  int label_prob_mode_;
  StorageOrder order_;
  int axis_;

  Tensor losses_{Context::GetDeviceType()}; // Per example loss
  Tensor rowmax_{Context::GetDeviceType()}; // per example row max
  Tensor weights_{Context::GetDeviceType()}; // unignored weights
  Tensor sum_multiplier_{
      Context::GetDeviceType()}; // Vector of ones for summing via dot prod
  Tensor total_weight_ptr_{Context::GetDeviceType()};
  Tensor scratch_{Context::GetDeviceType()};
};

template <typename T, class Context>
class SoftmaxWithLossGradientOp final : public Operator<Context> {
 public:
  SoftmaxWithLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        label_prob_mode_(this->template GetSingleArgument<int>("label_prob", 0)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))),
        only_loss_(this->template GetSingleArgument<bool>("only_loss", false)),
        axis_(this->template GetSingleArgument<int>("axis", 1)) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float scale_;
  int label_prob_mode_;
  Tensor sum_multiplier_{Context::GetDeviceType()};
  Tensor weights_{Context::GetDeviceType()}; // unignored weights
  Tensor total_weight_ptr_{Context::GetDeviceType()};
  StorageOrder order_;
  bool only_loss_;
  int axis_;
  Tensor scratch_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // SOFTMAX_WITH_LOSS_OP_H_

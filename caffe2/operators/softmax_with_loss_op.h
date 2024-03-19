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
  template <class... Args>
  explicit SoftmaxWithLossOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        label_prob_mode_(
            this->template GetSingleArgument<int>("label_prob", 0)),
        average_by_batch_size_(
            this->template GetSingleArgument<int>("average_by_batch_size", 0)),
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
  int average_by_batch_size_;
  StorageOrder order_;
  int axis_;

  Tensor losses_; // Per example loss
  Tensor rowmax_; // per example row max
  Tensor weights_; // unignored weights
  Tensor sum_multiplier_; // Vector of ones for summing via dot prod
  Tensor total_weight_ptr_;
  // passed to a function
  Tensor scratch_{Context::GetDeviceType()};
};

template <typename T, class Context>
class SoftmaxWithLossGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit SoftmaxWithLossGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        label_prob_mode_(
            this->template GetSingleArgument<int>("label_prob", 0)),
        average_by_batch_size_(
            this->template GetSingleArgument<int>("average_by_batch_size", 0)),
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
  int average_by_batch_size_;
  // not used?
  Tensor sum_multiplier_{Context::GetDeviceType()};
  Tensor weights_; // unignored weights
  Tensor total_weight_ptr_;
  StorageOrder order_;
  bool only_loss_;
  int axis_;
  Tensor scratch_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // SOFTMAX_WITH_LOSS_OP_H_

#pragma once

#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(ResizeNearest);

namespace caffe2 {

template <typename T, class Context>
class ResizeNearestOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit ResizeNearestOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        width_scale_(1),
        height_scale_(1),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))) {
    if (HasArgument("width_scale")) {
      width_scale_ = static_cast<T>(
          this->template GetSingleArgument<float>("width_scale", 1));
    }
    if (HasArgument("height_scale")) {
      height_scale_ = static_cast<T>(
          this->template GetSingleArgument<float>("height_scale", 1));
    }

    CAFFE_ENFORCE_GT(width_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);

    CAFFE_ENFORCE(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
  bool RunOnDeviceWithOrderNCHW();
  bool RunOnDeviceWithOrderNHWC();

 protected:
  T width_scale_;
  T height_scale_;
  StorageOrder order_;
};

template <typename T, class Context>
class ResizeNearestGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit ResizeNearestGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        width_scale_(1),
        height_scale_(1),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))) {
    width_scale_ = static_cast<T>(
        this->template GetSingleArgument<float>("width_scale", 1));
    height_scale_ = static_cast<T>(
        this->template GetSingleArgument<float>("height_scale", 1));

    CAFFE_ENFORCE_GT(width_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);

    CAFFE_ENFORCE(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
  bool RunOnDeviceWithOrderNCHW();
  bool RunOnDeviceWithOrderNHWC();

 protected:
  T width_scale_;
  T height_scale_;
  StorageOrder order_;
};

} // namespace caffe2

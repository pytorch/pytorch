#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ResizeNearestOp final : public Operator<Context> {
 public:
  ResizeNearestOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        width_scale_(1),
        height_scale_(1),
        order_(StorageOrder::NCHW) {
    if (HasArgument("width_scale")) {
      width_scale_ = static_cast<T>(
          OperatorBase::GetSingleArgument<float>("width_scale", 1));
    }
    if (HasArgument("height_scale")) {
      height_scale_ = static_cast<T>(
          OperatorBase::GetSingleArgument<float>("height_scale", 1));
    }
    if (HasArgument("order")) {
      order_ = StringToStorageOrder(
          OperatorBase::GetSingleArgument<string>("order", "NCHW"));
    }
    CAFFE_ENFORCE_GT(width_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

  bool RunOnDeviceWithOrderNHWC();
  bool RunOnDeviceWithOrderNCHW();

 protected:
  T width_scale_;
  T height_scale_;
  StorageOrder order_;
};

template <typename T, class Context>
class ResizeNearestGradientOp final : public Operator<Context> {
 public:
  ResizeNearestGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        width_scale_(1),
        height_scale_(1),
        order_(StorageOrder::NCHW) {
    width_scale_ = static_cast<T>(
        OperatorBase::GetSingleArgument<float>("width_scale", 1));
    height_scale_ = static_cast<T>(
        OperatorBase::GetSingleArgument<float>("height_scale", 1));
    order_ = StringToStorageOrder(
        OperatorBase::GetSingleArgument<string>("order", "NCHW"));
    CAFFE_ENFORCE_GT(width_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

  bool RunOnDeviceWithOrderNHWC();
  bool RunOnDeviceWithOrderNCHW();

 protected:
  T width_scale_;
  T height_scale_;
  StorageOrder order_;
};

} // namespace caffe2

#pragma once

#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(ResizeNearest3D);


namespace caffe2 {

template <typename T, class Context>
class ResizeNearest3DOp final : public Operator<Context> {
public:
 template <class... Args>
 explicit ResizeNearest3DOp(Args&&... args)
     : Operator<Context>(std::forward<Args>(args)...),
      temporal_scale_(1),
      height_scale_(1),
      width_scale_(1),
      order_(StringToStorageOrder(
        this->template GetSingleArgument<std::string>("order", "NCHW"))) {
    if (HasArgument("temporal_scale")) {
     temporal_scale_ = static_cast<T>(
         this->template GetSingleArgument<float>("temporal_scale", 1));
    }
    if (HasArgument("height_scale")) {
     height_scale_ = static_cast<T>(
         this->template GetSingleArgument<float>("height_scale", 1));
    }
    if (HasArgument("width_scale")) {
     width_scale_ = static_cast<T>(
         this->template GetSingleArgument<float>("width_scale", 1));
    }

    CAFFE_ENFORCE_GT(temporal_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);
    CAFFE_ENFORCE_GT(width_scale_, 0);

    CAFFE_ENFORCE(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
  bool RunOnDeviceWithOrderNCHW();

 protected:
  T temporal_scale_;
  T height_scale_;
  T width_scale_;
  StorageOrder order_;
};

template <typename T, class Context>
class ResizeNearest3DGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit ResizeNearest3DGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        temporal_scale_(1),
        height_scale_(1),
        width_scale_(1),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))) {
    temporal_scale_ = static_cast<T>(
        this->template GetSingleArgument<float>("temporal_scale", 1));
    height_scale_ = static_cast<T>(
        this->template GetSingleArgument<float>("height_scale", 1));
    width_scale_ = static_cast<T>(
        this->template GetSingleArgument<float>("width_scale", 1));

    CAFFE_ENFORCE_GT(temporal_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);
    CAFFE_ENFORCE_GT(width_scale_, 0);

    CAFFE_ENFORCE(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
  bool RunOnDeviceWithOrderNCHW();

 protected:
  T temporal_scale_;
  T height_scale_;
  T width_scale_;
  StorageOrder order_;
};

} // namespace caffe2

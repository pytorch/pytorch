
#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ResizeNearestOp final : public Operator<Context> {
 public:
  ResizeNearestOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), width_scale_(1), height_scale_(1) {
    if (HasArgument("width_scale")) {
      width_scale_ = static_cast<T>(
          this->template GetSingleArgument<float>("width_scale", 1));
    }
    if (HasArgument("height_scale")) {
      height_scale_ = static_cast<T>(
          this->template GetSingleArgument<float>("height_scale", 1));
    }
    if (InputSize() == 2) {
      const auto& scales = Input(1);
      CAFFE_ENFORCE_EQ(scales.ndim(), 1);
      CAFFE_ENFORCE_EQ(scales.size(), 4);
      const float* scales_data = scales.template data<float>();
      height_scale_ = scales_data[2];
      width_scale_ = scales_data[3];
    }
    CAFFE_ENFORCE_GT(width_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  T width_scale_;
  T height_scale_;
};

template <typename T, class Context>
class ResizeNearestGradientOp final : public Operator<Context> {
 public:
  ResizeNearestGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), width_scale_(1), height_scale_(1) {
    width_scale_ = static_cast<T>(
        this->template GetSingleArgument<float>("width_scale", 1));
    height_scale_ = static_cast<T>(
        this->template GetSingleArgument<float>("height_scale", 1));
    if (InputSize() == 3) {
      const auto& scales = Input(2);
      CAFFE_ENFORCE_EQ(scales.ndim(), 1);
      CAFFE_ENFORCE_EQ(scales.size(), 4);
      const float* scales_data = scales.template data<float>();
      height_scale_ = scales_data[2];
      width_scale_ = scales_data[3];
    }
    CAFFE_ENFORCE_GT(width_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  T width_scale_;
  T height_scale_;
};

} // namespace caffe2

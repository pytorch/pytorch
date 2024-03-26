#pragma once

#include "caffe2/operators/resize_op.h"
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace caffe2 {

using ResizeNearestFP32Op = ResizeNearestOp<float, CPUContext>;

template <typename T>
class ResizeNearestDNNLowPOp final : public DNNLowPOp<T, ResizeNearestFP32Op> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, ResizeNearestFP32Op);

  ResizeNearestDNNLowPOp(const OperatorDef& operator_def, Workspace* ws)
      : BaseType(operator_def, ws),
        width_scale_(this->template GetSingleArgument<float>("width_scale", 1)),
        height_scale_(
            this->template GetSingleArgument<float>("height_scale", 1)) {
    CAFFE_ENFORCE_GT(width_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);

    const auto& order = StringToStorageOrder(
        this->template GetSingleArgument<std::string>("order", "NHWC"));
    CAFFE_ENFORCE_EQ(order, StorageOrder::NHWC);
  }

  bool RunOnDevice() override;

 private:
  float width_scale_;
  float height_scale_;
};

} // namespace caffe2

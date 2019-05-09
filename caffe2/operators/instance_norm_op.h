#ifndef CAFFE2_OPERATORS_INSTANCE_NORM_OP_H_
#define CAFFE2_OPERATORS_INSTANCE_NORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class InstanceNormOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit InstanceNormOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        epsilon_(this->template GetSingleArgument<T>("epsilon", 1e-5f)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(epsilon_ >= 0, "Must pass a nonnegative epsilon.");
  }
  ~InstanceNormOp() {}

  bool RunOnDevice() {
    switch (order_) {
      case StorageOrder::NHWC:
        return RunOnDeviceWithOrderNHWC();
      case StorageOrder::NCHW:
        return RunOnDeviceWithOrderNCHW();
      default:
        CAFFE_THROW("Unknown storage order: ", order_);
    }
  }

  bool RunOnDeviceWithOrderNHWC();
  bool RunOnDeviceWithOrderNCHW();

 protected:
  // parameters
  T epsilon_;
  StorageOrder order_;

  // temp results that get passed to the gradient, but are otherwise stored here
  Tensor mean_{Context::GetDeviceType()};
  Tensor inv_stdev_{Context::GetDeviceType()};

  INPUT_TAGS(INPUT, SCALE, BIAS);
  OUTPUT_TAGS(OUTPUT, MEAN, INV_STDEV);
};

template <typename T, class Context>
class InstanceNormGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit InstanceNormGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        epsilon_(this->template GetSingleArgument<T>("epsilon", 1e-5f)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(epsilon_ >= 0, "Must pass a nonnegative epsilon.");
  }
  ~InstanceNormGradientOp() {}

  bool RunOnDevice() {
    switch (order_) {
      case StorageOrder::NHWC:
        return RunOnDeviceWithOrderNHWC();
      case StorageOrder::NCHW:
        return RunOnDeviceWithOrderNCHW();
      default:
        CAFFE_THROW("Unknown storage order: ", order_);
    }
  }

  bool RunOnDeviceWithOrderNHWC();
  bool RunOnDeviceWithOrderNCHW();

 protected:
  // parameters
  T epsilon_;
  StorageOrder order_;

  // temp results that could get passed through to this gradient, but if not,
  // are stored here
  Tensor mean_;
  Tensor inv_stdev_;

  INPUT_TAGS(INPUT, SCALE, BIAS, OUTPUT_GRAD, MEAN, INV_STDEV);
  OUTPUT_TAGS(INPUT_GRAD, SCALE_GRAD, BIAS_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INSTANCE_NORM_OP_H_

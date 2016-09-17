#ifndef CAFFE2_OPERATORS_INSTANCE_NORM_OP_H_
#define CAFFE2_OPERATORS_INSTANCE_NORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class InstanceNormOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  InstanceNormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(epsilon_ >= 0, "Must pass a nonnegative epsilon.");
  }
  ~InstanceNormOp() {}

  bool RunOnDevice() final {
    switch (order_) {
      case StorageOrder::NHWC:
        return RunOnDeviceWithOrderNHWC();
      case StorageOrder::NCHW:
        return RunOnDeviceWithOrderNCHW();
      default:
        CAFFE_THROW("Unknown storage order: ", order_);
    }
    // To suppress old compiler warnings
    return true;
  }

  bool RunOnDeviceWithOrderNHWC();
  bool RunOnDeviceWithOrderNCHW();

 protected:
  float epsilon_;
  StorageOrder order_;
  Tensor<Context> mean_;
  Tensor<Context> inv_var_;
  INPUT_TAGS(INPUT, SCALE, BIAS);
  OUTPUT_TAGS(OUTPUT, MEAN, INV_VAR);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INSTANCE_NORM_OP_H_

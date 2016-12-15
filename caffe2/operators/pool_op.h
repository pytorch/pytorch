#ifndef CAFFE2_OPERATORS_POOL_OP_H_
#define CAFFE2_OPERATORS_POOL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context, typename PoolType>
class PoolOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  PoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {
    CAFFE_ENFORCE(
        dilation_h_ == 1 && dilation_w_ == 1,
        "Pooling op does not support dilation right now.");
    if (!global_pooling_) {
      CAFFE_ENFORCE(
          pad_t_ < kernel_h_ && pad_b_ < kernel_h_ && pad_l_ < kernel_w_ &&
              pad_r_ < kernel_w_,
          "Pad should be smaller than kernel.");
    }
  }
  ~PoolOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  // Input: X
  // Output: Y
};

template <typename T, class Context, class PoolType>
class PoolGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  PoolGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {}
  ~PoolGradientOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  // Input: X, Y, dY
  // Output: dX
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_POOL_OP_H_

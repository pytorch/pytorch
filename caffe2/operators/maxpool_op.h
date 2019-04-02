#ifndef CAFFE2_OPERATORS_MAXPOOL_OP_H_
#define CAFFE2_OPERATORS_MAXPOOL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"
#include "glog/logging.h"

namespace caffe2 {

// MaxPool will produce the max values as well as the indices of the original
// input that leads to the max value. Note that the indices are PER IMAGE,
// meaning that if you compute the offset in the original raw data buffer, you
// will need to deal with the number of images and channels accordingly.
template <typename dtype, class DeviceContext>
class MaxPoolOp final : public ConvPoolOpBase<dtype, DeviceContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS;
  MaxPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<dtype, DeviceContext>(operator_def, ws) {}
  ~MaxPoolOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  // Input: X
  // Output: Y, index
  INPUT_OUTPUT_STATS(1, 1, 2, 2);
  DISABLE_COPY_AND_ASSIGN(MaxPoolOp);
};

template <typename dtype, class DeviceContext>
class MaxPoolGradientOp final : public ConvPoolOpBase<dtype, DeviceContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS;
  MaxPoolGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<dtype, DeviceContext>(operator_def, ws) {}
  ~MaxPoolGradientOp() {}

  bool RunOnDevice() override;

  // Input: X, dY, index
  // Output: dX
  INPUT_OUTPUT_STATS(3, 3, 1, 1);
  DISABLE_COPY_AND_ASSIGN(MaxPoolGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_MAXPOOL_OP_H_

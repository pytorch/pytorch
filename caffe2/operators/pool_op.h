#ifndef CAFFE2_OPERATORS_POOL_OP_H_
#define CAFFE2_OPERATORS_POOL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <typename T, class Context, typename PoolType>
class PoolOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS;
  PoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {}
  ~PoolOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  // Input: X
  // Output: Y
};

template <typename T, class Context, class PoolType>
class PoolGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS;
  PoolGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {}
  ~PoolGradientOp() {}


  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  // Input: X, Y, dY
  // Output: dX
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_POOL_OP_H_

#ifndef CAFFE2_OPERATORS_AVERAGEPOOL_OP_H_
#define CAFFE2_OPERATORS_AVERAGEPOOL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <typename T, class Context>
class AveragePoolOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS;
  AveragePoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {}
  ~AveragePoolOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  // Input: X
  // Output: Y
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(AveragePoolOp);
};

template <typename T, class Context>
class AveragePoolGradientOp final :
    public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS;
  AveragePoolGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {}
  ~AveragePoolGradientOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  // Input: X, Y, Y_grad. Y is in fact not used, but to keep compatibility
  // with CuDNN we keep it here. Definitely not optimal, but probably does not
  // hurt that much.
  // Output: X_grad
  INPUT_OUTPUT_STATS(3, 3, 1, 1);
  DISABLE_COPY_AND_ASSIGN(AveragePoolGradientOp);
};


}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_AVERAGEPOOL_OP_H_

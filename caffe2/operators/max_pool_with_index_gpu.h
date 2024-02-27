#pragma once

#include <cfloat>
#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/pool_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

class MaxPoolWithIndexOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext);
  MaxPoolWithIndexOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws) {}
  ~MaxPoolWithIndexOp() {}

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override;

  // Input: X
  // Output: Y, mask
};

class MaxPoolWithIndexGradientOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext);
  MaxPoolWithIndexGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws) {}
  ~MaxPoolWithIndexGradientOp() {}

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override;

  // Input: X, dY, mask
  // Output: dX
};

}; // namespace caffe2

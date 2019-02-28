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

class MaxPoolWithIndexOp final : public ConvPoolOpBase<CUDAContext, true> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext, true);
  template<class... Args>
  explicit MaxPoolWithIndexOp(Args&&... args)
      : ConvPoolOpBase<CUDAContext, true>(std::forward<Args>(args)...) {}
  ~MaxPoolWithIndexOp() {}

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override;

  // Input: X
  // Output: Y, mask
};

class MaxPoolWithIndexGradientOp final : public ConvPoolOpBase<CUDAContext, true> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext, true);
  template<class... Args>
  explicit MaxPoolWithIndexGradientOp(Args&&... args)
      : ConvPoolOpBase<CUDAContext, true>(std::forward<Args>(args)...) {}
  ~MaxPoolWithIndexGradientOp() {}

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override;

  // Input: X, dY, mask
  // Output: dX
};

}; // namespace caffe2

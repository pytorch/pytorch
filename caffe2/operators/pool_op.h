/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CAFFE2_OPERATORS_POOL_OP_H_
#define CAFFE2_OPERATORS_POOL_OP_H_

#include "caffe2/core/common_omp.h"
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
    for (int i = 0; i < kernel_.size(); ++i) {
      CAFFE_ENFORCE(
          dilation_[i] == 1, "Pooling op does not support dilation right now.");
    }
    if (!global_pooling_) {
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE(
            pads_[i] < kernel_[i] && pads_[i + kernel_.size()] < kernel_[i],
            "Pad should be smaller than kernel.");
      }
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

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
  InstanceNormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<T>("epsilon", 1e-5f)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
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
  Tensor<Context> mean_;
  Tensor<Context> inv_stdev_;

  INPUT_TAGS(INPUT, SCALE, BIAS);
  OUTPUT_TAGS(OUTPUT, MEAN, INV_STDEV);
};

template <typename T, class Context>
class InstanceNormGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  InstanceNormGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<T>("epsilon", 1e-5f)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
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
  Tensor<Context> mean_;
  Tensor<Context> inv_stdev_;

  INPUT_TAGS(INPUT, SCALE, BIAS, OUTPUT_GRAD, MEAN, INV_STDEV);
  OUTPUT_TAGS(INPUT_GRAD, SCALE_GRAD, BIAS_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INSTANCE_NORM_OP_H_

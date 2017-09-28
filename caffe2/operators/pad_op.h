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

#ifndef CAFFE2_OPERATORS_PAD_OP_H_
#define CAFFE2_OPERATORS_PAD_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Padding mode similar to numpy.
enum class PadMode {
  CONSTANT = 0, // pad constant values, with string "constant"
  REFLECT = 1, // pads with reflect values, with string "reflect"
  EDGE = 2, // pads with the edge values, with string "edge"
};

PadMode StringToPadMode(const string&);

template <typename T, class Context>
class PadImageOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  PadImageOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws),
        mode_(StringToPadMode(
            OperatorBase::GetSingleArgument<string>("mode", "constant"))),
        value_(static_cast<T>(
            OperatorBase::GetSingleArgument<float>("value", 0.0))) {
    CAFFE_ENFORCE(
        legacy_pad_ == LegacyPadding::NOTSET,
        "Padding layer only supports explicit pad values.");
    CAFFE_ENFORCE(
        dilation_h() == 1 && dilation_w() == 1,
        "Pooling op does not support dilation right now.");
    CAFFE_ENFORCE(
        stride_h() == 1 && stride_w() == 1,
        "Pooling op does not support stride right now.");
    // Pad op does not use kernel sizes, so we set it to 1 for computing the
    // output size.
    kernel_.assign(pads_.size() / 2, 1);
  }
  ~PadImageOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  static std::vector<TensorShape> PadTensorInference(
      const OperatorDef& def,
      const vector<TensorShape>& in);

 private:
  PadMode mode_;
  T value_;

  // Input: X
  // Output: Y
};

template <typename T, class Context>
class PadImageGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  PadImageGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws),
        mode_(StringToPadMode(
            OperatorBase::GetSingleArgument<string>("mode", "constant"))) {
    CAFFE_ENFORCE(
        legacy_pad_ == LegacyPadding::NOTSET,
        "Padding layer only supports explicit pad values.");
    CAFFE_ENFORCE(
        dilation_h() == 1 && dilation_w() == 1,
        "Pooling op does not support dilation right now.");
    // Pad op does not use kernel sizes, so we set it to 1 for computing the
    // output size.
    kernel_.assign(pads_.size() / 2, 1);
  }
  ~PadImageGradientOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  PadMode mode_;
  // Input: dY
  // Output: dX
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_PAD_OP_H_

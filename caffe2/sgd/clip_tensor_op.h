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

#ifndef CAFFE2_OPERATORS_CLIP_TENSOR_OP_H_
#define CAFFE2_OPERATORS_CLIP_TENSOR_OP_H_

#include <vector>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename Context>
class ClipTensorByScalingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ClipTensorByScalingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    threshold_ = OperatorBase::GetSingleArgument<float>("threshold", 0.0);
    CAFFE_ENFORCE_GT(threshold_, 0, "Threshold must be greater than 0");
  }

  bool RunOnDevice() override {
    const auto& input_tensor = Input(0);
    CAFFE_ENFORCE_GT(input_tensor.size(), 0);
    const auto& val = Input(1);
    CAFFE_ENFORCE_EQ(val.size(), 1);

    const auto* input_tensor_data = input_tensor.template data<float>();
    const auto* val_data = val.template data<float>();

    auto* clipped = Output(0);
    clipped->ResizeLike(input_tensor);
    float* clipped_tensor_data = clipped->template mutable_data<float>();

    if (*val_data > threshold_) {
      float ratio = threshold_ / *val_data;

      math::Scale<float, Context>(
          clipped->size(),
          ratio,
          input_tensor_data,
          clipped_tensor_data,
          &context_);
    } else {
      if (input_tensor_data != clipped_tensor_data) {
        clipped->CopyFrom(input_tensor, &context_);
      }
    }

    return true;
  }

 private:
  float threshold_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CLIP_TENSOR_OP_H_

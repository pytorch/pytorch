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

#ifndef PS_ROI_POOL_OP_H_
#define PS_ROI_POOL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class PSRoIPoolOp final : public Operator<Context> {
 public:
  PSRoIPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        spatial_scale_(this->template GetSingleArgument<float>(
              "spatial_scale", 1.)),
        group_size_(this->template GetSingleArgument<int>("group_size", 1)),
        output_dim_(this->template GetSingleArgument<int>("output_dim", 1)) {
    DCHECK_GT(spatial_scale_, 0);
    DCHECK_GT(group_size_, 0);
    pooled_height_ = group_size_;
    pooled_width_ = group_size_;
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
   float spatial_scale_;
   int group_size_;
   int output_dim_;
   int pooled_height_;
   int pooled_width_;
   int channels_;
   int height_;
   int width_;
 };

template <typename T, class Context>
class PSRoIPoolGradientOp final : public Operator<Context> {
 public:
  PSRoIPoolGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        spatial_scale_(this->template GetSingleArgument<float>(
              "spatial_scale", 1.)),
        group_size_(this->template GetSingleArgument<int>("group_size", 1)),
        output_dim_(this->template GetSingleArgument<int>("output_dim", 1)) {
    DCHECK_GT(spatial_scale_, 0);
    DCHECK_GT(group_size_, 0);
    pooled_height_ = group_size_;
    pooled_width_ = group_size_;
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
  int group_size_;
  int output_dim_;
  int pooled_height_;
  int pooled_width_;
  int channels_;
  int height_;
  int width_;
};

} // namespace caffe2

#endif // PS_ROI_POOL_OP_H_

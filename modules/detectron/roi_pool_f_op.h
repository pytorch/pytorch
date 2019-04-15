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

#ifndef ROI_POOL_F_OP_H_
#define ROI_POOL_F_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class RoIPoolFOp final : public Operator<Context> {
 public:
  RoIPoolFOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        spatial_scale_(this->template GetSingleArgument<float>(
              "spatial_scale", 1.)),
        pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
        pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)) {
    DCHECK_GT(spatial_scale_, 0);
    DCHECK_GT(pooled_height_, 0);
    DCHECK_GT(pooled_width_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
};

template <typename T, class Context>
class RoIPoolFGradientOp final : public Operator<Context> {
 public:
  RoIPoolFGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        spatial_scale_(this->template GetSingleArgument<float>(
              "spatial_scale", 1.)),
        pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
        pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)) {
    DCHECK_GT(spatial_scale_, 0);
    DCHECK_GT(pooled_height_, 0);
    DCHECK_GT(pooled_width_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
};

} // namespace caffe2

#endif // ROI_POOL_F_OP_H_

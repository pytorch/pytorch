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

#ifndef GROUP_SPATIAL_SOFTMAX_OP_H_
#define GROUP_SPATIAL_SOFTMAX_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class GroupSpatialSoftmaxOp final : public Operator<Context> {
 public:
  GroupSpatialSoftmaxOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_classes_(this->template GetSingleArgument<int>("num_classes", 81)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int num_classes_;
  StorageOrder order_;
};

template <typename T, class Context>
class GroupSpatialSoftmaxGradientOp final : public Operator<Context> {
 public:
  GroupSpatialSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        num_classes_(this->template GetSingleArgument<int>("num_classes", 81)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int num_classes_;
  StorageOrder order_;
  Tensor sum_probs_;
};

} // namespace caffe2

#endif // GROUP_SPATIAL_SOFTMAX_OP_H_

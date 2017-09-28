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

#ifndef SOFTMAX_WITH_LOSS_OP_H_
#define SOFTMAX_WITH_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SoftmaxWithLossOp final : public Operator<Context> {
 public:
  SoftmaxWithLossOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        label_prob_mode_(OperatorBase::GetSingleArgument<int>("label_prob", 0)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float scale_;
  int label_prob_mode_;
  StorageOrder order_;
  int axis_;

  Tensor<Context> losses_; // Per example loss
  Tensor<Context> rowmax_; // per example row max
  Tensor<Context> weights_; // unignored weights
  Tensor<Context> sum_multiplier_; // Vector of ones for summing via dot prod
  Tensor<Context> total_weight_ptr_;
  Tensor<Context> scratch_;
};

template <typename T, class Context>
class SoftmaxWithLossGradientOp final : public Operator<Context> {
 public:
  SoftmaxWithLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        label_prob_mode_(OperatorBase::GetSingleArgument<int>("label_prob", 0)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
        only_loss_(OperatorBase::GetSingleArgument<bool>("only_loss", false)),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float scale_;
  int label_prob_mode_;
  Tensor<Context> sum_multiplier_;
  Tensor<Context> weights_; // unignored weights
  Tensor<Context> total_weight_ptr_;
  StorageOrder order_;
  bool only_loss_;
  int axis_;
  Tensor<Context> scratch_;
};

} // namespace caffe2

#endif // SOFTMAX_WITH_LOSS_OP_H_

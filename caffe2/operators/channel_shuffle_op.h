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

#pragma once
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

template <typename Context>
class ChannelShuffleOp final : public ConvPoolOpBase<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);
  ChannelShuffleOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "ChannelShuffleOp only supports NCHW order");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    const auto C = X.dim32(1);
    CAFFE_ENFORCE(C % this->group_ == 0, "");
    const auto K = C / this->group_;
    const auto S = X.dim32(2) * X.dim32(3);
    const auto G = this->group_;
    for (auto n = 0; n < X.dim32(0); ++n) {
      for (auto g = 0; g < G; ++g) {
        // Scatter the group g block (of size KxS) to output channels
        // g + 0 * G, g + 1 * G, g + 2 * G, g + G * (K - 1) etc.
        math::CopyMatrix<Context>(
            X.itemsize(),
            K,
            S,
            X.template data<float>() + g * K * S + n * C * S,
            S,
            Y->template mutable_data<float>() + g * S + n * C * S,
            G * S,
            &context_,
            X.meta().copy());
      }
    }
    return true;
  }
};

template <typename Context>
class ChannelShuffleGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);
  ChannelShuffleGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "ChannelShuffleOp only supports NCHW order");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& dY = Input(0);
    auto* dX = Output(0);
    dX->ResizeLike(dY);
    const auto C = dY.dim32(1);
    CAFFE_ENFORCE(C % this->group_ == 0, "");
    const auto K = C / this->group_;
    const auto S = dY.dim32(2) * dY.dim32(3);
    const auto G = this->group_;
    for (auto n = 0; n < dY.dim32(0); ++n) {
      for (auto g = 0; g < G; ++g) {
        // Gather the group g block (of size KxS) from output channels
        // g + 0 * G, g + 1 * G, g + 2 * G, g + G * (K - 1) etc.
        math::CopyMatrix<Context>(
            dY.itemsize(),
            K,
            S,
            dY.template data<float>() + g * S + n * C * S,
            G * S,
            dX->template mutable_data<float>() + g * K * S + n * C * S,
            S,
            &context_,
            dY.meta().copy());
      }
    }
    return true;
  }
};
}

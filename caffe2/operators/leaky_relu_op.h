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

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class LeakyReluOp : public Operator<Context> {
 public:
  LeakyReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), alpha_(0) {
    if (HasArgument("alpha")) {
      alpha_ =
          static_cast<T>(OperatorBase::GetSingleArgument<float>("alpha", 0));
    }
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  T alpha_;
};

template <typename T, class Context>
class LeakyReluGradientOp final : public Operator<Context> {
 public:
  LeakyReluGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), alpha_(0) {
    if (HasArgument("alpha")) {
      alpha_ =
          static_cast<T>(OperatorBase::GetSingleArgument<float>("alpha", 0));
    }
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  T alpha_;
};

} // namespace caffe2

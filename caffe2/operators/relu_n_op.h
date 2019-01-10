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

#ifndef CAFFE2_OPERATORS_RELU_N_OP_H_
#define CAFFE2_OPERATORS_RELU_N_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ReluNOp final : public Operator<Context> {
 public:
  ReluNOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        n(OperatorBase::GetSingleArgument<float>("n", 6.0)) {
    CAFFE_ENFORCE_GT(n, 0, "n should be greater than 0");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float n;
};

template <typename T, class Context>
class ReluNGradientOp final : public Operator<Context> {
 public:
  ReluNGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        n(OperatorBase::GetSingleArgument<float>("n", 6.0)) {
    CAFFE_ENFORCE_GT(n, 0, "n should be greater than 0");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // Input: Y, dY; Output: dX
  float n;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RELU_N_OP_H_

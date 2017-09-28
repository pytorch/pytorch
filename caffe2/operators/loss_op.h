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

#ifndef CAFFE2_OPERATORS_LOSS_OP_H_
#define CAFFE2_OPERATORS_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/reduction_ops.h"
#include "caffe2/operators/utility_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// AveragedLoss takes in the input and produces the output loss value as
// the average of the input.
template <typename T, class Context>
class AveragedLoss final : public SumElementsOp<T, Context> {
 public:
  AveragedLoss(const OperatorDef& operator_def, Workspace* ws)
      : SumElementsOp<T, Context>(operator_def, ws, true) {}
  ~AveragedLoss() {}
};

template <typename T, class Context>
class AveragedLossGradient final : public SumElementsGradientOp<T, Context> {
 public:
  AveragedLossGradient(const OperatorDef& operator_def, Workspace* ws)
      : SumElementsGradientOp<T, Context>(operator_def, ws, true) {}
  ~AveragedLossGradient() {}
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOSS_OP_H_

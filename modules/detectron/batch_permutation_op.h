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

#ifndef BATCHPERMUTATION_OP_H_
#define BATCHPERMUTATION_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include <cstring>

namespace caffe2 {

template <typename T, class Context>
class BatchPermutationOp final : public Operator<Context> {
 public:
  BatchPermutationOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto& indices = Input(1);
    auto* Y = Output(0);

    CAFFE_ENFORCE(indices.ndim() == 1, "indices must be 1-d");
    CAFFE_ENFORCE(
      X.dim32(0) == indices.dim32(0),
      "X.dim32(0) must be equal to indices.dim32(0)",
      "(",
      X.dim32(0),
      " vs. ",
      indices.dim32(0),
      ")");

    Y->ResizeLike(X);

    const int N = X.dim32(0);
    const int C = X.dim32(1);
    const int H = X.dim32(2);
    const int W = X.dim32(3);

    const float *src = X.template data<float>();
    float *dst = Y->template mutable_data<float>();

    for (int i = 0; i < N; i++) {
      int idx = indices.template data<int>()[i];

      memcpy(dst + i * C * H * W, src + idx * C * H * W, sizeof(float) * C * H * W);
    }

    return true;
  }
};

template <typename T, class Context>
class BatchPermutationGradientOp final : public Operator<Context> {
 public:
  BatchPermutationGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }
};

} // namespace caffe2

#endif // BATCHPERMUTATION_OP_H_

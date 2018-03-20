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

#include "caffe2/operators/arg_max_op.h"

namespace caffe2 {

template <>
bool RowWiseArgMaxOp<CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* result = Output(0);
  CAFFE_ENFORCE(X.ndim() == 2, "Input should be a 2D tensor");
  const int N = X.dim32(0);
  const int D = X.dim32(1);
  const float* X_data = X.data<float>();
  result->Resize(N, 1);
  int* result_data = result->mutable_data<int>();
  for (int n = 0; n < N; ++n) {
    float mx = X_data[n * D];
    int argmx = n * D;
    for (int d = 1; d < D; ++d) {
      int idx = n * D + d;
      if (X_data[idx] > mx) {
        mx = X_data[idx];
        argmx = idx;
      }
      result_data[n] = argmx - (n * D);
    }
  }
  return true;
}

// RowWiseArgMax
REGISTER_CPU_OPERATOR(RowWiseArgMax, RowWiseArgMaxOp<CPUContext>);
OPERATOR_SCHEMA(RowWiseArgMax)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
    Given a 2D (N X D) input tensor, this operator returns a 2D (N X 1) output
    tensor with with the index of the maximum value in each row. If there are
    duplicate max values in a row the index of the first occurence is returned.
    )DOC")
    .Input(0, "X", "2D (N X D) input tensor")
    .Output(0, "Z", "2D (N X 1) output tensor");

NO_GRADIENT(RowWiseArgMax);
} // namespace caffe2

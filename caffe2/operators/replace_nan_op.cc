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

#include "caffe2/operators/replace_nan_op.h"

namespace caffe2 {

template <>
template <typename T>
void ReplaceNaNOp<CPUContext>::ReplaceNaN(
    const T& value,
    const TIndex size,
    const T* X,
    T* Y) {
  for (TIndex i = 0; i < size; i++) {
    if (std::isnan(X[i])) {
      Y[i] = value;
    } else {
      Y[i] = X[i];
    }
  }
}

REGISTER_CPU_OPERATOR(ReplaceNaN, ReplaceNaNOp<CPUContext>);

OPERATOR_SCHEMA(ReplaceNaN)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Replace the NaN (not a number) element in the input tensor with argument `value`
)DOC")
    .Arg("value (optional)", "the value to replace NaN, the default is 0")
    .Input(0, "input", "Input tensor")
    .Input(1, "output", "Output tensor");

SHOULD_NOT_DO_GRADIENT(ReplaceNaN);

} // namespace caffe2

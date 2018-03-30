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

#include "caffe2/operators/stump_func_op.h"

namespace caffe2 {

template <>
bool StumpFuncOp<float, float, CPUContext>::RunOnDevice() {
  auto& in = Input(0);
  const float* in_data = in.template data<float>();
  auto* out = Output(0);
  out->ResizeLike(in);
  float* out_data = out->template mutable_data<float>();
  for (int i = 0; i < in.size(); i++) {
    out_data[i] = (in_data[i] <= threshold_) ? low_value_ : high_value_;
  }
  return true;
}

REGISTER_CPU_OPERATOR(StumpFunc, StumpFuncOp<float, float, CPUContext>);

OPERATOR_SCHEMA(StumpFunc)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "X", "tensor of float")
    .Output(0, "Y", "tensor of float")
    .SetDoc(R"DOC(
Converts each input element into either high_ or low_value
based on the given threshold.
)DOC");

NO_GRADIENT(StumpFunc);

} // caffe2

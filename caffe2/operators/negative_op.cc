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

#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

struct NegativeCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    EigenVectorMap<T>(y, n) = -ConstEigenVectorMap<T>(x, n);
    // for (int i = 0; i < n; ++i) {
    //  y[i] = -x[i];
    //}
  }
};

REGISTER_CPU_OPERATOR(
    Negative, UnaryElementwiseOp<
        TensorTypes<float, double, int, long>, CPUContext, NegativeCPUFunctor>);

// Input: X, output: Y
OPERATOR_SCHEMA(Negative)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Computes the element-wise negative of the input.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

class GetNegativeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Negative", "",
        vector<string>{GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Negative, GetNegativeGradient);
}  // namespace caffe2

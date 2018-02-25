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

#include "caffe2/operators/math_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct SqrCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Sqr<T, CPUContext>(n, x, y, device_context);
  }
};

REGISTER_CPU_OPERATOR(
    Sqr,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SqrCPUFunctor>);

OPERATOR_SCHEMA(Sqr)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc("Square (x^2) the elements of the input")
    .Input(0, "input", "Input tensor")
    .Output(0, "output", "Squared elements of the input");

class GetSqrGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    Argument scale_arg;
    scale_arg.set_name("scale");
    scale_arg.set_f(2.0);
    return vector<OperatorDef>{CreateOperatorDef(
                                   "Scale",
                                   "",
                                   std::vector<string>{GO(0)},
                                   std::vector<string>{GO(0)},
                                   std::vector<Argument>{scale_arg}),
                               CreateOperatorDef(
                                   "Mul",
                                   "",
                                   std::vector<string>{GO(0), I(0)},
                                   std::vector<string>{GI(0)})};
  }
};
REGISTER_GRADIENT(Sqr, GetSqrGradient);

struct SignCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    for (int i = 0; i < n; ++i) {
      y[i] = (-T(1) * (x[i] < 0)) + (x[i] > 0);
    }
  }
};

REGISTER_CPU_OPERATOR(
    Sign,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SignCPUFunctor>);

OPERATOR_SCHEMA(Sign)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Computes sign for each element of the input: -1, 0 or 1.")
    .IdenticalTypeAndShape();
SHOULD_NOT_DO_GRADIENT(Sign);

} // namespace caffe2

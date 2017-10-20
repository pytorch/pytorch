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

#include "swish_op.h"
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(
    Swish,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        CPUContext,
        SwishCPUFunctor>);
REGISTER_CPU_OPERATOR(SwishGradient, SwishGradientOp<CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(Swish)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Swish takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the swish function, y = x / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");
// Input: X, Y, dY, output: dX
OPERATOR_SCHEMA(SwishGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{2, 0}})
    .SetDoc(R"DOC(
SwishGradient takes X, Y and dY and uses this to update dX according to the
chain rule and derivatives of the swish function.
)DOC");

REGISTER_GRADIENT(Swish, GetSwishGradient);
} // namespace caffe2

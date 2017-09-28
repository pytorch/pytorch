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

#include "caffe2/operators/accumulate_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(Accumulate, AccumulateOp<float, CPUContext>);

OPERATOR_SCHEMA(Accumulate)
  .NumInputs(1)
  .NumOutputs(1)
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
Accumulate operator accumulates the input tensor to the output tensor. If the
output tensor already has the right size, we add to it; otherwise, we first
initialize the output tensor to all zeros, and then do accumulation. Any
further calls to the operator, given that no one else fiddles with the output
in the interim, will do simple accumulations.
Accumulation is done using Axpby operation as shown:
  Y = 1*X + gamma*Y
where X is the input tensor, Y is the output tensor and gamma is the multiplier
argument.
)DOC")
  .Arg("gamma", "(float, default 1.0) Accumulation multiplier")
  .Input(0, "input", "The input tensor that has to be accumulated to the "
         "output tensor. If the output size is not the same as input size, the "
         "output tensor is first reshaped and initialized to zero, and only "
         "then, accumulation is done.")
  .Output(0, "output", "Accumulated output tensor");

SHOULD_NOT_DO_GRADIENT(Accumulate);
}  // namespace caffe2

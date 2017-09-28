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

#include "caffe2/operators/stop_gradient.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(StopGradient, StopGradientOp<CPUContext>);

// TODO(jiayq): Add example to the doc string.
OPERATOR_SCHEMA(StopGradient)
    .NumInputs(1, 1)
    .NumOutputs(1, 1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
StopGradient is a helper operator that does no actual numerical computation,
and in the gradient computation phase stops the gradient from being computed
through it.
)DOC");

NO_GRADIENT(StopGradient);
}  // namespace caffe2

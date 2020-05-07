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

#include "weight_scale_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(WeightScale, WeightScaleOp<float, CPUContext>);
OPERATOR_SCHEMA(WeightScale)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(
Every `stepsize` iterations, multiply the weights by a constant `scale`:
    nw = w * scale
)DOC")
    .Input(0, "w", "Current weights")
    .Input(1, "iter", "Training Iteration")
    .Output(0, "nw", "Updated weights")
    .Arg("stepsize", "Every iteration number to do weight scaling")
    .Arg(
        "upper_bound_iter",
        "After iter passes this bound, do not perform the weight rescaling")
    .Arg("scale", "The multiplicative factor applied to weights.");

SHOULD_NOT_DO_GRADIENT(WeightScale);
} // namespace caffe2

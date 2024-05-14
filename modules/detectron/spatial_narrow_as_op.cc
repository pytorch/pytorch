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

#include "spatial_narrow_as_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SpatialNarrowAs, SpatialNarrowAsOp<CPUContext>);
REGISTER_CPU_OPERATOR(
    SpatialNarrowAsGradient,
    SpatialNarrowAsGradientOp<CPUContext>);

OPERATOR_SCHEMA(SpatialNarrowAs)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Reduces ("narrows") the spatial extent of A to that of B by removing rows and
columns from the bottom and right.
)DOC")
    .Input(
        0,
        "A",
        "3D or 4D input of shape (N, H0, W0) or (N, C, H0, W0).")
    .Input(
        1,
        "B",
        "3D or 4D input of shape (N, H1, W1) or (N, C, H1, W1), where H1 <= H0 "
        "and W1 <= W0.")
    .Output(
        0,
        "C",
        "Sub window of A containing rows [0, H1 - 1] (inclusive) and columns "
        "[0, W1 - 1] (inclusive).");

OPERATOR_SCHEMA(SpatialNarrowAsGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Input(
        0,
        "A",
        "See SpatialNarrowAs.")
    .Input(
        1,
        "B",
        "See SpatialNarrowAs.")
    .Input(
        2,
        "dC",
        "Gradient of forward output 0 (C).")
    .Output(
        0,
        "dA",
        "Gradient of forward input 0 (A)");

class SpatialNarrowAsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SpatialNarrowAsGradient", "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(SpatialNarrowAs, SpatialNarrowAsGradient);

} // namespace caffe2

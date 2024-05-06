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

#include "caffe2/operators/spatial_batch_norm_op.h"

namespace caffe2 {

namespace {

class SpatialBNReluOp : public SpatialBNOp<CPUContext> {
 public:
  SpatialBNReluOp(const OperatorDef& operator_def, Workspace* ws)
      : SpatialBNOp<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    if (!SpatialBNOp<CPUContext>::RunOnDevice()) {
      return false;
    }

    auto* output = Output(0);
    float* output_data = output->template mutable_data<float>();
    for (int i = 0; i < output->size(); ++i) {
      output_data[i] = std::max(0.0f, output_data[i]);
    }
    return true;
  }
};

} // anonymous namespace

OPERATOR_SCHEMA(SpatialBNRelu)
    .NumInputs({5, 7})
    .NumOutputs({1, 5})
    .AllowInplace({{0, 0}, {5, 3}, {6, 4}})
    .EnforceInplace({{3, 1}, {4, 2}});

REGISTER_CPU_OPERATOR(SpatialBNRelu, SpatialBNReluOp);

} // namespace caffe2

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

#include "caffe2/operators/half_float_ops.h"

namespace caffe2 {
OPERATOR_SCHEMA(FloatToHalf)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          vector<TensorShape> out;
          const TensorShape& X = in[0];
          out.push_back(X);
          out[0].set_data_type(TensorProto_DataType_FLOAT16);

          return out;
        });

OPERATOR_SCHEMA(HalfToFloat)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          vector<TensorShape> out;
          const TensorShape& X = in[0];
          out.push_back(X);
          out[0].set_data_type(TensorProto_DataType_FLOAT);

          return out;
        });
OPERATOR_SCHEMA(Float16ConstantFill)
    .NumInputs(0)
    .NumOutputs(1)
    .TensorInferenceFunction(Float16FillerTensorInference)
    .Arg("value", "The value for the elements of the output tensor.")
    .Arg("shape", "The shape of the output tensor.")
    .Output(
        0,
        "output",
        "Output tensor of constant values specified by 'value'");

class GetFloatToHalfGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "HalfToFloat", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(FloatToHalf, GetFloatToHalfGradient);

class GetHalfToFloatGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "FloatToHalf", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(HalfToFloat, GetHalfToFloatGradient);
NO_GRADIENT(Float16ConstantFill);
} // namespace caffe2

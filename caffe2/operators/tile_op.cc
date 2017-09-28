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

#include "caffe2/operators/tile_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Tile, TileOp<CPUContext>);
REGISTER_CPU_OPERATOR(TileGradient, TileGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Tile)
    .NumInputs(1, 3)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          vector<TensorShape> out(1);
          out[0] = TensorShape(in[0]);
          ArgumentHelper helper(def);

          auto tiles = helper.GetSingleArgument<int32_t>("tiles", 1);
          auto axis = helper.GetSingleArgument<int32_t>("axis", 0);
          if (in.size() > 1) {
            // Tile or axis is specified as input; we can't determine
            // the size
            out[0].set_unknown_shape(true);
          } else {
            const auto canonical_axis =
                canonical_axis_index_(axis, out[0].dims().size());
            out[0].set_dims(
                canonical_axis, out[0].dims().Get(canonical_axis) * tiles);
          }
          return out;
        })
    .SetDoc(R"DOC(
Constructs a tensor by tiling a given tensor along a specified axis.

This operation creates a new tensor by replicating the input tensor 'tiles'
times along dimension 'axis'. The output tensor's 'axis'th dimension has
input.dims(axis) * tiles elements, and the values of input are replicated
'tiles' times along the 'axis'th dimension.
For example, tiling [[a b c d]] by tile=2, axis=0 produces
[[a b c d], [a b c d]].
)DOC")
    .Arg("tiles", "Number of replicas")
    .Arg("axis", "Axis to replicate along")
    .Input(0, "data", "The input tensor.")
    .Input(1, "tiles", "(optional) Number of replicas (overrides argument)")
    .Input(2, "axis", "(optional) Axis to replicate along (overrides argument)")
    .Output(
        0,
        "tiled_data",
        "Tensor that will contain input replicated along the given axis.");

OPERATOR_SCHEMA(TileGradient).NumInputs(1, 3).NumOutputs(1);

class GetTileGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // Check whether the tiles/axis information was
    // passed through input arguments
    vector<std::string> g_inputs({GO(0)});
    if (Def().input_size() > 1) {
      g_inputs.push_back(I(1));
    }
    if (Def().input_size() > 2) {
      g_inputs.push_back(I(2));
    }
    return SingleGradientDef(
        "TileGradient", "", g_inputs, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Tile, GetTileGradient);

} // namespace caffe2

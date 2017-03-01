#include "caffe2/operators/tile_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(Tile, TileOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(TileGradient, TileGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Tile)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          vector<TensorShape> out(1);
          out[0] = TensorShape(in[0]);
          ArgumentHelper helper(def);

          auto tiles = helper.GetSingleArgument<int32_t>("tiles", 1);
          auto axis = helper.GetSingleArgument<int32_t>("axis", 0);
          const auto canonical_axis =
              canonical_axis_index_(axis, out[0].dims().size());
          out[0].set_dims(
              canonical_axis, out[0].dims().Get(canonical_axis) * tiles);
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
    .Output(
        0,
        "tiled_data",
        "Tensor that will contain input replicated along the given axis.");

OPERATOR_SCHEMA(TileGradient).NumInputs(1).NumOutputs(1);

class GetTileGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TileGradient", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Tile, GetTileGradient);

} // namespace

} // namespace caffe2

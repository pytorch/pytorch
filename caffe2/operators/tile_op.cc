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
Constructs a tensor by tiling a given tensor along a specified axis. This operation creates a new tensor by replicating the input tensor a number of times specified by the `tiles` argument along the `axis` dimension. The output tensor's `axis` dimension has $(X.dims(axis) * tiles)$ elements.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tile_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Tile",
    ["X", "tiles", "axis"],
    ["Y"]
)

workspace.FeedBlob("X", np.random.randint(10, size=(5,5)))
workspace.FeedBlob("tiles", np.array([5]).astype(np.int32))
workspace.FeedBlob("axis", np.array([1]).astype(np.int32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[9 1 7 1 3]
 [2 3 6 2 5]
 [0 9 2 6 4]
 [5 8 1 5 9]
 [2 0 1 3 7]]
Y:
[[9 1 7 1 3 9 1 7 1 3 9 1 7 1 3 9 1 7 1 3 9 1 7 1 3]
 [2 3 6 2 5 2 3 6 2 5 2 3 6 2 5 2 3 6 2 5 2 3 6 2 5]
 [0 9 2 6 4 0 9 2 6 4 0 9 2 6 4 0 9 2 6 4 0 9 2 6 4]
 [5 8 1 5 9 5 8 1 5 9 5 8 1 5 9 5 8 1 5 9 5 8 1 5 9]
 [2 0 1 3 7 2 0 1 3 7 2 0 1 3 7 2 0 1 3 7 2 0 1 3 7]]

```

</details>

)DOC")
    .Arg("tiles", "(*int*): number of replicas")
    .Arg("axis", "(*int*): axis to replicate along")
    .Input(0, "X", "(*Tensor*): input tensor")
    .Input(
        1,
        "tiles",
        "(*Tensor`<int>`*): [OPTIONAL] number of replicas (overrides `tiles` argument)")
    .Input(
        2,
        "axis",
        "(*Tensor`<int>`*): [OPTIONAL] axis to replicate along (overrides `axis` argument)")
    .Output(0, "Y", "(*Tensor*): output tensor")
    .InheritOnnxSchema();

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

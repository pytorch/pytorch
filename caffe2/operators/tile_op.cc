#include "caffe2/operators/tile_op.h"

#include <string>

namespace caffe2 {

template <>
bool TileOp<CPUContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<
      at::Half,
      std::uint8_t,
      std::int32_t,
      std::int64_t,
      float,
      double,
      std::string>>::call(this, Input(0));
}

template <>
template <>
bool TileOp<CPUContext>::DoRunWithType<std::string>() {
  if (InputSize() > 1) {
    // We potentially have tiles and/or axis specified as inputs
    // as well. We will check for them in that order. In other words:
    // InputSize() == 2: tiles is specified
    // InputSize() == 3: tiles is specified and axis.
    // Anything specified as input will override the arguments
    CAFFE_ENFORCE(
        Input(1).dim() == 1 && Input(1).numel() == 1,
        "Input `tiles` should be a vector of size 1.");
    tiles_ = GetArgFromTensor(Input(1));

    // Because of a bug in original code, temporarily adds this part to keep
    // backward compatibility.
    // TODO(yangxm): Remove this part when prod runtime upgraded with fixed
    // model config.
    if (Input(1).IsType<std::int64_t>()) {
      axis_ = 0;
    }

    if (InputSize() > 2) {
      CAFFE_ENFORCE(
          Input(2).dim() == 1 && Input(2).numel() == 1,
          "Input `axis` should be a vector of size 1.");
      axis_ = GetArgFromTensor(Input(2));
    } else {
      CAFFE_ENFORCE(
          OperatorBase::HasArgument("axis"),
          "Argument `axis` is missing and was not specified as input.");
    }
  } else {
    CAFFE_ENFORCE(
        OperatorBase::HasArgument("tiles"),
        "Argument `tiles` is missing and was not specified as input.");
    CAFFE_ENFORCE(
        OperatorBase::HasArgument("axis"),
        "Argument `axis` is missing and was not specified as input.");
  }

  const auto& X = Input(0);
  auto* Y = Output(0);
  const int axis = X.canonical_axis_index(axis_);

  // reshape output to be input tiled along the axis
  std::vector<std::int64_t> Y_dims = X.sizes().vec();
  Y_dims[axis] *= tiles_;
  Y->Resize(Y_dims);

  // size up to (and not including) axis
  const int outer_size = X.size_to_dim(axis);
  // size from axis up
  const int inner_size = X.size_from_dim(axis);

  const TypeMeta meta = X.dtype();
  const int item_size = X.itemsize();
  const char* X_ptr = reinterpret_cast<const char*>(X.raw_data());
  char* Y_ptr = reinterpret_cast<char*>(Y->raw_mutable_data(meta));
  for (int i = 0; i < outer_size; ++i) {
    for (int t = 0; t < tiles_; ++t) {
      context_.CopyItemsSameDevice(meta, inner_size, X_ptr, Y_ptr);
      Y_ptr += inner_size * item_size;
    }
    X_ptr += inner_size * item_size;
  }
  return true;
}

REGISTER_CPU_OPERATOR(Tile, TileOp<CPUContext>);
REGISTER_CPU_OPERATOR(TileGradient, TileGradientOp<CPUContext>);

OPERATOR_SCHEMA(Tile)
    .NumInputs(1, 3)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const std::vector<TensorShape>& in) {
      std::vector<TensorShape> out(1);
      out[0] = TensorShape(in[0]);
      ArgumentHelper helper(def);
      const std::int32_t tiles =
          helper.GetSingleArgument<std::int32_t>("tiles", 1);
      const std::int32_t axis =
          helper.GetSingleArgument<std::int32_t>("axis", 0);
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

namespace {

class GetTileGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    // Check whether the tiles/axis information was
    // passed through input arguments
    std::vector<std::string> g_inputs({GO(0)});
    if (Def().input_size() > 1) {
      g_inputs.push_back(I(1));
    }
    if (Def().input_size() > 2) {
      g_inputs.push_back(I(2));
    }
    return SingleGradientDef(
        "TileGradient", "", g_inputs, std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Tile, GetTileGradient);

} // namespace caffe2

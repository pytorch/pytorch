#include "caffe2/operators/expand_squeeze_dims_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(ExpandDims, ExpandDimsOp<CPUContext>);
REGISTER_CPU_OPERATOR(Squeeze, SqueezeOp<CPUContext>);

OPERATOR_SCHEMA(ExpandDims)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto dims = helper.template GetRepeatedArgument<int>("dims");
      auto originalSize = dims.size();
      CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");

      std::sort(dims.begin(), dims.end());
      dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
      if (dims.size() < originalSize) {
        LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
      }

      CAFFE_ENFORCE(dims.front() >= 0, "Dimension ids must be non-negative.");
      CAFFE_ENFORCE_GE(
          in[0].dims_size() + dims.size(),
          dims.back() + 1,
          "Input needs at least ",
          (1 + dims.back() - dims.size()),
          " dimensions given `dims`.");

      vector<TensorShape> out(1);

      int cur_pos = 0;
      int idx = 0;
      for (const auto new_dim : dims) {
        for (int i = cur_pos; i < new_dim; i++) {
          out[0].add_dims(in[0].dims(idx++));
        }
        out[0].add_dims(1);
        cur_pos = new_dim + 1;
      }
      for (; idx < in[0].dims_size(); idx++) {
        out[0].add_dims(in[0].dims(idx));
      }
      out[0].set_data_type(in[0].data_type());
      return out;
    })
    .SetDoc(R"DOC(
The *ExpandDims* op inserts single-dimensional entries into the shape of the input tensor *data,* and produces a single output tensor *expanded*. The op also takes an argument *dims* with a list of dimensions for where to add the single dimensional entries. If the same blob is provided as input and output, the operation is copy-free. This is the exact inverse operation of *Squeeze*.

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/expand_squeeze_dims_op.h
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/expand_squeeze_dims_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "ExpandDims",
    ["data"],
    ["expanded"],
    dims=[0,1],
)

workspace.FeedBlob("data", np.zeros((100,100)).astype(np.float32))
print("data.shape:", workspace.FetchBlob("data").shape)

workspace.RunOperatorOnce(op)
print("expanded.shape:", workspace.FetchBlob("expanded").shape)

```

**Result**

```

data.shape: (100, 100)
expanded.shape: (1, 1, 100, 100)

```

</details>



)DOC")
    .Input(0, "data", "Input tensor of data to be operated on.")
    .Output(0, "expanded", "Reshaped tensor with same data as input.")
    .Arg(
        "dims",
        "*(type: [int])* List of dimensions of *data* to add single dimensional entry.")
    .InheritOnnxSchema();

OPERATOR_SCHEMA(Squeeze)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
The *Squeeze* op removes single-dimensional entries from the shape of the input tensor *data,* and produces a single output tensor *squeezed*. The op also takes an argument *dims* with a list of dimensions to squeeze. If the same blob is provided as input and output, the operation is copy-free. This is the exact inverse operation of *ExpandDims* given the same *dims* argument.

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/expand_squeeze_dims_op.h
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/expand_squeeze_dims_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Squeeze",
    ["data"],
    ["squeezed"],
    dims=[0,1],
)

workspace.FeedBlob("data", np.zeros((1,1,100,100)).astype(np.float32))
print("data.shape:", workspace.FetchBlob("data").shape)

workspace.RunOperatorOnce(op)
print("squeezed.shape:", workspace.FetchBlob("squeezed").shape)

```

**Result**

```

data.shape: (1, 1, 100, 100)
squeezed.shape: (100, 100)

```

</details>

)DOC")
    .Input(0, "data", "Input tensor of data to be operated on.")
    .Output(0, "squeezed", "Reshaped tensor with same data as input.")
    .Arg("dims", "*(type: [int])* List of dimensions of *data* to squeeze out.")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto dims = helper.template GetRepeatedArgument<int>("dims");
      auto originalSize = dims.size();
      std::sort(dims.begin(), dims.end());
      dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
      if (dims.size() < originalSize) {
        LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
      }
      CAFFE_ENFORCE(dims.front() >= 0, "Dimension ids must be non-negative.");

      vector<TensorShape> out(1);
      std::vector<int> newDims =
          SqueezeOp<CPUContext>::ComputeDims(GetDimsVector(in[0]), dims);
      out[0] = CreateTensorShape(newDims, in[0].data_type());
      return out;
    })
    .InheritOnnxSchema();

class GetSqueezeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ExpandDims", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Squeeze, GetSqueezeGradient);

class GetExpandDimsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Squeeze", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ExpandDims, GetExpandDimsGradient);
}

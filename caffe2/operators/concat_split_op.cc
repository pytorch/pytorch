#include "caffe2/operators/concat_split_op.h"

namespace caffe2 {
namespace {
std::pair<std::vector<DeviceOption>, std::vector<DeviceOption>> splitOpDevInfer(
    const OperatorDef& def) {
  auto op_device =
      def.has_device_option() ? def.device_option() : DeviceOption();
  vector<DeviceOption> in_dev(def.input_size(), op_device);
  vector<DeviceOption> out_dev(def.output_size(), op_device);

  // If we obtain split from input tensor, then 2nd input's type is always CPU.
  if (def.input_size() == SplitOp<CPUContext>::kSplitOpInputSize) {
    CAFFE_ENFORCE_GT(in_dev.size(), 1);
    in_dev[1] = DeviceOption();
  }
  return std::make_pair(in_dev, out_dev);
}
} // namespace.

REGISTER_CPU_OPERATOR(Split, SplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(SplitByLengths, SplitByLengthsOp<CPUContext>);
OPERATOR_SCHEMA(Split)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .Input(0, "input", "(*Tensor*): tensor to split")
    .Input(
        1,
        "split",
        "(*Tensor`<int>`*): [OPTIONAL] list of output lengths (see also arg `split`)")
    .Arg("axis", "(*int*): axis to split on")
    .Arg("split", "(*Tuple(int)*): length of each output")
    .Arg(
        "order",
        "(*string*): order of dimensions of input and output blobs; either \"NCHW\" or \"NHWC\"")
    .Output(0, "[output_0, output_1, ...]", "(*Tensor*): output tensor")
    .DeviceInferenceFunction(splitOpDevInfer)
    .SetDoc(R"DOC(
Split an `input` tensor into a list of tensors, along the axis specified by the `axis` dimension. The lengths of the split can be specified using argument `split` or optional second input blob to the operator. Otherwise, the tensor is split to equal sized parts.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Split",
    ["input"],
    ["output_0","output_1","output_2"],
    split=(3,2,4),
    axis=0
)

workspace.FeedBlob("input", np.random.randint(10, size=(9)))
print("input:", workspace.FetchBlob("input"))
workspace.RunOperatorOnce(op)
print("output_0:", workspace.FetchBlob("output_0"))
print("output_1:", workspace.FetchBlob("output_1"))
print("output_2:", workspace.FetchBlob("output_2"))

```

**Result**

```

input: [2 2 6 6 6 0 5 7 4]
output_0: [2 2 6]
output_1: [6 6]
output_2: [0 5 7 4]

```

</details>

)DOC")
    .InheritOnnxSchema();

OPERATOR_SCHEMA(SplitByLengths)
    .NumInputs(2)
    .NumOutputs(1, INT_MAX)
    .Input(0, "input", "The tensor to split")
    .Input(1, "legnths", "The tensor `l_i` indicates the logic block of input.")
    .Arg("axis", "Which axis to split on")
    .Arg("order", "Either NHWC or NCWH, will split on C axis, defaults to NCHW")
    .DeviceInferenceFunction([](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);
      // lengths input should be on CPU
      in_dev[1] = DeviceOption();
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(
Split a tensor into a list of tensors, given a lengths input, along the specified
'axis'. If `K` outputs are provided, the op assumes `len(lengths) % K == 0`.
The `input` will be split into `K` parts. Each part of length
`sum(lengths[i*k:i*k+k))`)DOC");

namespace {
OpSchema::Cost CostInferenceForConcat(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  ArgumentHelper helper(def);
  const int axis = helper.HasArgument("axis")
      ? helper.GetSingleArgument<int>("axis", -1)
      : GetDimFromOrderString(
            helper.GetSingleArgument<string>("order", "NCHW"));
  bool add_axis = helper.GetSingleArgument<int>("add_axis", 0) != 0;
  const int canonical_axis = canonical_axis_index_(axis, in[0].dims_size());
  CAFFE_ENFORCE_GT(in.size(), 0);
  vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
  if (add_axis) {
    out_shape.insert(out_shape.begin() + canonical_axis, in.size());
  } else {
    for (int i = 1; i < in.size(); ++i) {
      out_shape[canonical_axis] += in[i].dims(canonical_axis);
    }
  }
  uint64_t nElemRead = 1;
  for (int i = 0; i < in.size(); ++i) {
    nElemRead += nElemFromDim(in[i]);
  }
  int size = 1;
  for (auto& s : out_shape) {
    size *= s;
  }

  struct OpSchema::Cost cost;
  cost.flops = 0;
  cost.bytes_read = nElemRead * sizeof(in[0].data_type());
  cost.bytes_written = size * sizeof(in[0].data_type());
  cost.params_bytes = 0;
  return cost;
}

std::pair<std::vector<DeviceOption>, std::vector<DeviceOption>>
concatOpDevInfer(const OperatorDef& def) {
  auto op_device =
      def.has_device_option() ? def.device_option() : DeviceOption();
  vector<DeviceOption> in_dev(def.input_size(), op_device);
  vector<DeviceOption> out_dev(def.output_size(), op_device);

  // 2nd output's type is always CPU irrespective of op's device option.
  CAFFE_ENFORCE_GT(out_dev.size(), 1);
  out_dev[1] = DeviceOption();
  return std::make_pair(in_dev, out_dev);
}
} // namespace

REGISTER_CPU_OPERATOR(Concat, ConcatOp<CPUContext>);
OPERATOR_SCHEMA(Concat)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2)
    .Arg("axis", "*(type: int; default: -1)* Axis to concatenate on.")
    .Arg(
        "order",
        "*(type: string; default='NCHW')* Order of blob dimensions. Concats on the C dimension.")
    .Arg(
        "add_axis",
        "*(type: int)* Pass non-zero integer to add the axis specified in `axis` to all input tensors.")
    .TensorInferenceFunction(OpSchema::NeedsAllInputShapes([](const OperatorDef&
                                                                  def,
                                                              const vector<
                                                                  TensorShape>&
                                                                  in) {
      ArgumentHelper helper(def);
      const int axis = helper.HasArgument("axis")
          ? helper.GetSingleArgument<int>("axis", -1)
          : GetDimFromOrderString(
                helper.GetSingleArgument<string>("order", "NCHW"));
      bool add_axis = helper.GetSingleArgument<int>("add_axis", 0) != 0;
      const int canonical_axis = canonical_axis_index_(axis, in[0].dims_size());
      CAFFE_ENFORCE_GT(in.size(), 0);
      vector<int> split_shape(1, in.size());
      vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
      if (add_axis) {
        for (int i = 1; i < in.size(); ++i) {
          CAFFE_ENFORCE_EQ(
              in[0].dims().size(),
              in[i].dims().size(),
              "All inputs of Concat should have same dims when add_axis = 1. "
              "Got different sizes for inputs 0 and ",
              i);
          for (int j = 0; j < in[0].dims().size(); ++j) {
            CAFFE_ENFORCE_EQ(
                in[0].dims(j),
                in[i].dims(j),
                "All inputs of Concat should have same dims when add_axis = 1. "
                "Got different dims for inputs 0 and ",
                i,
                ". At dim: ",
                j);
          }
        }
        out_shape.insert(out_shape.begin() + canonical_axis, in.size());
      } else {
        for (int i = 1; i < in.size(); ++i) {
          CAFFE_ENFORCE_EQ(
              in[0].dims().size(),
              in[i].dims().size(),
              "All inputs of Concat should have same dims except "
              "canonical_axis dim that is equal to ",
              canonical_axis,
              "Got different sizes for inputs 0 and ",
              i);
          for (int j = 0; j < in[0].dims().size(); ++j) {
            if (j == canonical_axis) {
              continue;
            }
            CAFFE_ENFORCE_EQ(
                in[0].dims(j),
                in[i].dims(j),
                "All inputs of Concat should have same dims except "
                "canonical_axis dim that is equal to ",
                canonical_axis,
                "Got different dims for inputs 0 and ",
                i,
                ". At dim: ",
                j);
          }
        }

        for (int i = 1; i < in.size(); ++i) {
          out_shape[canonical_axis] += in[i].dims(canonical_axis);
        }
      }
      if (def.output_size() == 1) {
        return vector<TensorShape>{
            CreateTensorShape(out_shape, in[0].data_type())};
      }
      return vector<TensorShape>{
          CreateTensorShape(out_shape, in[0].data_type()),
          CreateTensorShape(split_shape, TensorProto::INT32)};
    }))
    .CostInferenceFunction(CostInferenceForConcat)
    .DeviceInferenceFunction(concatOpDevInfer)
    .SetDoc(R"DOC(
Concatenate a list of tensors into a single tensor. Similar functionality to
Numpy's [concatenate](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html)
function. The `axis` argument specifies what axis along which the arrays will be concatenated.
When set to non-zero (default=0), the `add_axis` argument adds the axis specified in `axis` to
all input tensors.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Concat",
    ["X1",  "X2"],
    ["Y", "split_info"],
    axis=0
)

workspace.FeedBlob("X1", np.array([[1,2],[3,4]]))
workspace.FeedBlob("X2", np.array([[5,6]]))
print("X1:", workspace.FetchBlob("X1"))
print("X2:", workspace.FetchBlob("X2"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
print("split_info:", workspace.FetchBlob("split_info"))

```

**Result**

```

X1: [[1 2]
 [3 4]]
X2: [[5 6]]
Y: [[1 2]
 [3 4]
 [5 6]]
split_info: [2 1]

```

</details>

<details>

<summary> <b>Example 2</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Concat",
    ["X1",  "X2"],
    ["Y", "split_info"],
    add_axis=1,
    axis=3
)

workspace.FeedBlob("X1", np.random.randint(10, size=(1, 1, 5, 5))) // NCHW
workspace.FeedBlob("X2", np.random.randint(10, size=(1, 1, 5, 5))) // NCHW
print("X1:", workspace.FetchBlob("X1"))
print("X2:", workspace.FetchBlob("X2"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
print("split_info:", workspace.FetchBlob("split_info"))

```

**Result**

```

X1: [[[[1 8 3 9 0]
   [6 4 6 5 6]
   [3 9 1 9 9]
   [5 1 0 7 7]
   [9 4 0 0 9]]]]
X2: [[[[7 0 2 6 1]
   [3 9 4 0 3]
   [5 3 8 9 4]
   [3 4 2 1 0]
   [0 8 8 8 1]]]]
Y: [[[[[1 8 3 9 0]
    [7 0 2 6 1]]

   [[6 4 6 5 6]
    [3 9 4 0 3]]

   [[3 9 1 9 9]
    [5 3 8 9 4]]

   [[5 1 0 7 7]
    [3 4 2 1 0]]

   [[9 4 0 0 9]
    [0 8 8 8 1]]]]]
split_info: [1 1]

```

</details>

    )DOC")
    .Input(0, "X1, X2, ...", "*(type: Tensor`<float>`)* List of input tensors.")
    .Output(
        0,
        "concat_result",
        "*(type: Tensor`<float>`)* Concatenated tensor.")
    .Output(
        1,
        "split_info",
        "*(type: Tensor`<int>`)* The dimensions of the inputs.")
    .InheritOnnxSchema();

// Backward compatibility names.
REGISTER_CPU_OPERATOR(DepthSplit, SplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(DepthConcat, ConcatOp<CPUContext>);
OPERATOR_SCHEMA(DepthSplit)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .SetDoc("Backward compatible operator name for Split.");
OPERATOR_SCHEMA(DepthConcat)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2)
    .SetDoc("Backward compatible operator name for Concat.");

class GetSplitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> output_grads;
    for (int i = 0; i < def_.output_size(); ++i) {
      if (!GradOut(i).IsEmpty()) {
        output_grads.push_back(GO(i));
      }
    }
    if (output_grads.empty()) {
      return {};
    }
    return SingleGradientDef(
        "Concat",
        "",
        output_grads,
        vector<string>{GI(0), "_" + GI(0) + "_dims"});
  }
};
REGISTER_GRADIENT(Split, GetSplitGradient);
REGISTER_GRADIENT(DepthSplit, GetSplitGradient);
REGISTER_GRADIENT(SplitByLengths, GetSplitGradient);

class GetConcatGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (GradOut(0).IsEmpty()) {
      return {};
    }
    vector<string> grads;
    for (int i = 0; i < def_.input_size(); ++i) {
      grads.push_back(GI(i));
    }
    return SingleGradientDef("Split", "", vector<string>{GO(0), O(1)}, grads);
  }
};
REGISTER_GRADIENT(Concat, GetConcatGradient);
REGISTER_GRADIENT(DepthConcat, GetConcatGradient);
} // namespace caffe2

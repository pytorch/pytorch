#include "caffe2/core/operator_gradient.h"
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

const char* kBroadcastDoc = R"DOC(
If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of size 1 (a scalar value), or having its shape as a
contiguous subset of the first tensor's shape. The starting of the mutually
equal shape is specified by the argument "axis", and if it is not set, suffix
matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):
```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```
Argument `broadcast=1` needs to be passed to enable broadcasting.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc

)DOC";

const char* kAddExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Add",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([[1,2],[3,4]]))
workspace.FeedBlob("B", np.array([[5,6],[7,8]]))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A:
[[1 2]
 [3 4]]
B:
[[5 6]
 [7 8]]
C:
[[ 6  8]
 [10 12]]

```

</details>

)DOC";
const char* kSubExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sub",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([[10,12],[4,14]]))
workspace.FeedBlob("B", np.array([[5,16],[1,19]]))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A:
[[10 12]
 [ 4 14]]
B:
[[ 5 16]
 [ 1 19]]
C:
[[ 5 -4]
 [ 3 -5]]

```

</details>

)DOC";
const char* kMulExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Mul",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([[1,2],[3,4]]))
workspace.FeedBlob("B", np.array([[5,6],[7,8]]))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A:
[[1 2]
 [3 4]]
B:
[[5 6]
 [7 8]]
C:
[[ 5 12]
 [21 32]]

```

</details>

)DOC";
const char* kDivExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Div",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([[18,8],[2,9]]))
workspace.FeedBlob("B", np.array([[9,2],[3,2]]))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A:
[[18  8]
 [ 2  9]]
B:
[[9 2]
 [3 2]]
C:
[[2 4]
 [0 4]]

```

</details>
)DOC";

std::function<void(OpSchema&)> MathDocGenerator(const char* name, const char* extra) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise binary {name} (with limited broadcast support).
{broadcast_doc}

{extra}
)DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    ReplaceAll(doc, "{extra}", extra);
    schema.SetDoc(doc);
    schema.Arg("broadcast", "*(type: int; default: 0)* Pass 1 to enable broadcasting");
    schema.Arg(
        "axis",
        "*(type: int; default: -1)* Axis to concatenate on.");
    schema.Input(
        0,
        "A",
        "*(type: Tensor`<float>`)* First operand, should share the type with the second operand.");
    schema.Input(
        1,
        "B",
        "*(type: Tensor`<float>`)* Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size as A.");
    schema.Output(0, "C", "*(type: Tensor`<float>`)* Output tensor with same dimensions and type as A");
  };
}

OPERATOR_SCHEMA(Add)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("addition",kAddExample))
    .InheritOnnxSchema("Add");
OPERATOR_SCHEMA(Sub)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("subtraction",kSubExample))
    .InheritOnnxSchema("Sub");
OPERATOR_SCHEMA(Mul)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("multiplication",kMulExample))
    .InheritOnnxSchema("Mul");
OPERATOR_SCHEMA(Div)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("division",kDivExample))
    .InheritOnnxSchema("Div");
OPERATOR_SCHEMA(DivGradient).NumInputs(3).NumOutputs(2).AllowInplace({{0, 0}});

OPERATOR_SCHEMA(SumReduceLike)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
SumReduceLike operator takes 2 tensors as input. It performs reduce sum to the
first input so that the output looks like the second one.
It assumes that the first input
has more dimensions than the second, and the dimensions of the second input is
the contiguous subset of the dimensions of the first.
For example, the following tensor shapes are supported:
```
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 2, 5), shape(B) = (2), with axis=0
```
    )DOC")
    .Arg(
        "axis",
        "If set, defines the starting dimension for reduction. Args `axis` and "
        "`axis_str` cannot be used simultaneously.")
    .Arg(
        "axis_str",
        "If set, it could only be N or C or H or W. `order` arg should also be "
        "provided. It defines the reduction dimensions on NCHW or NHWC. Args "
        "`axis` and `axis_str` cannot be used simultaneously.")
    .Arg("order", "Either NHWC or HCWH")
    .Input(
        0,
        "A",
        "First operand, should share the type with the second operand.")
    .Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.")
    .Output(0, "C", "Result, has same dimensions and type as B");

class GetAddGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (!ArgumentHelper::HasArgument(Def(), "broadcast")) {
      SetDense(0, GO(0));
      SetDense(1, GO(0));
      return vector<OperatorDef>();
    }
    SetDense(0, GO(0));

    return SingleGradientDef(
        "SumReduceLike",
        "",
        vector<string>{GO(0), I(1)},
        vector<string>{GI(1)});
  }
};
REGISTER_GRADIENT(Add, GetAddGradient);

// TODO(jiayq): Although we have Sub gradient implemented, we are still missing
// the Negative unary operator to be implemented.
class GetSubGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (!ArgumentHelper::HasArgument(Def(), "broadcast")) {
      SetDense(0, GO(0));
      return SingleGradientDef(
          "Negative", "", vector<string>{GO(0)}, vector<string>{GI(1)});
    } else {
      SetDense(0, GO(0));
      vector<OperatorDef> grad_ops;
      grad_ops.push_back(CreateOperatorDef(
          "Negative",
          "",
          vector<string>{GO(0)},
          vector<string>{GI(1) + "_autogen_pre_red"}));

      Argument axis, axis_str, order;
      if (ArgumentHelper::HasArgument(Def(), "axis")) {
        axis = GetArgument(Def(), "axis");
      } else {
        axis = MakeArgument<int>("axis", -1);
      }
      if (ArgumentHelper::HasArgument(Def(), "axis_str")) {
        axis_str = GetArgument(Def(), "axis_str");
      } else {
        axis_str = MakeArgument<string>("axis_str", "");
      }
      if (ArgumentHelper::HasArgument(Def(), "order")) {
        order = GetArgument(Def(), "order");
      } else {
        order = MakeArgument<string>("order", "NCHW");
      }
      grad_ops.push_back(CreateOperatorDef(
          "SumReduceLike",
          "",
          vector<string>{GI(1) + "_autogen_pre_red", I(1)},
          vector<string>{GI(1)},
          vector<Argument>{axis, axis_str, order}));

      return grad_ops;
    }
  }
  // Make sure the broadcast argument is not copied over.
  bool CopyArguments() const override {
    return false;
  }
};
REGISTER_GRADIENT(Sub, GetSubGradient);

class GetMulGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(
        Def().input(0) != Def().output(0) && Def().input(1) != Def().output(0),
        "Gradient computation cannot be carried out if Mul uses in-place "
        "computation: ",
        ProtoDebugString(Def()));
    if (!ArgumentHelper::HasArgument(Def(), "broadcast")) {
      return vector<OperatorDef>{
          CreateOperatorDef(
              "Mul", "", vector<string>{GO(0), I(1)}, vector<string>{GI(0)}),
          CreateOperatorDef(
              "Mul", "", vector<string>{GO(0), I(0)}, vector<string>{GI(1)})};
    } else {
      Argument broadcast, axis, axis_str, order;
      if (ArgumentHelper::HasArgument(Def(), "broadcast")) {
        broadcast = GetArgument(Def(), "broadcast");
      } else {
        broadcast = MakeArgument<int>("broadcast", 0);
      }
      if (ArgumentHelper::HasArgument(Def(), "axis")) {
        axis = GetArgument(Def(), "axis");
      } else {
        axis = MakeArgument<int>("axis", -1);
      }
      if (ArgumentHelper::HasArgument(Def(), "axis_str")) {
        axis_str = GetArgument(Def(), "axis_str");
      } else {
        axis_str = MakeArgument<string>("axis_str", "");
      }
      if (ArgumentHelper::HasArgument(Def(), "order")) {
        order = GetArgument(Def(), "order");
      } else {
        order = MakeArgument<string>("order", "NCHW");
      }

      vector<OperatorDef> grad_ops;
      grad_ops.push_back(CreateOperatorDef(
          "Mul",
          "mul_grad_1st_op",
          vector<string>{GO(0), I(1)},
          vector<string>{GI(0)},
          vector<Argument>{broadcast, axis, axis_str, order}));
      grad_ops.push_back(CreateOperatorDef(
          "Mul",
          "mul_gradient_2nd_op",
          vector<string>{GO(0), I(0)},
          vector<string>{GI(1) + "_autogen_pre_red"}));

      grad_ops.push_back(CreateOperatorDef(
          "SumReduceLike",
          "mul_with_broadcast_grad_3",
          vector<string>{GI(1) + "_autogen_pre_red", I(1)},
          vector<string>{GI(1)},
          vector<Argument>{axis, axis_str, order}));

      return grad_ops;
    }
  }

  // Make sure the broadcast argument is not copied over.
  bool CopyArguments() const override {
    return false;
  }
};
REGISTER_GRADIENT(Mul, GetMulGradient);

class GetDivGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(
        !ArgumentHelper::HasArgument(Def(), "broadcast"),
        "Gradient not ready yet for Div with broadcasting.");
    return SingleGradientDef(
        "DivGradient",
        "",
        vector<string>{I(1), O(0), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(Div, GetDivGradient);


const char* kLTExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LT",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A: [ 1  5  2  9 12  3]
B: [ 1  3  4  9 12  8]
C: [False False  True False False  True]

```

</details>
)DOC";

const char* kLEExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LE",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A: [ 1  5  2  9 12  3]
B: [ 1  3  4  9 12  8]
C: [ True False  True  True  True  True]

```

</details>
)DOC";

const char* kGTExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "GT",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A: [ 1  5  2  9 12  3]
B: [ 1  3  4  9 12  8]
C: [False  True False False False False]

```

</details>
)DOC";

const char* kGEExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "GE",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A: [ 1  5  2  9 12  3]
B: [ 1  3  4  9 12  8]
C: [ True  True False  True  True False]

```

</details>
)DOC";

const char* kEQExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "EQ",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A: [ 1  5  2  9 12  3]
B: [ 1  3  4  9 12  8]
C: [ True False False  True  True False]

```

</details>
)DOC";

std::function<void(OpSchema&)> ComparisonDocGenerator(
    const char* name,
    const char* desc,
    const char* extra) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise {desc} comparison **{name}** (with limited broadcast support).

{broadcast_doc}

{extra}
)DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{desc}", desc);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    ReplaceAll(doc, "{extra}", extra);
    schema.SetDoc(doc);
    schema.Arg("broadcast", "*(type: int; default: 0)* Pass 1 to enable broadcasting.");
    schema.Arg(
        "axis",
        "*(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.");
    schema.Input(
        0,
        "A",
        "*(type: Tensor`<bool>`)* First operand, should share the type with the second operand.");
    schema.Input(
        1,
        "B",
        "*(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(0, "C", "*(type: Tensor`<bool>`)* Output tensor with same dimensions as `A`.");
  };
}

<<<<<<< HEAD
#define CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(name, symbol, desc, extra) \
  OPERATOR_SCHEMA(name).NumInputs(2).NumOutputs(1).FillUsing(      \
      ComparisonDocGenerator(symbol, desc, extra));                       \
=======
#define CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(name, symbol, desc)             \
  OPERATOR_SCHEMA(name)                                                        \
      .NumInputs(2)                                                            \
      .NumOutputs(1)                                                           \
      .TensorInferenceFunction(                                                \
          [](const OperatorDef& def, const vector<TensorShape>& in) {          \
            ArgumentHelper helper(def);                                        \
            const auto broadcasted =                                           \
                helper.GetSingleArgument<bool>("broadcast", false);            \
            if (!broadcasted) {                                                \
              CAFFE_ENFORCE_EQ(in[0].dims().size(), in[1].dims().size());      \
              for (int i = 0; i < in[0].dims().size(); ++i) {                  \
                CAFFE_ENFORCE_EQ(in[0].dims(i), in[1].dims(i));                \
              }                                                                \
            }                                                                  \
            auto output_dims =                                                 \
                std::vector<TIndex>(in[0].dims().begin(), in[0].dims().end()); \
            return vector<TensorShape>{                                        \
                CreateTensorShape(output_dims, TensorProto::BOOL)};            \
          })                                                                   \
      .FillUsing(ComparisonDocGenerator(symbol, desc));                        \
>>>>>>> 5596260b9e9b051400e6fcc8b0fad39ee918335e
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(LT, "<", "less than", kLTExample);
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(LE, "<=", "less or equal than", kLEExample);
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(GT, ">", "greater than", kGTExample);
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(GE, ">=", "greater or equal than", kGEExample);
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(EQ, "==", "equality", kEQExample);

const char* kAndExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "And",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5))
workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A:
 [[ True False False]
 [False  True False]
 [False False  True]]
B:
 [[ True False  True]
 [False False False]
 [False False False]]
C:
 [[ True False False]
 [False False False]
 [False False False]]

```

</details>
)DOC";

const char* kOrExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Or",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5))
workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A:
[[False  True  True]
 [False  True  True]
 [ True  True  True]]
B:
[[False  True False]
 [ True  True  True]
 [False  True False]]
C:
[[False  True  True]
 [ True  True  True]
 [ True  True  True]]

```

</details>
)DOC";

const char* kXorExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Xor",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5))
workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("C"))

```

**Result**

```

A:
[[ True  True  True]
 [False False  True]
 [False  True False]]
B:
[[False False False]
 [ True  True  True]
 [False False False]]
C:
[[ True  True  True]
 [ True  True False]
 [False  True False]]

```

</details>
)DOC";

std::function<void(OpSchema&)> LogicalDocGenerator(const char* name, const char* extra) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise logical operation **{name}** (with limited broadcast support).
Both input operands should be of type `bool`.

{broadcast_doc}

{extra}
    )DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    ReplaceAll(doc, "{extra}", extra);
    schema.SetDoc(doc);
    schema.Arg("broadcast", "*(type: int; default: 0)* Pass 1 to enable broadcasting.");
    schema.Arg(
        "axis",
        "*(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.");
    schema.Input(0, "A", "*(type: Tensor`<bool>`)* First operand.");
    schema.Input(
        1,
        "B",
        "*(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(0, "C", "*(type: Tensor`<bool>`)* Output tensor of booleans. Has same dimensions as input `A`.");
  };
}

#define CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(name, symbol, onnx_schema, extra) \
  OPERATOR_SCHEMA(name)                                   \
      .NumInputs(2)                                       \
      .NumOutputs(1)                                      \
      .AllowInplace({{0, 0}})                             \
      .FillUsing(LogicalDocGenerator(symbol,extra))       \
      .InheritOnnxSchema(onnx_schema);                    \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Or, "or", "Or", kOrExample);
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(And, "and", "And", kAndExample);
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Xor, "xor", "Xor", kXorExample);

OPERATOR_SCHEMA(Not)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Performs element-wise negation on input tensor `X`.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Not",
    ["X"],
    ["Y"],
)

workspace.FeedBlob("X", (np.random.rand(3, 3) > 0.5))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[ True False False]
 [False False False]
 [ True  True  True]]
Y:
[[False  True  True]
 [ True  True  True]
 [False False False]]

```

</details>

    )DOC")
    .Input(0, "X", "*(Tensor`<bool>`)* Input tensor.")
    .Output(0, "Y", "*(Tensor`<bool>`)* Negated output tensor.")
    .InheritOnnxSchema("Not");
SHOULD_NOT_DO_GRADIENT(Not);

} // namespace caffe2

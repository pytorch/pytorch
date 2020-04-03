#include "caffe2/operators/elementwise_ops.h"

#include "caffe2/core/operator_gradient.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

namespace {

const char kBroadcastDoc[] = R"DOC(
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

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc

)DOC";

const char kAddExample[] = R"DOC(
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

const char kSubExample[] = R"DOC(
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

const char kMulExample[] = R"DOC(
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

const char kDivExample[] = R"DOC(
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

std::function<void(OpSchema&)> MathDocGenerator(
    const char* name,
    const char* extra) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise binary {name} (with limited broadcast support).
{broadcast_doc}

{extra}
)DOC";
    c10::ReplaceAll(doc, "{name}", name);
    c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    c10::ReplaceAll(doc, "{extra}", extra);
    schema.SetDoc(doc);
    schema.Arg(
        "broadcast", "*(type: int; default: 0)* Pass 1 to enable broadcasting");
    schema.Arg("axis", "*(type: int; default: -1)* Axis to concatenate on.");
    schema.Input(
        0,
        "A",
        "*(type: Tensor`<float>`)* First operand, should share the type with the second operand.");
    schema.Input(
        1,
        "B",
        "*(type: Tensor`<float>`)* Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size as A.");
    schema.Output(
        0,
        "C",
        "*(type: Tensor`<float>`)* Output tensor with same dimensions and type as A.");
  };
}

std::vector<TensorShape> ElementwiseOpShapeInference(
    const OperatorDef& def,
    const std::vector<TensorShape>& in) {
  std::vector<TensorShape> out(1);
  out[0].set_data_type(in[0].data_type());
  ArgumentHelper helper(def);
  const bool broadcast = helper.GetSingleArgument<bool>("broadcast", false);
  if (broadcast) {
    out[0].mutable_dims()->CopyFrom(in[0].dims());
  } else {
    const std::vector<int> A_dims(in[0].dims().begin(), in[0].dims().end());
    const std::vector<int> B_dims(in[1].dims().begin(), in[1].dims().end());
    const std::vector<int> C_dims =
        elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
            A_dims, B_dims);
    for (const int dim : C_dims) {
      out[0].add_dims(dim);
    }
  }
  return out;
}

std::vector<TensorShape> ElementwiseGradientOpShapeInference(
    const OperatorDef& def,
    const std::vector<TensorShape>& in) {
  std::vector<TensorShape> out;
  out.push_back(in.at(1));
  out.push_back(in.at(2));
  return out;
}

} // namespace

OPERATOR_SCHEMA(Add)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .TensorInferenceFunction(ElementwiseOpShapeInference)
    .FillUsing(MathDocGenerator("addition", kAddExample))
    .InheritOnnxSchema();
OPERATOR_SCHEMA(AddGradient)
    .NumInputs(3)
    .NumOutputs(2)
    .TensorInferenceFunction(ElementwiseGradientOpShapeInference)
    .AllowInplace({{0, 0}, {0, 1}});

OPERATOR_SCHEMA(Sub)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .TensorInferenceFunction(ElementwiseOpShapeInference)
    .FillUsing(MathDocGenerator("subtraction", kSubExample))
    .InheritOnnxSchema();
OPERATOR_SCHEMA(SubGradient)
    .NumInputs(3)
    .NumOutputs(2)
    .TensorInferenceFunction(ElementwiseGradientOpShapeInference)
    .AllowInplace({{0, 0}, {0, 1}});

OPERATOR_SCHEMA(Mul)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .TensorInferenceFunction(ElementwiseOpShapeInference)
    .FillUsing(MathDocGenerator("multiplication", kMulExample))
    .InheritOnnxSchema();
OPERATOR_SCHEMA(MulGradient)
    .NumInputs(3)
    .NumOutputs(2)
    .TensorInferenceFunction(ElementwiseGradientOpShapeInference)
    .AllowInplace({{0, 0}, {0, 1}});

OPERATOR_SCHEMA(Div)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .TensorInferenceFunction(ElementwiseOpShapeInference)
    .FillUsing(MathDocGenerator("division", kDivExample))
    .InheritOnnxSchema();
OPERATOR_SCHEMA(DivGradient)
    .NumInputs(3, 4)
    .NumOutputs(2)
    .TensorInferenceFunction(ElementwiseGradientOpShapeInference)
    .AllowInplace({{0, 0}});

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

  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 2, 5), shape(B) = (2), with axis=0
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

const char kLTExample[] = R"DOC(
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

const char kLEExample[] = R"DOC(
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

const char kGTExample[] = R"DOC(
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

const char kGEExample[] = R"DOC(
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

const char kEQExample[] = R"DOC(
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

const char kNEExample[] = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "NE",
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
C: [False  True  True False False  True]
```

</details>
)DOC";

std::function<void(OpSchema&)>
ComparisonDocGenerator(const char* name, const char* desc, const char* extra) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise {desc} comparison **{name}** (with limited broadcast support).

{broadcast_doc}

{extra}
)DOC";
    c10::ReplaceAll(doc, "{name}", name);
    c10::ReplaceAll(doc, "{desc}", desc);
    c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    c10::ReplaceAll(doc, "{extra}", extra);
    schema.SetDoc(doc);
    schema.Arg(
        "broadcast",
        "*(type: int; default: 0)* Pass 1 to enable broadcasting.");
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
    schema.Output(
        0,
        "C",
        "*(type: Tensor`<bool>`)* Output tensor with same dimensions as `A`.");
  };
}

#define CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(name, symbol, desc, extra)   \
  OPERATOR_SCHEMA(name)                                                     \
      .NumInputs(2)                                                         \
      .NumOutputs(1)                                                        \
      .TensorInferenceFunction([](const OperatorDef& def,                   \
                                  const vector<TensorShape>& in) {          \
        ArgumentHelper helper(def);                                         \
        const auto broadcasted =                                            \
            helper.GetSingleArgument<bool>("broadcast", false);             \
        if (!broadcasted) {                                                 \
          CAFFE_ENFORCE_EQ(in[0].dims().size(), in[1].dims().size());       \
          for (int i = 0; i < in[0].dims().size(); ++i) {                   \
            CAFFE_ENFORCE_EQ(in[0].dims(i), in[1].dims(i));                 \
          }                                                                 \
        }                                                                   \
        auto output_dims =                                                  \
            std::vector<int64_t>(in[0].dims().begin(), in[0].dims().end()); \
        return vector<TensorShape>{                                         \
            CreateTensorShape(output_dims, TensorProto::BOOL)};             \
      })                                                                    \
      .FillUsing(ComparisonDocGenerator(symbol, desc, extra));              \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(EQ, "==", "equal to", kEQExample);
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(NE, "!=", "not equal to", kNEExample);
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(LT, "<", "less than", kLTExample);
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(
    LE,
    "<=",
    "less or equal than",
    kLEExample);
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(GT, ">", "greater than", kGTExample);
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(
    GE,
    ">=",
    "greater or equal than",
    kGEExample);

const char kAndExample[] = R"DOC(
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

const char kOrExample[] = R"DOC(
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

const char kXorExample[] = R"DOC(
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

std::function<void(OpSchema&)> LogicalDocGenerator(
    const char* name,
    const char* extra) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise logical operation **{name}** (with limited broadcast support).
Both input operands should be of type `bool`.

{broadcast_doc}

{extra}
    )DOC";
    c10::ReplaceAll(doc, "{name}", name);
    c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    c10::ReplaceAll(doc, "{extra}", extra);
    schema.SetDoc(doc);
    schema.Arg(
        "broadcast",
        "*(type: int; default: 0)* Pass 1 to enable broadcasting.");
    schema.Arg(
        "axis",
        "*(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.");
    schema.Input(0, "A", "*(type: Tensor`<bool>`)* First operand.");
    schema.Input(
        1,
        "B",
        "*(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(
        0,
        "C",
        "*(type: Tensor`<bool>`)* Output tensor of booleans. Has same dimensions as input `A`.");
  };
}

#define CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(name, symbol, onnx_schema, extra) \
  OPERATOR_SCHEMA(name)                                                       \
      .NumInputs(2)                                                           \
      .NumOutputs(1)                                                          \
      .AllowInplace({{0, 0}})                                                 \
      .FillUsing(LogicalDocGenerator(symbol, extra))                          \
      .TensorInferenceFunction(ElementwiseOpShapeInference)                   \
      .InheritOnnxSchema(onnx_schema);                                        \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Or, "or", "Or", kOrExample);
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(And, "and", "And", kAndExample);
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Xor, "xor", "Xor", kXorExample);

#undef CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP

std::function<void(OpSchema&)> BitwiseDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise bitwise operation `{name}` (with limited broadcast support).
Both input operands should be of type `bool`.
{broadcast_doc})DOC";
    c10::ReplaceAll(doc, "{name}", name);
    c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    schema.SetDoc(doc);
    schema.Arg(
        "broadcast",
        "*(type: int; default: 0)* Pass 1 to enable broadcasting.");
    schema.Arg(
        "axis",
        "*(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.");
    schema.Input(0, "A", "*(type: Tensor)* First operand.");
    schema.Input(
        1,
        "B",
        "*(type: Tensor)* Second operand. With broadcasting can be of smaller size than `A`. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(
        0,
        "C",
        "*(type: Tensor)* Output tensor. Has same dimensions as input `A`.");
  };
}

#define CAFFE2_SCHEMA_FOR_BINARY_BITWISE_OP(name, symbol)    \
  OPERATOR_SCHEMA(name)                                      \
      .NumInputs(2)                                          \
      .NumOutputs(1)                                         \
      .AllowInplace({{0, 0}})                                \
      .FillUsing(BitwiseDocGenerator(symbol))                \
      .TensorInferenceFunction(ElementwiseOpShapeInference); \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_BITWISE_OP(BitwiseOr, "bitwise_or");
CAFFE2_SCHEMA_FOR_BINARY_BITWISE_OP(BitwiseAnd, "bitwise_and");
CAFFE2_SCHEMA_FOR_BINARY_BITWISE_OP(BitwiseXor, "bitwise_xor");

#undef CAFFE2_SCHEMA_FOR_BINARY_BITWISE_OP

OPERATOR_SCHEMA(Not)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
Performs element-wise negation on input tensor `X`.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc

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
    .InheritOnnxSchema();
SHOULD_NOT_DO_GRADIENT(Not);

OPERATOR_SCHEMA(Sign)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
Computes sign for each element of the input: -1, 0 or 1.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
"Sign",
["X"],
["Y"],
)

workspace.FeedBlob("X", (np.random.rand(3, 3).astype(np.float32) - np.random.rand(3, 3).astype(np.float32)))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[ 0.02816287  0.22408086 -0.30342305]
[-0.18481976  0.03948995  0.39698976]
[-0.63304734 -0.6919183  -0.31524038]]
Y:
[[ 1.  1. -1.]
[-1.  1.  1.]
[-1. -1. -1.]]

```

</details>

    )DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input data tensor.")
    .Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.");
SHOULD_NOT_DO_GRADIENT(Sign);

} // namespace caffe2

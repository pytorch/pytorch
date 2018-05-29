#include "caffe2/operators/filler_op.h"

namespace caffe2 {

template <>
bool RangeFillOp<float, CPUContext>::Fill(
    TensorCPU* output) {
  float* data = output->mutable_data<float>();
  for (int i = 0; i < output->size(); ++i) {
    data[i] = i;
  }
  return true;
}

template <>
template <typename T>
bool DiagonalFillOp<CPUContext>::FillWithType(TensorCPU* output) {
  VerifyOutputShape(output);
  T value = OperatorBase::GetSingleArgument<T>("value", 0);
  auto* data = output->template mutable_data<T>();
  // first fill everything with 0
  math::Set<T, CPUContext>(output->size(), T(0), data, &context_);
  // then calculate step size for diagonal
  auto step = GetStepSize(output);
  for (TIndex i = 0; i < output->size(); i += step) {
    math::Set<T, CPUContext>(1, value, data, &context_);
    data += step;
  }
  return true;
}

REGISTER_CPU_OPERATOR(UniformFill, UniformFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(UniformIntFill, UniformFillOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(UniqueUniformFill, UniqueUniformFillOp<CPUContext>);
REGISTER_CPU_OPERATOR(ConstantFill, ConstantFillOp<CPUContext>);
REGISTER_CPU_OPERATOR(DiagonalFill, DiagonalFillOp<CPUContext>);
REGISTER_CPU_OPERATOR(GaussianFill, GaussianFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(XavierFill, XavierFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MSRAFill, MSRAFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(RangeFill, RangeFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(LengthsRangeFill, LengthsRangeFillOp<CPUContext>);

OPERATOR_SCHEMA(ConstantFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>)
    .SetDoc(R"DOC(
This operator fills the elements of the output tensor with a constant value
specified by the `value` argument.

- The data type is specified by the `dtype` argument

- Currently, the data types supported are *float*, *int32*, *int64*, and *bool*

- If the `dtype` argument is not provided, the data type of `value` is used

- The output tensor shape is either specified by the `shape` argument or will
match the shape of the input tensor if one is provided (if an input tensor is
provided, a shape argument should not be set)

- Optional additional dimensions can be appended at the end as specified by
`extra_shape` argument

- If `input_as_shape` is set to True, the input should be a 1D tensor
containing the desired output shape (the dimensions specified in `extra_shape`
will also be appended)

When specifying `dtype` argument, use the integer keys from the *DataType* enum
in TensorProto:

```
message TensorProto {
  ...
  enum DataType {
    UNDEFINED = 0;
    FLOAT = 1;  // float
    INT32 = 2;  // int
    BYTE = 3;  // BYTE, when deserialized, is going to be restored as uint8.
    STRING = 4;  // string
    BOOL = 5;  // bool
    UINT8 = 6;  // uint8_t
    INT8 = 7;  // int8_t
    UINT16 = 8;  // uint16_t
    INT16 = 9;  // int16_t
    INT64 = 10;  // int64_t
    FLOAT16 = 12;  // caffe2::__f16, caffe2::float16
    DOUBLE = 13;  // double
  }
```

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "ConstantFill",
    [],
    ["Y"],
    shape=(1,5,5)
)

workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
```

**Result**

```
Y: [[[0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]]]
```
</details>

<details>
<summary> <b>Example 2</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "ConstantFill",
    ["X"],
    ["Y"],
    value=4.0,
    dtype=1,
    extra_shape=(1,2)
)

workspace.FeedBlob("X", (np.random.randint(100, size=(3,3))).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
```

**Result**

```
X: [[86. 30. 84.]
 [34. 51.  9.]
 [29. 86. 59.]]
Y: [[[[4. 4.]]

  [[4. 4.]]

  [[4. 4.]]]


 [[[4. 4.]]

  [[4. 4.]]

  [[4. 4.]]]


 [[[4. 4.]]

  [[4. 4.]]

  [[4. 4.]]]]
```

</details>

)DOC")
    .Arg(
        "value",
        "*(type: primitive; default: 0.0f) value to populate output tensor with.")
    .Arg(
        "dtype",
        "*(type: int)* The data type for the elements of the output tensor. "
        "Strictly must be one of the types from *DataType* enum in TensorProto.")
    .Arg(
        "shape",
        "*(type: int | Tuple(int))* Shape of the output tensor. Cannot pass an "
        "input blob and this arg at the same time.")
    .Arg(
        "extra_shape",
        "*(type: int | Tuple(int))* Additional dimensions appended at the end "
        "of the shape indicated by the input blob. Cannot set this"
        "argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "*(type: int | Tuple(int))* 1D tensor containing the desired output "
        "shape. First input must be in CPU context.")
    .Input(
        0,
        "X",
        "*(type: Tensor)* [OPTIONAL] Input tensor to provide shape information.")
    .Output(
        0,
        "Y",
        "*(type: Tensor)* Output tensor of constant values.");

OPERATOR_SCHEMA(DiagonalFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>)
    .SetDoc(R"DOC(
The operator fills the diagonal elements of the output tensor (>= 2D)
with a constant value specified by the 'value' argument, and others 0. If
number of dimensions of the output tensor is greater than 2, all dimensions
must be equal.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message. If the 'dtype' argument is not provided, the data type of
'value' is used.

The output tensor shape is specified by the 'shape' argument. If the number of
input is 1, the shape will be identical to that of the input at run time with
optional additional dimensions appended at the end as specified by 'extra_shape'
argument. In that case the 'shape' argument should not be set.

If input_as_shape is set to true, then the input should be a 1D tensor
containing the desired output shape (the dimensions specified in extra_shape
will also be appended)

NOTE: Currently, it supports data type of float, int32, int64, and bool.
)DOC")
    .Arg("value", "The value for the elements of the output tensor.")
    .Arg(
        "dtype",
        "The data type for the elements of the output tensor."
        "Strictly must be one of the types from DataType enum in TensorProto.")
    .Arg(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.")
    .Arg(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.")
    .Arg("input_as_shape", "1D tensor containing the desired output shape")
    .Input(0, "input", "Input tensor (optional) to provide shape information.")
    .Output(
        0,
        "output",
        "Output tensor"
        "argument and its type is specified by the 'dtype' argument");

OPERATOR_SCHEMA(UniformFill)
    .NumInputs({0, 1, 3})
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>)
    .SetDoc(R"DOC(
Fill the output tensor with FLOAT samples from uniform distribution [min, max].

The range can be defined either by arguments or input blobs. If the range is
given by input blobs, you also need to give the shape as input. When the range
is given as arguments, this operator enforces min <= max. When the range is
given as inputs, the constraint is not enforced. When MAX < MIN, the first
dimension of the output is set to 0. This behavior is allowed so that
dynamically sampling indices into a dynamically sized tensor is possible.

The shape of the output can be given as argument or input.
)DOC")
    .Arg("min", "minimum value, inclusive")
    .Arg("max", "maximum value, inclusive")
    .Arg("shape", "shape of the output, do not set when input_as_shape=1")
    .Arg(
        "input_as_shape",
        "set to 1 to use the first input as shape. First input must be in CPU context.")
    .Input(
        0,
        "SHAPE",
        "1-D tensor of the shape of the output, "
        "must be used with input_as_shape")
    .Input(1, "MIN", "scalar blob of mininum value")
    .Input(2, "MAX", "scalar blob of maximum value")
    .Output(0, "OUTPUT", "output tensor");
OPERATOR_SCHEMA(UniformIntFill)
    .NumInputs({0, 1, 3})
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>)
    .SetDoc(R"DOC(
Like `UniformFill` but fill with INT32.
)DOC");
OPERATOR_SCHEMA(UniqueUniformFill)
    .NumInputs(0, 2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>)
    .SetDoc(R"DOC(
Fill the output tensor with uniform samples between min and max (inclusive).
If the second input is given, its elements will be excluded from uniform
sampling. Using the second input will require you to provide shape via the first
input.
)DOC")
    .Arg("min", "Minimum value, inclusive")
    .Arg("max", "Maximum value, inclusive")
    .Arg(
        "dtype",
        "The data type for the elements of the output tensor."
        "Strictly must be one of the types from DataType enum in TensorProto."
        "This only supports INT32 and INT64 now. If not set, assume INT32")
    .Arg(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.")
    .Arg(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob. "
        "Cannot set the extra_shape argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "1D tensor containing the desired output shape. First input must be in CPU context.")
    .Input(0, "input", "Input tensor to provide shape information")
    .Input(
        1,
        "avoid",
        "(optional) Avoid elements in this tensor. Elements must be unique.")
    .Output(0, "output", "Output tensor of unique uniform samples");
OPERATOR_SCHEMA(GaussianFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>);
OPERATOR_SCHEMA(XavierFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>);
OPERATOR_SCHEMA(MSRAFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>);
OPERATOR_SCHEMA(RangeFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>);

NO_GRADIENT(UniformFill);
NO_GRADIENT(UniformIntFill);
NO_GRADIENT(UniqueUniformFill);
NO_GRADIENT(ConstantFill);
NO_GRADIENT(DiagonalFill);
NO_GRADIENT(GaussianFill);
NO_GRADIENT(XavierFill);
NO_GRADIENT(MSRAFill);
NO_GRADIENT(RangeFill);

OPERATOR_SCHEMA(LengthsRangeFill)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Convert a length vector to a range sequence. For example, input=[4,3,1], the
output would be [0,1,2,3,0,1,2,0].
)DOC")
    .Input(0, "lengths", "1D tensor of int32 or int64 segment lengths.")
    .Output(
        0,
        "range_sequence",
        "1D tensor whose size is the sum of `lengths`");
NO_GRADIENT(LengthsRangeFill);

}  // namespace caffe2

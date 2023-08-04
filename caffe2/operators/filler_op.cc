#include "caffe2/operators/filler_op.h"

namespace caffe2 {

template <>
bool RangeFillOp<float, CPUContext>::Fill(Tensor* output) {
  float* data = output->template mutable_data<float>();
  for (int i = 0; i < output->numel(); ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    data[i] = i;
  }
  return true;
}

template <>
template <typename T>
bool DiagonalFillOp<CPUContext>::FillWithType(Tensor* output) {
  VerifyOutputShape(output);
  T value = OperatorBase::GetSingleArgument<T>("value", 0);
  auto* data = output->template mutable_data<T>();
  // first fill everything with 0
  math::Set<T, CPUContext>(output->numel(), T(0), data, &context_);
  // then calculate step size for diagonal
  auto step = GetStepSize(output);
  for (int64_t i = 0; i < output->numel(); i += step) {
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
    .NumInputs(0, 2)
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

- If a second input V is passed, fill the output with the first element of V

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
    FLOAT16 = 12;  // at::Half
    DOUBLE = 13;  // double
  }
```

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/filler_op.cc

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
Fill the output tensor with float samples from uniform distribution [`min`, `max`].

- The range can be defined either by arguments or input blobs. `min` and `max` are inclusive.
    - If the range is given by input blobs, you also need to give the shape as input.
    - When the range is given as arguments, this operator enforces min <= max. When the range is given as inputs, the constraint is not enforced.
    - When the range is given as inputs and max < min, the first dimension of the output is set to 0. This behavior is allowed so that dynamically sampling indices into a dynamically sized tensor is possible.
- The shape of the output can be given as argument or input.

Github Links:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op_1 = core.CreateOperator(
    "UniformFill",
    [],
    ["output"],
    min=5.5,
    max=10.5,
    shape=(3,3)
)

op_2 = core.CreateOperator(
    "UniformFill",
    ["shape", "min", "max"],
    ["output"],
    input_as_shape=1
)

// Test arg-based op
workspace.RunOperatorOnce(op_1)
print("output (op_1):\n", workspace.FetchBlob("output"))

// Test input-based op
workspace.ResetWorkspace()
workspace.FeedBlob("shape", np.array([5,5]))
workspace.FeedBlob("min", np.array(13.8, dtype=np.float32))
workspace.FeedBlob("max", np.array(19.3, dtype=np.float32))
workspace.RunOperatorOnce(op_2)
print("output (op_2):\n", workspace.FetchBlob("output"))

```

**Result**

```

output (op_1):
 [[8.894862  8.225005  6.7890406]
 [9.588293  7.1072135 7.7234955]
 [8.210596  6.0202913 9.665462 ]]
output (op_2):
 [[18.965155 15.603871 15.038921 17.14872  18.134571]
 [18.84237  17.845276 19.214737 16.970337 15.494069]
 [18.754795 16.724329 15.311974 16.962536 18.60965 ]
 [15.186268 15.264773 18.73341  19.077969 14.237255]
 [15.917589 15.844325 16.248466 17.006554 17.502048]]

```

</details>

)DOC")
    .Arg("min", "(*float*): minimum value, inclusive")
    .Arg("max", "(*float*): maximum value, inclusive")
    .Arg("shape", "(*Tuple(int)*): shape of the output, do not set when `input_as_shape`=1")
    .Arg(
        "input_as_shape",
        "(*int*): set to 1 to use the first input as shape; `shape` input must be in CPU context")
    .Input(
        0,
        "shape",
        "(*Tensor`<int>`*): 1-D tensor of the shape of the output, must be used with `input_as_shape` argument")
    .Input(1, "min", "(*Tensor`<float>`*): scalar tensor containing minimum value, inclusive")
    .Input(2, "max", "(*Tensor`<float>`*): scalar tensor containing maximum value, inclusive")
    .Output(0, "output", "(*Tensor`<float>`*): filled output tensor");
OPERATOR_SCHEMA(UniformIntFill)
    .NumInputs({0, 1, 3})
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<TensorProto_DataType_INT32>)
    .SetDoc(R"DOC(
Fill the output tensor with int32 samples from uniform distribution [`min`, `max`].

- The range can be defined either by arguments or input blobs. `min` and `max` are inclusive.
    - If the range is given by input blobs, you also need to give the shape as input.
    - When the range is given as arguments, this operator enforces min <= max. When the range is given as inputs, the constraint is not enforced.
    - When the range is given as inputs and max < min, the first dimension of the output is set to 0. This behavior is allowed so that dynamically sampling indices into a dynamically sized tensor is possible.
- The shape of the output can be given as argument or input.

Github Links:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op_1 = core.CreateOperator(
    "UniformIntFill",
    [],
    ["output"],
    min=5,
    max=10,
    shape=(3,3)
)

op_2 = core.CreateOperator(
    "UniformIntFill",
    ["shape", "min", "max"],
    ["output"],
    input_as_shape=1
)

// Test arg-based op
workspace.RunOperatorOnce(op_1)
print("output (op_1):\n", workspace.FetchBlob("output"))

// Test input-based op
workspace.ResetWorkspace()
workspace.FeedBlob("shape", np.array([5,5]))
workspace.FeedBlob("min", np.array(13, dtype=np.int32))
workspace.FeedBlob("max", np.array(19, dtype=np.int32))
workspace.RunOperatorOnce(op_2)
print("output (op_2):\n", workspace.FetchBlob("output"))

```

**Result**

```

output (op_1):
 [[ 6 10  7]
 [ 5 10  6]
 [ 7  5 10]]
output (op_2):
 [[19 13 15 13 13]
 [14 17 14 15 15]
 [17 14 19 13 13]
 [17 18 16 13 18]
 [14 15 16 18 16]]

```

</details>

    )DOC")
    .Arg("min", "(*int*): minimum value, inclusive")
    .Arg("max", "(*int*): maximum value, inclusive")
    .Arg(
        "shape",
        "(*Tuple(int)*): shape of the output, do not set when `input_as_shape`=1")
    .Arg(
        "input_as_shape",
        "(*int*): set to 1 to use the first input as shape; `shape` input must be in CPU context")
    .Input(0, "shape", "(*Tensor`<int>`*): 1-D tensor of the shape of the output, must be used with `input_as_shape` argument")
    .Input(1, "min", "(*Tensor`<int>`*): scalar tensor containing minimum value, inclusive")
    .Input(2, "max", "(*Tensor`<int>`*): scalar tensor containing maximum value, inclusive")
    .Output(0, "output", "(*Tensor`<int>`*): filled output tensor");
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
    .TensorInferenceFunction(FillerTensorInference<>)
    .SetDoc(R"DOC(
This op fills an output tensor with samples drawn from a normal distribution specified by the mean and standard deviation arguments. The output tensor shape is specified by the *shape* argument. However, if *input_as_shape* is set to *true*, then the *input* should be a 1D tensor containing the desired output shape (the dimensions specified in *extra_shape* will also be appended). In this case, the *shape* argument should **not** be set.

*Note: cannot set the shape argument and pass in an input at the same time.*

Github Links:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "GaussianFill",
    [],
    ["out"],
    shape=[3,3],
    mean=2.0,
    std=1.1
)

workspace.RunOperatorOnce(op)
print("Out:\n", workspace.FetchBlob("out"))

```

**Result**

```

Out:
 [[1.2084167  2.3336504  2.827349  ]
 [2.7108908  0.9374752  1.7173369 ]
 [0.03320992 2.1775863  1.0894578 ]]

```

</details>

)DOC")
    .Arg(
        "mean",
        "*(type: float; default: 0.)* Mean of the distribution to draw from.")
    .Arg(
        "std",
        "*(type: float; default: 1.)* Standard deviation of the distribution to draw from.")
    .Arg(
        "shape",
        "*(type: [int])* Desired shape of the *output* tensor.")
    .Arg(
        "extra_shape",
        "*(type: [int])* The additional dimensions appended at the end of the *shape* indicated by the input blob. Cannot set the *extra_shape* argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "*(type: bool; default: False)* set to *True* to use the *input* as shape. First, input must be in CPU context.")
    .Input(
        0,
        "input",
        "(Optional) 1D tensor specifying the shape of the output. Must be used with *input_as_shape=True*")
    .Output(
        0,
        "output",
        "Output tensor of random values drawn from a normal distribution. If the shape argument is set, this is the shape specified, and if the *input* exists and *input_as_shape=True*, it is the shape specified by the *input* tensor.");
OPERATOR_SCHEMA(XavierFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>)
    .SetDoc(R"DOC(
This op fills an output tensor with values sampled from a uniform distribution with the range determined by the desired shape of the output. Rather, than specifying the range of values manually, the novelty of Xavier Fill is that it automatically scales the range of the distribution it draws from based on the size of the desired output tensor. For more information check out the paper [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf). The output tensor shape is specified by the *shape* argument. However, if *input_as_shape* is set to *true*, then the *input* should be a 1D tensor containing the desired output shape (the dimensions specified in *extra_shape* will also be appended). In this case, the *shape* argument should **not** be set.

*Note: Do not set the shape argument and pass in an input at the same time.*

Github Links:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "XavierFill",
    [],
    ["out"],
    shape=[3,3],
)

workspace.RunOperatorOnce(op)
print("Out:\n", workspace.FetchBlob("out"))

```

**Result**

```

Out:
 [[-0.8412168   0.33207083 -0.88418937]
 [ 0.43059897 -0.8340702   0.07781601]
 [ 0.93261135 -0.24542928 -0.3980782 ]]

```

</details>

)DOC")
    .Arg(
        "shape",
        "*(type: [int])* Desired shape of the *output* tensor.")
    .Arg(
        "extra_shape",
        "*(type: [int])* The additional dimensions appended at the end of the *shape* indicated by the input blob. Cannot set the *extra_shape* argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "*(type: bool; default: False)* set to *True* to use the *input* as shape. First, input must be in CPU context.")
    .Input(
        0,
        "input",
        "(Optional) 1D tensor specifying the shape of the output. Must be used with *input_as_shape=True*")
    .Output(
        0,
        "output",
        "Output tensor of random values drawn from an automatically scaled uniform distribution, based on the size of the output tensor. If the shape argument is set, this is the shape specified by the shape argument, and if the *input* exists and *input_as_shape=True*, it is the shape specified by the *input* tensor.");

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
The *LengthsRangeFill* op takes a single input *lengths* and outputs a single tensor *range_sequence*. For each element of *lengths*, the op appends the range(0,lengths) vector to the end of *range_sequence*. For example, if input=[2,4,1], the output would be [0,1,0,1,2,3,0].

Github Links:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LengthsRangeFill",
    ["lengths"],
    ["range_sequence"],
)

workspace.FeedBlob("lengths", np.array([2,4,1]).astype(np.int32))
print("lengths:\n", workspace.FetchBlob("lengths"))

workspace.RunOperatorOnce(op)
print("range_sequence: \n", workspace.FetchBlob("range_sequence"))

```

**Result**

```

lengths:
 [2 4 1]
range_sequence:
 [0 1 0 1 2 3 0]

```

</details>

)DOC")
    .Input(0, "lengths", "1D tensor of int32 or int64 segment lengths.")
    .Output(
        0,
        "range_sequence",
        "1D tensor whose size is the sum of *lengths*");
NO_GRADIENT(LengthsRangeFill);

}  // namespace caffe2

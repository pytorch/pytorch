#include "caffe2/operators/given_tensor_fill_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(GivenTensorFill, GivenTensorFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    GivenTensorDoubleFill,
    GivenTensorFillOp<double, CPUContext>);
REGISTER_CPU_OPERATOR(GivenTensorBoolFill, GivenTensorFillOp<bool, CPUContext>);
REGISTER_CPU_OPERATOR(
    GivenTensorInt16Fill,
    GivenTensorFillOp<int16_t, CPUContext>);
REGISTER_CPU_OPERATOR(GivenTensorIntFill, GivenTensorFillOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(
    GivenTensorInt64Fill,
    GivenTensorFillOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(
    GivenTensorStringFill,
    GivenTensorFillOp<std::string, CPUContext>);

NO_GRADIENT(GivenTensorFill);
NO_GRADIENT(GivenTensorDoubleFill);
NO_GRADIENT(GivenTensorBoolFill);
NO_GRADIENT(GivenTensorInt16Fill);
NO_GRADIENT(GivenTensorIntFill);
NO_GRADIENT(GivenTensorInt64Fill);
NO_GRADIENT(GivenTensorStringFill);

OPERATOR_SCHEMA(GivenTensorFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
This op fills an output tensor with the data specified by the *value* and *dtype* arguments.  The output tensor shape is specified by the *shape* argument. Beware, when using this argument *value* should have a value for every element of the *output*, as missing values will not be initialized automatically. If *input_as_shape* is set to *true*, then the *input* should be a 1D tensor containing the desired output shape (the dimensions specified in *extra_shape* will also be appended). In this case, the *shape* argument should **not** be set.

*Note: Do not set the shape argument and pass in an input at the same time.*

Github Links:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "GivenTensorFill",
    [],
    ["out"],
    values=[1., 2., 3.],
    shape=[3],
)

workspace.RunOperatorOnce(op)
print("Out:\n", workspace.FetchBlob("out"))

```

**Result**

```

Out:
 [1. 2. 3.]

```

</details>

)DOC")
    .Arg(
        "values",
        "*(type depends on dtype, Required=True)* The value of the elements to go in the *output* tensor.",
        true /* required */)
    .Arg(
        "dtype",
        "The data type for the elements of the output tensor. Strictly must be one of the types from DataType enum in TensorProto.")
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
        "Output tensor with desired dimension filled with specified data. If the shape argument is set, this is the shape specified, and if the *input* exists and *input_as_shape=True*, it is the shape specified by the *input* tensor.")
    .TensorInferenceFunction(FillerTensorInference<>);

OPERATOR_SCHEMA(GivenTensorDoubleFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .Arg(
        "values",
        "The value for the elements of the output tensor.",
        true /* required */)
    .Arg(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.")
    .Arg(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "1D tensor containing the desired output shape. First input must be in CPU context.")
    .TensorInferenceFunction(
        FillerTensorInference<TensorProto_DataType_DOUBLE>);

OPERATOR_SCHEMA(GivenTensorBoolFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .Arg(
        "values",
        "The value for the elements of the output tensor.",
        true /* required */)
    .Arg(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.")
    .Arg(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "1D tensor containing the desired output shape. First input must be in CPU context.")
    .TensorInferenceFunction(FillerTensorInference<TensorProto_DataType_BOOL>);

OPERATOR_SCHEMA(GivenTensorInt16Fill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .Arg(
        "values",
        "The value for the elements of the output tensor.",
        true /* required */)
    .Arg(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.")
    .Arg(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "1D tensor containing the desired output shape. First input must be in CPU context.")
    .TensorInferenceFunction(FillerTensorInference<TensorProto_DataType_INT16>);

OPERATOR_SCHEMA(GivenTensorIntFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .Arg(
        "values",
        "The value for the elements of the output tensor.",
        true /* required */)
    .Arg(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.")
    .Arg(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "1D tensor containing the desired output shape. First input must be in CPU context.")
    .TensorInferenceFunction(FillerTensorInference<TensorProto_DataType_INT32>);

OPERATOR_SCHEMA(GivenTensorInt64Fill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .Arg(
        "values",
        "The value for the elements of the output tensor.",
        true /* required */)
    .Arg(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.")
    .Arg(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "1D tensor containing the desired output shape. First input must be in CPU context.")
    .TensorInferenceFunction(FillerTensorInference<TensorProto_DataType_INT64>);

OPERATOR_SCHEMA(GivenTensorStringFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .Arg(
        "values",
        "The value for the elements of the output tensor.",
        true /* required */)
    .Arg(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.")
    .Arg(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "1D tensor containing the desired output shape. First input must be in CPU context.")
    .TensorInferenceFunction(
        FillerTensorInference<TensorProto_DataType_STRING>);

} // namespace caffe2

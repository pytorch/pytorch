#include "caffe2/operators/given_tensor_byte_string_to_uint8_fill_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(
    GivenTensorByteStringToUInt8Fill,
    GivenTensorByteStringToUInt8FillOp<CPUContext>);

NO_GRADIENT(GivenTensorByteStringToUInt8Fill);

OPERATOR_SCHEMA(GivenTensorByteStringToUInt8Fill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
This op fills a uint8 output tensor with the data specified by the *value* argument. The data must previously be serialized as a byte string. The output tensor shape is specified by the *shape* argument. Beware, when using this argument *value* should have a value for every element of the *output*, as missing values will not be initialized automatically. If *input_as_shape* is set to *true*, then the *input* should be a 1D tensor containing the desired output shape (the dimensions specified in *extra_shape* will also be appended). In this case, the *shape* argument should **not** be set.

This op allows us to write uint8 tensors to Protobuf as byte strings and read them back as uint8 tensors in order to avoid the Protobuf uint32_t varint encoding size penalty.

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

val = np.array([1, 2, 3], dtype=np.uint8)
op = core.CreateOperator(
    "GivenTensorByteStringToUInt8Fill",
    [],
    ["out"],
    values=[val.tobytes()],
    shape=val.shape,
)

workspace.RunOperatorOnce(op)
print("Out:\n", workspace.FetchBlob("out"))

```

**Result**

```

Out:
 [1 2 3]

```

</details>

)DOC")
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

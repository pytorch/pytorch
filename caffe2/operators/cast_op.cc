#include "caffe2/operators/cast_op.h"

namespace caffe2 {

template <>
template <typename DstType, typename SrcType>
bool CastOp<CPUContext>::DoRunWithType() {
  auto& input = Input(0);
  auto* output = Output(0);
  output->ResizeLike(input);
  const auto* data = input.template data<SrcType>();
  auto* out = output->template mutable_data<DstType>();
  auto N = input.size();
  for (TIndex i = 0; i < N; ++i) {
    out[i] = static_cast<DstType>(data[i]);
  }
  return true;
}

REGISTER_CPU_OPERATOR(Cast, CastOp<CPUContext>);

OPERATOR_SCHEMA(Cast)
  .NumInputs(1)
  .NumOutputs(1)
  .SetDoc(R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message. If the 'to' argument
is not provided or is not one of the enumerated types in DataType, Caffe2
throws an Enforce error.

NOTE: Casting to and from strings is not supported yet.
)DOC")
  .Arg("to", "The data type to which the elements of the input tensor are cast."
       "Strictly must be one of the types from DataType enum in TensorProto")
  .Input(0, "input", "Input tensor to be cast.")
  .Output(0, "output", "Output tensor with the same shape as input with type "
          "specified by the 'to' argument");

// Some Casts are compatible with gradients, but for now we don't support it
GRADIENT_NOT_IMPLEMENTED_YET(Cast);

}  // namespace caffe2

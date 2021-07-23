#include "caffe2/operators/alias_with_name.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(AliasWithName, AliasWithNameOp<CPUContext>);

OPERATOR_SCHEMA(AliasWithName)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Similar with AliasOp, storing the alias name as operator argument.
)DOC")
    .Arg("name", "name of the aliasing")
    .Arg("is_backward", "weather or not to alias forward or backward")
    .Input(0, "input", "Input tensor whose storage will be shared.")
    .Output(0, "output", "Tensor of same shape as input, sharing its storage.");

} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    AliasWithName,
    "_caffe2::AliasWithName(Tensor input, str name, bool is_backward = False) -> (Tensor output)",
    caffe2::AliasWithNameOp<caffe2::CPUContext>);

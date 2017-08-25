#include "caffe2/operators/do_op.h"

namespace caffe2 {

template <>
bool DoOp<CPUContext>::RunOnDevice() {
  return net_->Run();
}

REGISTER_CPU_OPERATOR(Do, DoOp<CPUContext>);

OPERATOR_SCHEMA(Do)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC(
'Do' control operator, creates a new workspace and executes a subnet in it.
Accepts 'net' argument for a subnet, arguments 'inner_blobs' and 'outer_blobs_idx'
provide a mapping between selected inner blob names and corresponding outer blobs
indices: [0..NumInputs-1] indices correspond to input blobs and [NumInputs..NumOutputs+NumInputs-1] -
output blobs, in the order specified in 'Do' operator definition.
    )DOC")
    .Arg("net", "Subnet with blob bindings")
    .Arg(
        "inner_blobs",
        "List of inner net blob names to bind to outer workspace")
    .Arg(
        "outer_blobs_idx",
        "Indices of corresponding outer workspace blobs, "
        "in order: operator inputs, operator outputs")
    .AllowInplace([](int in, int out) -> bool { return true; });

} // namespace caffe2

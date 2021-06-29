#include "caffe2/core/operator.h"
#include "caffe2/operators/no_default_engine_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(CreateCommonWorld)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a common world for communication operators.
)DOC")
    .Input(0, "kv_handler", "Key/value handler for rendezvous (optional).")
    .Output(0, "comm_world", "A common world for collective operations.")
    .Arg("size", "(int) size of the common world.")
    .Arg("rank", "(int) rank of this node in the common world.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(CloneCommonWorld)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Clones existing common world.
)DOC")
    .Input(0, "existing_comm_world", "Existing common world to clone.")
    .Output(0, "comm_world", "A common world for collective operations.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(DestroyCommonWorld)
    .NumInputs(1)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc("Closes all connections managed by a common world.")
    .Input(0, "common_world", "The common world to be destroyed.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Broadcast)
    .NumInputsOutputs([](int in, int out) {
      return in >= 2 && out == (in - 1);
    })
    .EnforceInplace([](int in, int out) { return (in - 1) == out; })
    .InputsCanCrossDevices()
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
Does a broadcast operation from the root node to every other node. The tensor
on each node should have been pre-created with the same shape and data type.
)DOC")
    .Input(0, "comm_world", "The common world.")
    .Input(1, "X", "A tensor to be broadcasted.")
    .Output(0, "X", "In-place as input 1.")
    .Arg("root", "(int, default 0) the root to run broadcast from.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Reduce)
    .NumInputs(2)
    .NumOutputs(1)
    .InputsCanCrossDevices()
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
Does a reduce operation from every node to the root node. Currently only
Sum is supported.
)DOC")
    .Input(0, "comm_world", "The common world.")
    .Input(1, "X", "A tensor to be reduced.")
    .Output(0, "Y", "The reduced result on root, not set for other nodes.")
    .Arg("root", "(int, default 0) the root to run reduce into.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Allreduce)
    .NumInputsOutputs([](int in, int out) {
      return in >= 2 && out == (in - 1);
    })
    .EnforceInplace([](int in, int out) { return (in - 1) == out; })
    .IdenticalTypeAndShapeOfInput(0)
    .InputsCanCrossDevices()
    .SetDoc(R"DOC(
Does an allreduce operation among the nodes. Currently only Sum is supported.
)DOC")
    .Input(0, "comm_world", "The common world.")
    .Input(1, "X", "A tensor to be allreduced.")
    .Output(0, "Y", "The allreduced tensor, same on all nodes.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ReduceScatter)
    .NumInputsOutputs([](int in, int out) {
      return in >= 2 && out == (in - 1);
    })
    .EnforceInplace([](int in, int out) { return (in - 1) == out; })
    .IdenticalTypeAndShapeOfInput(0)
    .InputsCanCrossDevices()
    .SetDoc(R"DOC(
Does reduce-scatter operation among the nodes. Currently only Sum is supported.
)DOC")
    .Input(0, "comm_world", "The common world.")
    .Input(1, "X", "A tensor to be reduce-scattered.")
    .Output(0, "Y", "The reduced tensor, scattered on all nodes.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Allgather)
    .NumInputs(2, INT_MAX)
    .NumOutputs(1)
    .InputsCanCrossDevices()
    .SetDoc(R"DOC(
Does an allgather operation among the nodes.
)DOC")
    .Input(0, "comm_world", "The common world.")
    .Input(1, "X", "A tensor to be allgathered.")
    .Output(0, "Y", "The allgathered tensor, same on all nodes.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Barrier)
    .NumInputs(1)
    .SetDoc(R"DOC(
Does a barrier operation among the nodes.
)DOC")
    .Input(0, "comm_world", "The common world.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SendTensor)
    .NumInputs({2, 4})
    .NumOutputs(0)
    .SetDoc(R"DOC(
Sends the tensor to another node.
)DOC")
    .Input(0, "comm_world", "The common world.")
    .Input(1, "X", "A tensor to be allgathered.")
    .Input(
        2,
        "dst",
        "An int CPUtensor of size 1 specifying the rank. If "
        "given, this overrides the 'to' argument of the op.")
    .Input(
        3,
        "tag",
        "An int CPUtensor of size 1 specifying the tag to "
        "send the tensor with. This overrides the 'tag' "
        "argument of the op.")
    .Arg("dst", "The rank to send the tensor to.")
    .Arg("tag", "(int) a tag to send the tensor with.")
    .Arg(
        "raw_buffer",
        "(bool) if set, only send the content and assume that the receiver "
        "has already known the tensor's shape and information.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ReceiveTensor)
    .NumInputs({2, 4})
    .NumOutputs(3)
    .EnforceInplace({{1, 0}})
    .AllowInplace({{2, 1}, {3, 2}})
    .SetDoc(R"DOC(
Receives the tensor from another node.
)DOC")
    .Input(0, "comm_world", "The common world.")
    .Input(
        1,
        "Y",
        "In-place output. If raw_buffer is specified, "
        "Y should have pre-allocated data and type..")
    .Input(
        2,
        "src",
        "An int CPUtensor of size 1 specifying the rank. If "
        "given, this overrides the 'from' argument of the op.")
    .Input(
        3,
        "tag",
        "An int CPUtensor of size 1 specifying the tag to "
        "send the tensor with. This overrides the 'tag' "
        "argument of the op.")
    .Output(0, "Y", "The received tensor.")
    .Output(
        1,
        "src",
        "The sender that sent the message as a CPUTensor "
        "of size 1 and of type int.")
    .Output(
        2,
        "tag",
        "The tag that the message is sent with as a CPUTensor "
        "of size 1 and of type int.")
    .Arg("src", "(int) he rank to receive the tensor from.")
    .Arg("tag", "(int) a tag to receive the tensor with.")
    .Arg(
        "raw_buffer",
        "(bool) if set, only send the content and assume that the receiver "
        "has already known the tensor's shape and information.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(CreateCommonWorld);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(CloneCommonWorld);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(DestroyCommonWorld);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Broadcast);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Reduce);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Allgather);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Allreduce);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(ReduceScatter);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Barrier);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(SendTensor);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(ReceiveTensor);

// Communication operators do not have default engines.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(CreateCommonWorld, NoDefaultEngineOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(CloneCommonWorld, NoDefaultEngineOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(DestroyCommonWorld, NoDefaultEngineOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Broadcast, NoDefaultEngineOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Reduce, NoDefaultEngineOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Allgather, NoDefaultEngineOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Allreduce, NoDefaultEngineOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(ReduceScatter, NoDefaultEngineOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Barrier, NoDefaultEngineOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(SendTensor, NoDefaultEngineOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(ReceiveTensor, NoDefaultEngineOp<CPUContext>);

} // namespace caffe2

#include "caffe2/core/operator.h"

namespace caffe2 {
namespace {

OPERATOR_SCHEMA(CreateCommonWorld)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a common world for communication operators.
)DOC")
    .Output(0, "comm_world", "A common world for distributed messaging.");

OPERATOR_SCHEMA(Broadcast)
    .NumInputs(2)
    .NumOutputs(1)
    .EnforceInplace({{1, 0}})
    .SetDoc(R"DOC(
Does a broadcast operation from the root node to every other node.
)DOC")
    .Input(0, "comm_world", "The common world.")
    .Input(1, "X", "A tensor to be broadcasted.")
    .Output(0, "X", "In-place as input 1.")
    .Arg("root", "(int, default 0) the root to run broadcast from.");

OPERATOR_SCHEMA(Reduce)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Does a reduce operation from every node to the root node. Currently only
Sum is supported.
)DOC")
    .Input(0, "comm_world", "The common world.")
    .Input(1, "X", "A tensor to be reduced.")
    .Output(0, "Y", "The reduced result on root, not set for other nodes.")
    .Arg("root", "(int, default 0) the root to run reduce into.");

OPERATOR_SCHEMA(Allreduce)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
Does an allreduce operation among the nodes. Currently only Sum is supported.
)DOC")
    .Input(0, "comm_world", "The common world.")
    .Input(1, "X", "A tensor to be allreduced.")
    .Output(0, "Y", "The allreduced tensor, same on all nodes.");

OPERATOR_SCHEMA(Allgather)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Does an allgather operation among the nodes. Currently only Sum is supported.
)DOC")
    .Input(0, "comm_world", "The common world.")
    .Input(1, "X", "A tensor to be allgathered.")
    .Output(0, "Y", "The allgathered tensor, same on all nodes.");

SHOULD_NOT_DO_GRADIENT(Broadcast);
SHOULD_NOT_DO_GRADIENT(Reduce);
SHOULD_NOT_DO_GRADIENT(Allgather);
SHOULD_NOT_DO_GRADIENT(Allreduce);
} // namespace
} // namespace caffe2

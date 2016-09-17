#include "caffe2/mpi/mpi_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    CreateCommonWorld,
    MPI,
    MPICreateCommonWorldOp<CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(Broadcast, MPI, MPIBroadcastOp<CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(Reduce, MPI, MPIReduceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Allgather,
    MPI,
    MPIAllgatherOp<float, CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Allreduce,
    MPI,
    MPIAllreduceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(SendTensor, MPI, MPISendTensorOp<CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    ReceiveTensor,
    MPI,
    MPIReceiveTensorOp<CPUContext>);

}  // namespace
}  // namespace caffe2

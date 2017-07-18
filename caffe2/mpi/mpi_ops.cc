#include "caffe2/mpi/mpi_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(MPICreateCommonWorld, MPICreateCommonWorldOp<CPUContext>);
REGISTER_CPU_OPERATOR(MPIBroadcast, MPIBroadcastOp<CPUContext>);
REGISTER_CPU_OPERATOR(MPIReduce, MPIReduceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MPIAllgather, MPIAllgatherOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MPIAllreduce, MPIAllreduceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MPISendTensor, MPISendTensorOp<CPUContext>);
REGISTER_CPU_OPERATOR(MPIReceiveTensor, MPIReceiveTensorOp<CPUContext>);

}  // namespace
}  // namespace caffe2

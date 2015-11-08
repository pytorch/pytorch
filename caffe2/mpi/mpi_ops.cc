#include "caffe2/mpi/mpi_ops.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(Broadcast, BroadcastOp<CPUContext>);
REGISTER_CPU_OPERATOR(Allreduce, AllreduceOp<float, CPUContext>);

SHOULD_NOT_DO_GRADIENT(Broadcast);
SHOULD_NOT_DO_GRADIENT(Allreduce);
}  // namespace

}  // namespace caffe2

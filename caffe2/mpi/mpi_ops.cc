#include "caffe2/mpi/mpi_ops.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(Broadcast, BroadcastOp<CPUContext>);
REGISTER_CPU_OPERATOR(Allreduce, AllreduceOp<float, CPUContext>);
}  // namespace

}  // namespace caffe2

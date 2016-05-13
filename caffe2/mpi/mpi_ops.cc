#include "caffe2/mpi/mpi_ops.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(MPIBroadcast, MPIBroadcastOp<CPUContext>);
REGISTER_CPU_OPERATOR(MPIReduce, MPIReduceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MPIAllgather, MPIAllgatherOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MPIAllreduce, MPIAllreduceOp<float, CPUContext>);

OPERATOR_SCHEMA(MPIBroadcast)
    .NumInputs(1).NumOutputs(1).EnforceInplace({{0, 0}});
OPERATOR_SCHEMA(MPIReduce)
    .NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(MPIAllgather)
    .NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(MPIAllreduce)
    .NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});

SHOULD_NOT_DO_GRADIENT(MPIBroadcast);
SHOULD_NOT_DO_GRADIENT(MPIReduce);
SHOULD_NOT_DO_GRADIENT(MPIAllgather);
SHOULD_NOT_DO_GRADIENT(MPIAllreduce);
}  // namespace

}  // namespace caffe2

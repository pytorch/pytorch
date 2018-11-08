#include "caffe2/mpi/mpi_ops.h"

namespace caffe2 {

OPERATOR_SCHEMA(MPICreateCommonWorld)
  .NumInputs(0)
  .NumOutputs(1);
OPERATOR_SCHEMA(MPIBroadcast)
  .NumInputs(2)
  .NumOutputs(1)
  .EnforceInplace({{1, 0}});
OPERATOR_SCHEMA(MPIReduce)
  .NumInputs(2)
  .NumOutputs(1);
OPERATOR_SCHEMA(MPIAllgather)
  .NumInputs(2)
  .NumOutputs(1);
OPERATOR_SCHEMA(MPIAllreduce)
  .NumInputs(2)
  .NumOutputs(1)
  .AllowInplace({{1, 0}});
OPERATOR_SCHEMA(MPISendTensor);
OPERATOR_SCHEMA(MPIReceiveTensor);

REGISTER_CPU_OPERATOR(MPICreateCommonWorld, MPICreateCommonWorldOp<CPUContext>);
REGISTER_CPU_OPERATOR(MPIBroadcast, MPIBroadcastOp<CPUContext>);
REGISTER_CPU_OPERATOR(MPIReduce, MPIReduceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MPIAllgather, MPIAllgatherOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MPIAllreduce, MPIAllreduceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MPISendTensor, MPISendTensorOp<CPUContext>);
REGISTER_CPU_OPERATOR(MPIReceiveTensor, MPIReceiveTensorOp<CPUContext>);

}  // namespace caffe2

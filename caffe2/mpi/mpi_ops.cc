/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/mpi/mpi_ops.h"

namespace caffe2 {

OPERATOR_SCHEMA(MPICreateCommonWorld);
OPERATOR_SCHEMA(MPIBroadcast);
OPERATOR_SCHEMA(MPIReduce);
OPERATOR_SCHEMA(MPIAllgather);
OPERATOR_SCHEMA(MPIAllreduce);
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

#ifndef CAFFE2_MPI_MPI_OPS_H_
#define CAFFE2_MPI_MPI_OPS_H_

#include <mpi.h>

#include "caffe2/core/operator.h"
#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

// BroadcastOp does Broadcast using MPI.
template <typename dtype, class DeviceContext>
class BroadcastOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  BroadcastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)) {
    CHECK_EQ(operator_def.input(0), operator_def.output(0))
        << "Broadcast is an in-place operator.";
  }
  ~BroadcastOp() {}

  bool RunOnDevice() {
    auto* output = Output(0);
    CHECK_GT(output->size(), 0);
    MPI_Bcast(static_cast<void*>(output->mutable_data()), output->size(),
              MPIDataTypeWrapper<dtype>::type(), root_, MPI_COMM_WORLD);
    return true;
  }

 protected:
  int root_;
  // Input: X. Output: X.
  // Note that Broadcast works in-place by definition.
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(BroadcastOp);
};


// AllreduceOp does Allreduce using MPI. Currently, only SUM is supported.
template <typename dtype, class DeviceContext>
class AllreduceOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AllreduceOp);

  bool RunOnDevice() {
    auto& input = Input(0);
    auto* output = Output(0);
    output->ReshapeLike(input);
    MPI_Allreduce(const_cast<dtype*>(input.data()),
                  output->mutable_data(), input.size(),
                  MPIDataTypeWrapper<dtype>::type(), MPI_SUM, MPI_COMM_WORLD);
    return true;
  }

 protected:
  // Input: X; Output: X_reduced.
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(AllreduceOp);
};

}  // namespace caffe2

#endif  // CAFFE2_MPI_MPI_OPS_H_

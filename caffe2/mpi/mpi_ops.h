#ifndef CAFFE2_MPI_MPI_OPS_H_
#define CAFFE2_MPI_MPI_OPS_H_

#include <mpi.h>

#include "caffe2/core/operator.h"
#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

// BroadcastOp does Broadcast using MPI.
template <class Context>
class BroadcastOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BroadcastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)) {
    CAFFE_CHECK_EQ(operator_def.input(0), operator_def.output(0))
        << "Broadcast is an in-place operator.";
  }
  ~BroadcastOp() {}

  bool RunOnDevice() {
    auto* output = Output(0);
    // Make sure that output is already allocated.
    CAFFE_CHECK_GT(output->size(), 0);
    MPI_CHECK(MPI_Bcast(
        output->raw_mutable_data(),
        output->nbytes(), MPIDataTypeWrapper<char>::type(),
        root_, MPI_COMM_WORLD));
    return true;
  }

 protected:
  int root_;
  // Input: X. Output: X.
  // Note that Broadcast works in-place by definition.
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  IN_PLACE_ALLOWED({0, 0});
  DISABLE_COPY_AND_ASSIGN(BroadcastOp);
};


// AllreduceOp does Allreduce using MPI. Currently, only SUM is supported.
template <typename T, class Context>
class AllreduceOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AllreduceOp);

  bool RunOnDevice() {
    auto& input = Input(0);
    auto* output = Output(0);
    output->ReshapeLike(input);
    if (output->template mutable_data<T>() == input.template data<T>()) {
      // We are doing in-place call. Special case handling.
      MPI_CHECK(MPI_Allreduce(
          MPI_IN_PLACE, output->template mutable_data<T>(), input.size(),
          MPIDataTypeWrapper<T>::type(), MPI_SUM, MPI_COMM_WORLD));
    } else {
      // normal allreduce.
      MPI_CHECK(MPI_Allreduce(
          const_cast<T*>(input.template data<T>()),
          output->template mutable_data<T>(),
          input.size(), MPIDataTypeWrapper<T>::type(), MPI_SUM,
          MPI_COMM_WORLD));
    }
    return true;
  }

 protected:
  // Input: X; Output: X_reduced.
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  IN_PLACE_ALLOWED({0, 0});
  DISABLE_COPY_AND_ASSIGN(AllreduceOp);
};

}  // namespace caffe2

#endif  // CAFFE2_MPI_MPI_OPS_H_

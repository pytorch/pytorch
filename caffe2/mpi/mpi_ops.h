#ifndef CAFFE2_MPI_MPI_OPS_H_
#define CAFFE2_MPI_MPI_OPS_H_

#include <mpi.h>

#include "caffe2/core/operator.h"
#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

// MPIBroadcastOp does MPIBroadcast using MPI.
template <class Context>
class MPIBroadcastOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MPIBroadcastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)) {}
  ~MPIBroadcastOp() {}

  bool RunOnDevice() override {
    auto* output = Output(0);
    // Make sure that output is already allocated.
    CAFFE_CHECK_GT(output->size(), 0);
    MPI_CHECK(MPI_Bcast(
        output->raw_mutable_data(),
        output->nbytes(), MPIDataTypeWrapper<char>::type(),
        root_, MPIComm()));
    return true;
  }

 protected:
  int root_;
  DISABLE_COPY_AND_ASSIGN(MPIBroadcastOp);
};

// MPIReduceOp does Reduce using MPI. Currently, only SUM is supported.
template <typename T, class Context>
class MPIReduceOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MPIReduceOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)) {}
  ~MPIReduceOp() {}

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    output->ReshapeLike(input);
    if (output->template mutable_data<T>() == input.template data<T>()) {
      // We are doing in-place call. Special case handling.
      MPI_CHECK(MPI_Reduce(
          MPI_IN_PLACE, output->template mutable_data<T>(), input.size(),
          MPIDataTypeWrapper<T>::type(), MPI_SUM, root_, MPIComm()));
    } else {
      // normal allreduce.
      MPI_CHECK(MPI_Reduce(
          const_cast<T*>(input.template data<T>()),
          output->template mutable_data<T>(),
          input.size(), MPIDataTypeWrapper<T>::type(),
          MPI_SUM, root_, MPIComm()));
    }
    return true;
  }

 protected:
  int root_;
  DISABLE_COPY_AND_ASSIGN(MPIReduceOp);
};

// MPIAllgatherOp does MPIAllgather using MPI.
template <typename T, class Context>
class MPIAllgatherOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(MPIAllgatherOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    vector<TIndex> output_dims = input.dims();
    output_dims[0] *= MPISize();
    output->Reshape(output_dims);
    MPI_CHECK(MPI_Allgather(
        const_cast<T*>(input.template data<T>()), input.size(),
        MPIDataTypeWrapper<T>::type(),
        output->template mutable_data<T>(), input.size(),
        MPIDataTypeWrapper<T>::type(), MPIComm()));
    return true;
  }

 protected:
  DISABLE_COPY_AND_ASSIGN(MPIAllgatherOp);
};




// MPIAllreduceOp does MPIAllreduce using MPI. Currently, only SUM is supported.
template <typename T, class Context>
class MPIAllreduceOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(MPIAllreduceOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    output->ReshapeLike(input);
    if (output->template mutable_data<T>() == input.template data<T>()) {
      // We are doing in-place call. Special case handling.
      MPI_CHECK(MPI_Allreduce(
          MPI_IN_PLACE, output->template mutable_data<T>(), input.size(),
          MPIDataTypeWrapper<T>::type(), MPI_SUM, MPIComm()));
    } else {
      // normal allreduce.
      MPI_CHECK(MPI_Allreduce(
          const_cast<T*>(input.template data<T>()),
          output->template mutable_data<T>(),
          input.size(), MPIDataTypeWrapper<T>::type(), MPI_SUM,
          MPIComm()));
    }
    return true;
  }

 protected:
  DISABLE_COPY_AND_ASSIGN(MPIAllreduceOp);
};

}  // namespace caffe2

#endif  // CAFFE2_MPI_MPI_OPS_H_

#ifndef CAFFE2_MPI_MPI_OPS_H_
#define CAFFE2_MPI_MPI_OPS_H_

#include <mpi.h>

#include "caffe2/core/operator.h"
#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

// TODO(jiayq): if needed, write up the use of color and key with MPI split.
// Currently, the operator simply creates a communicator that has the
// same topology as the Caffe2 global communicator.
template <class Context>
class MPICreateCommonWorldOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MPICreateCommonWorldOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    OperatorBase::Outputs()[0]->Reset(new MPICommonWorldWrapper());
    return true;
  }
};

template <class Context>
class MPIBroadcastOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MPIBroadcastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)) {}
  ~MPIBroadcastOp() {}

  bool RunOnDevice() override {
    MPI_Comm comm = OperatorBase::Input<MPICommonWorldWrapper>(0).comm();
    CAFFE_ENFORCE(OperatorBase::OutputIsType<Tensor<Context>>(0),
                  "Output is of wrong type.");
    auto* output = Output(0);
    // Make sure that output is already allocated.
    CAFFE_ENFORCE(output->size() > 0,
                  "Broadcast op uses in-place operation so the output "
                  "should be already allocated.");
    MPI_CHECK(MPI_Bcast(
        output->raw_mutable_data(),
        output->nbytes(),
        MPIDataTypeWrapper<char>::type(),
        root_,
        comm));
    return true;
  }

 protected:
  int root_;
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
    MPI_Comm comm = OperatorBase::Input<MPICommonWorldWrapper>(0).comm();
    auto& input = Input(1);
    auto* output = Output(0);
    output->ResizeLike(input);
    MPI_CHECK(MPI_Reduce(
        const_cast<T*>(input.template data<T>()),
        output->template mutable_data<T>(),
        input.size(),
        MPIDataTypeWrapper<T>::type(),
        MPI_SUM,
        root_,
        comm));
    return true;
  }

 protected:
  int root_;
};

// MPIAllgatherOp does MPIAllgather using MPI.
template <typename T, class Context>
class MPIAllgatherOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(MPIAllgatherOp);

  bool RunOnDevice() override {
    MPI_Comm comm = OperatorBase::Input<MPICommonWorldWrapper>(0).comm();
    auto& input = Input(1);
    auto* output = Output(0);
    vector<TIndex> output_dims = input.dims();
    output_dims[0] *= OperatorBase::Input<MPICommonWorldWrapper>(0).size();
    output->Resize(output_dims);
    MPI_CHECK(MPI_Allgather(
        const_cast<T*>(input.template data<T>()),
        input.size(),
        MPIDataTypeWrapper<T>::type(),
        output->template mutable_data<T>(),
        input.size(),
        MPIDataTypeWrapper<T>::type(),
        comm));
    return true;
  }
};

// MPIAllreduceOp does MPIAllreduce using MPI. Currently, only SUM is supported.
template <typename T, class Context>
class MPIAllreduceOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(MPIAllreduceOp);

  bool RunOnDevice() override {
    MPI_Comm comm = OperatorBase::Input<MPICommonWorldWrapper>(0).comm();
    auto& input = Input(1);
    auto* output = Output(0);
    output->ResizeLike(input);
    void* source;
    if (output->template mutable_data<T>() == input.template data<T>()) {
      // We are doing in-place call. Special case handling.
      source = MPI_IN_PLACE;
    } else {
      // Normal allreduce takes the source from the input.
      source = const_cast<T*>(input.template data<T>());
    }
    MPI_CHECK(MPI_Allreduce(
        source,
        output->template mutable_data<T>(),
        input.size(),
        MPIDataTypeWrapper<T>::type(),
        MPI_SUM,
        comm));
    return true;
  }

 protected:
};

}  // namespace caffe2

#endif  // CAFFE2_MPI_MPI_OPS_H_

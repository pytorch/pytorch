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
    CAFFE_ENFORCE(
        OperatorBase::OutputIsTensorType(0, Context::GetDeviceType()),
        "Output is of wrong type.");
    auto* output = Output(0);
    // Make sure that output is already allocated.
    CAFFE_ENFORCE(
        output->numel() > 0,
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
    auto* output = Output(0, input.sizes(), at::dtype<T>());
    MPI_CHECK(MPI_Reduce(
        const_cast<T*>(input.template data<T>()),
        output->template mutable_data<T>(),
        input.numel(),
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
    vector<int64_t> output_dims = input.sizes().vec();
    output_dims[0] *= OperatorBase::Input<MPICommonWorldWrapper>(0).size();
    output->Resize(output_dims);
    MPI_CHECK(MPI_Allgather(
        const_cast<T*>(input.template data<T>()),
        input.numel(),
        MPIDataTypeWrapper<T>::type(),
        output->template mutable_data<T>(),
        input.numel(),
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
    auto* output = Output(0, input.sizes(), at::dtype<T>());
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
        input.numel(),
        MPIDataTypeWrapper<T>::type(),
        MPI_SUM,
        comm));
    return true;
  }
};

template <class Context>
class MPISendTensorOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MPISendTensorOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        OP_SINGLE_ARG(int, "dst", dst_, MPI_ANY_SOURCE),
        OP_SINGLE_ARG(int, "tag", tag_, MPI_ANY_TAG),
        OP_SINGLE_ARG(bool, "raw_buffer", raw_buffer_, false) {
    CAFFE_ENFORCE(raw_buffer_, "non-raw-buffer transfer not supported yet.");
    CAFFE_ENFORCE(
        dst_ != MPI_ANY_SOURCE || def.input_size() == 4,
        "You should explicitly specify the to rank either via "
        "argument or via input blobs.");
    CAFFE_ENFORCE(
        tag_ != MPI_ANY_TAG || def.input_size() == 4,
        "You should explicitly specify the tag either via "
        "argument or via input blobs.");
  }

  bool RunOnDevice() override {
    MPI_Comm comm = OperatorBase::Input<MPICommonWorldWrapper>(COMM).comm();
    auto& input = Input(INPUT);
    if (InputSize() == 4) {
      dst_ = OperatorBase::Input<Tensor>(DST, CPU).template data<int>()[0];
      tag_ = OperatorBase::Input<Tensor>(TAG, CPU).template data<int>()[0];
    }
    if (raw_buffer_) {
      // We need to do a const cast to cope with the fact that, before OpenMPI
      // 1.7, MPI_Send expects a non-const pointer although it uses it in a
      // const way.
      MPI_CHECK(MPI_Send(
          const_cast<void*>(input.raw_data()),
          input.nbytes(),
          MPI_CHAR,
          dst_,
          tag_,
          comm));
    } else {
      CAFFE_NOT_IMPLEMENTED;
    }
    return true;
  }

 protected:
  int dst_;
  int tag_;
  bool raw_buffer_;

  INPUT_TAGS(COMM, INPUT, DST, TAG);
};

template <class Context>
class MPIReceiveTensorOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MPIReceiveTensorOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        OP_SINGLE_ARG(int, "src", src_, MPI_ANY_SOURCE),
        OP_SINGLE_ARG(int, "tag", tag_, MPI_ANY_TAG),
        OP_SINGLE_ARG(bool, "raw_buffer", raw_buffer_, false) {
    CAFFE_ENFORCE(raw_buffer_, "non-raw-buffer transfer not supported yet.");
  }

  bool RunOnDevice() override {
    MPI_Comm comm = OperatorBase::Input<MPICommonWorldWrapper>(COMM).comm();
    if (InputSize() == 4) {
      src_ = OperatorBase::Input<Tensor>(SRC_IN, CPU).template data<int>()[0];
      tag_ = OperatorBase::Input<Tensor>(TAG_IN, CPU).template data<int>()[0];
    }
    MPI_Status status;
    if (raw_buffer_) {
      auto* output = Output(OUTPUT);
      MPI_CHECK(MPI_Recv(
          output->raw_mutable_data(),
          output->nbytes(),
          MPI_CHAR,
          src_,
          tag_,
          comm,
          &status));
    } else {
      CAFFE_NOT_IMPLEMENTED;
    }
    auto* src_out = OperatorBase::Output<Tensor>(SRC_OUT, CPU);
    src_out->Resize();
    src_out->template mutable_data<int>()[0] = status.MPI_SOURCE;
    auto* tag_out = OperatorBase::Output<Tensor>(TAG_OUT, CPU);
    tag_out->Resize();
    tag_out->template mutable_data<int>()[0] = status.MPI_TAG;
    return true;
  }

 protected:
  int src_;
  int tag_;
  bool raw_buffer_;
  INPUT_TAGS(COMM, INPUT, SRC_IN, TAG_IN);
  OUTPUT_TAGS(OUTPUT, SRC_OUT, TAG_OUT);
};

} // namespace caffe2

#endif // CAFFE2_MPI_MPI_OPS_H_

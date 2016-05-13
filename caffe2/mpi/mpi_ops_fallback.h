#ifndef CAFFE2_MPI_MPI_OPS_FALLBACK_H_
#define CAFFE2_MPI_MPI_OPS_FALLBACK_H_

#include <mpi.h>

#include "caffe2/core/operator.h"
#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

template <class Context>
class FallbackMPIBroadcastOp final : public Operator<Context> {
 public:
  static_assert(!std::is_same<Context, CPUContext>::value,
                "You should not FallbackMPIBroadcastOp for CPUContext. Use "
                "MPIBroadcastOp directly.");
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FallbackMPIBroadcastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)) {
    CAFFE_VLOG(1) << "Using FallbackMPIBroadcastOp.";
    CAFFE_CHECK_EQ(operator_def.input(0), operator_def.output(0))
        << "MPIBroadcast is an in-place operator.";
  }
  ~FallbackMPIBroadcastOp() {}

  bool RunOnDevice() override {
    auto* output = Output(0);
    int nbytes = output->nbytes();
    CAFFE_CHECK_GT(nbytes, 0);
    cpu_buffer_.ReshapeLike(*output);
    void* cpu_buffer_data = cpu_buffer_.raw_mutable_data(output->meta());
    context_.template Memcpy<Context, CPUContext>(
        nbytes, output->raw_data(), cpu_buffer_data);
    MPI_CHECK(MPI_Bcast(
        cpu_buffer_data, nbytes, MPIDataTypeWrapper<char>::type(),
        root_, MPIComm()));
    context_.template Memcpy<CPUContext, Context>(
        nbytes, cpu_buffer_.raw_data(), output->raw_mutable_data());
    return true;
  }

 protected:
  int root_;
  TensorCPU cpu_buffer_;
  // Input: X. Output: X.
  // Note that MPIBroadcast works in-place by definition.
  DISABLE_COPY_AND_ASSIGN(FallbackMPIBroadcastOp);
};

// FallbackReduceOp does Reduce using MPI. Currently, only SUM is
// supported.
// TODO(jiayq): maybe unify the Reduce and Allreduce code because they
// are largely the same.
template <typename T, class Context>
class FallbackReduceOp final : public Operator<Context> {
 public:
  static_assert(!std::is_same<Context, CPUContext>::value,
                "You should not run FallbackReduceOp for CPUContext. Use "
                "ReduceOp directly.");
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FallbackReduceOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)) {
    CAFFE_VLOG(1) << "Using FallbackReduceOp.";
  }

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    cpu_buffer_.ReshapeLike(input);
    output->ReshapeLike(input);
    context_.template Copy<T, Context, CPUContext>(
        input.size(), input.template data<T>(),
        cpu_buffer_.mutable_data<T>());
    MPI_CHECK(MPI_Reduce(
        MPI_IN_PLACE, cpu_buffer_.mutable_data<T>(), input.size(),
        MPIDataTypeWrapper<T>::type(), MPI_SUM, root_, MPIComm()));
    context_.template Copy<T, CPUContext, Context>(
        input.size(), cpu_buffer_.data<T>(),
        output->template mutable_data<T>());
    return true;
  }

 protected:
  int root_;
  TensorCPU cpu_buffer_;
  DISABLE_COPY_AND_ASSIGN(FallbackReduceOp);
};

// FallbackAllgatherOp does Allgather using MPI. Currently, only SUM is
// supported.
template <typename T, class Context>
class FallbackAllgatherOp final : public Operator<Context> {
 public:
  static_assert(!std::is_same<Context, CPUContext>::value,
                "You should not run FallbackAllgatherOp for CPUContext. Use "
                "AllgatherOp directly.");
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FallbackAllgatherOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    CAFFE_VLOG(1) << "Using FallbackAllgatherOp.";
  }

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    cpu_buffer_.ReshapeLike(input);

    vector<TIndex> output_dims = input.dims();
    output_dims[0] *= MPISize();
    output->ReshapeLike(output_dims);
    cpu_gather_buffer_.Reshape(output_dims);
    context_.template Copy<T, Context, CPUContext>(
        input.size(), input.template data<T>(),
        cpu_buffer_.mutable_data<T>());
    MPI_CHECK(MPI_Allgather(
        const_cast<T*>(cpu_buffer_.template data<T>()), input.size(),
        MPIDataTypeWrapper<T>::type(),
        cpu_gather_buffer_.template mutable_data<T>(), input.size(),
        MPIDataTypeWrapper<T>::type(), MPIComm()));
    context_.template Copy<T, CPUContext, Context>(
        output->size(), cpu_gather_buffer_.data<T>(),
        output->template mutable_data<T>());
    return true;
  }

 protected:
  TensorCPU cpu_buffer_;
  TensorCPU cpu_gather_buffer_;
  DISABLE_COPY_AND_ASSIGN(FallbackAllgatherOp);
};


// FallbackMPIAllreduceOp does MPIAllreduce using MPI. Currently, only SUM is
// supported.
template <typename T, class Context>
class FallbackMPIAllreduceOp final : public Operator<Context> {
 public:
  static_assert(!std::is_same<Context, CPUContext>::value,
                "You should not run FallbackMPIAllreduceOp for CPUContext. Use "
                "MPIAllreduceOp directly.");
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FallbackMPIAllreduceOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    CAFFE_VLOG(1) << "Using FallbackMPIAllreduceOp.";
  }

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    cpu_buffer_.ReshapeLike(input);
    output->ReshapeLike(input);
    context_.template Copy<T, Context, CPUContext>(
        input.size(), input.template data<T>(),
        cpu_buffer_.mutable_data<T>());
    MPI_CHECK(MPI_Allreduce(
        MPI_IN_PLACE, cpu_buffer_.mutable_data<T>(), input.size(),
        MPIDataTypeWrapper<T>::type(), MPI_SUM, MPIComm()));
    context_.template Copy<T, CPUContext, Context>(
        input.size(), cpu_buffer_.data<T>(),
        output->template mutable_data<T>());
    return true;
  }

 protected:
  // Input: X; Output: X_reduced.
  TensorCPU cpu_buffer_;
  DISABLE_COPY_AND_ASSIGN(FallbackMPIAllreduceOp);
};

}  // namespace caffe2

#endif  // CAFFE2_MPI_MPI_OPS_FALLBACK_H_

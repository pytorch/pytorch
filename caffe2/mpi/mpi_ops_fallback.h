#ifndef CAFFE2_MPI_MPI_OPS_FALLBACK_H_
#define CAFFE2_MPI_MPI_OPS_FALLBACK_H_

#include <mpi.h>

#include "caffe2/core/operator.h"
#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

template <class Context>
class FallbackBroadcastOp final : public Operator<Context> {
 public:
  static_assert(!std::is_same<Context, CPUContext>::value,
                "You should not FallbackBroadcastOp for CPUContext. Use "
                "BroadcastOp directly.");
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FallbackBroadcastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)) {
    CAFFE_VLOG(1) << "Using FallbackBroadcastOp.";
    CAFFE_CHECK_EQ(operator_def.input(0), operator_def.output(0))
        << "Broadcast is an in-place operator.";
  }
  ~FallbackBroadcastOp() {}

  bool RunOnDevice() override {
    auto* output = Output(0);
    int nbytes = output->nbytes();
    CAFFE_CHECK_GT(nbytes, 0);
    cpu_buffer_.ReshapeLike(*output);
    void* cpu_buffer_data = cpu_buffer_.raw_mutable_data(output->meta());
    device_context_.template Memcpy<Context, CPUContext>(
        nbytes, output->raw_data(), cpu_buffer_data);
    MPI_CHECK(MPI_Bcast(
        cpu_buffer_data, nbytes, MPIDataTypeWrapper<char>::type(),
        root_, MPI_COMM_WORLD));
    device_context_.template Memcpy<CPUContext, Context>(
        nbytes, cpu_buffer_.raw_data(), output->raw_mutable_data());
    return true;
  }

 protected:
  int root_;
  TensorCPU cpu_buffer_;
  // Input: X. Output: X.
  // Note that Broadcast works in-place by definition.
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  IN_PLACE_ALLOWED({0, 0});
  DISABLE_COPY_AND_ASSIGN(FallbackBroadcastOp);
};


// FallbackAllreduceOp does Allreduce using MPI. Currently, only SUM is
// supported.
template <typename T, class Context>
class FallbackAllreduceOp final : public Operator<Context> {
 public:
  static_assert(!std::is_same<Context, CPUContext>::value,
                "You should not run FallbackAllreduceOp for CPUContext. Use "
                "AllreduceOp directly.");
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FallbackAllreduceOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    CAFFE_VLOG(1) << "Using FallbackAllreduceOp.";
  }

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    cpu_buffer_.ReshapeLike(input);
    output->ReshapeLike(input);
    device_context_.template Copy<T, Context, CPUContext>(
        input.size(), input.template data<T>(),
        cpu_buffer_.mutable_data<T>());
    MPI_CHECK(MPI_Allreduce(
        MPI_IN_PLACE, cpu_buffer_.mutable_data<T>(), input.size(),
        MPIDataTypeWrapper<T>::type(), MPI_SUM, MPI_COMM_WORLD));
    device_context_.template Copy<T, CPUContext, Context>(
        input.size(), cpu_buffer_.data<T>(),
        output->template mutable_data<T>());
    return true;
  }

 protected:
  // Input: X; Output: X_reduced.
  TensorCPU cpu_buffer_;
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  IN_PLACE_ALLOWED({0, 0});
  DISABLE_COPY_AND_ASSIGN(FallbackAllreduceOp);
};

}  // namespace caffe2

#endif  // CAFFE2_MPI_MPI_OPS_FALLBACK_H_

#ifndef CAFFE2_MPI_MPI_OPS_FALLBACK_H_
#define CAFFE2_MPI_MPI_OPS_FALLBACK_H_

#include <mpi.h>

#include "caffe2/core/operator.h"
#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

template <typename dtype, class DeviceContext>
class FallbackBroadcastOp final : public Operator<dtype, DeviceContext> {
 public:
  static_assert(!std::is_same<DeviceContext, CPUContext>::value,
                "You should not FallbackBroadcastOp for CPUContext. Use "
                "BroadcastOp directly.");
  USE_OPERATOR_BASE_FUNCTIONS;
  FallbackBroadcastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)) {
    VLOG(1) << "Using FallbackBroadcastOp.";
    CHECK_EQ(operator_def.input(0), operator_def.output(0))
        << "Broadcast is an in-place operator.";
  }
  ~FallbackBroadcastOp() {}

  bool RunOnDevice() {
    auto* output = Output(0);
    int size = output->size();
    CHECK_GT(size, 0);
    cpu_buffer_.ReshapeLike(*output);
    device_context_.template Copy<dtype, DeviceContext, CPUContext>(
        size, output->data(), cpu_buffer_.mutable_data());
    MPI_CHECK(MPI_Bcast(
        static_cast<void*>(cpu_buffer_.mutable_data()), size,
        MPIDataTypeWrapper<dtype>::type(), root_, MPI_COMM_WORLD));
    device_context_.template Copy<dtype, CPUContext, DeviceContext>(
        size, cpu_buffer_.data(), output->mutable_data());
    return true;
  }

 protected:
  int root_;
  Tensor<dtype, CPUContext> cpu_buffer_;
  // Input: X. Output: X.
  // Note that Broadcast works in-place by definition.
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(FallbackBroadcastOp);
};


// FallbackAllreduceOp does Allreduce using MPI. Currently, only SUM is supported.
template <typename dtype, class DeviceContext>
class FallbackAllreduceOp final : public Operator<dtype, DeviceContext> {
 public:
  static_assert(!std::is_same<DeviceContext, CPUContext>::value,
                "You should not FallbackAllreduceOp for CPUContext. Use "
                "AllreduceOp directly.");
  USE_OPERATOR_BASE_FUNCTIONS;
  FallbackAllreduceOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws) {
    VLOG(1) << "Using FallbackAllreduceOp.";
  }

  bool RunOnDevice() {
    auto& input = Input(0);
    auto* output = Output(0);
    cpu_buffer_.ReshapeLike(input);
    output->ReshapeLike(input);
    device_context_.template Copy<dtype, DeviceContext, CPUContext>(
        input.size(), input.data(), cpu_buffer_.mutable_data());
    MPI_CHECK(MPI_Allreduce(
        MPI_IN_PLACE, cpu_buffer_.mutable_data(), input.size(),
        MPIDataTypeWrapper<dtype>::type(), MPI_SUM, MPI_COMM_WORLD));
    device_context_.template Copy<dtype, CPUContext, DeviceContext>(
        input.size(), cpu_buffer_.data(), output->mutable_data());
    return true;
  }

 protected:
  // Input: X; Output: X_reduced.
  Tensor<dtype, CPUContext> cpu_buffer_;
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(FallbackAllreduceOp);
};

}  // namespace caffe2

#endif  // CAFFE2_MPI_MPI_OPS_H_

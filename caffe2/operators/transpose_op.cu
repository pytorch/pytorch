#include "caffe2/operators/transpose_op.h"

#include <limits>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

// Cuda memory is precious so let's do a lower ndim limit.
#define COMPILE_TIME_CUDA_MAX_TRANSPOSE_DIMS 5

namespace {
// TODO(jiayq): one possible optimization is to copy the buffer into a shared
// memory location to speed up access.
template <typename Dtype>
__global__ void transpose_gpu(const int nthreads, const Dtype* from_data,
  Dtype* to_data, const int* buffer, const int num_axes) {
  int from_inds[COMPILE_TIME_CUDA_MAX_TRANSPOSE_DIMS];
  const int* from_counts = buffer;
  const int* to_counts = buffer + num_axes;
  const int* axes = buffer + num_axes * 2;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int from_index = index, to_index = 0;
    for (int i = num_axes - 1; i >= 0; --i) {
      from_inds[i] = from_index % from_counts[i];
      from_index = from_index / from_counts[i];
    }
    for (int i = 0; i < num_axes - 1; i++) {
      to_index = (to_index + from_inds[axes[i]]) * to_counts[i + 1];
    }
    to_index += from_inds[axes[num_axes - 1]];
    to_data[to_index] = from_data[index];
  }
}
}  // namespace

template <>
template <typename T>
bool TransposeOp<CUDAContext>::DoRunWithType() {
  const auto& input = Input(0);
  auto* output = Output(0);
  int count = input.size();
  int ndim = input.ndim();
  CAFFE_ENFORCE(
      count < std::numeric_limits<int>::max(),
      "Transpose op on GPU only supports int32");
  CAFFE_ENFORCE(
      ndim <= COMPILE_TIME_CUDA_MAX_TRANSPOSE_DIMS,
      "Input ndim exceeds compile time max.");

  // Buffer contains the following data:
  // (1) the dimenions of the inputs
  // (2) the dimension of the outputs
  // (3) the axis mapping from inputs to outputs
  buffer_cpu_.Resize(3 * ndim);
  int* buffer_data = buffer_cpu_.mutable_data<int>();
  for (int i = 0; i < ndim; ++i) {
    *(buffer_data++) = input.dim32(i);
  }
  for (int i = 0; i < ndim; ++i) {
    *(buffer_data++) = output->dim32(i);
  }
  for (int i = 0; i < ndim; ++i) {
    *(buffer_data++) = axes_[i];
  }
  // Copy the dimension information to GPU.
  buffer_.CopyFrom(buffer_cpu_, &context_);
  transpose_gpu<T><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
                     0, context_.cuda_stream()>>>(
      count, input.template data<T>(), output->template mutable_data<T>(),
      buffer_.data<int>(), ndim);
  return true;
}

REGISTER_CUDA_OPERATOR(Transpose, TransposeOp<CUDAContext>);
} // namespace caffe2

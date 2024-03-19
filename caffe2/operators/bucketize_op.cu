#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/bucketize_op.h"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

namespace caffe2 {

__global__ void BucketizeOpKernel(
    const int N,
    const int M,
    const float* bounds,
    const float* X,
    int32_t* out) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int32_t low = -1, high = M;
    while (high - low > 1) {
      const int32_t median = low + (high - low) / 2;
      if (bounds[median] < X[i]) {
        low = median;
      } else {
        high = median;
      }
    }
    out[i] = high;
  }
}

template <>
bool BucketizeOp<CUDAContext>::RunOnDevice() {
  auto& input = Input(X);
  CAFFE_ENFORCE_GE(input.dim(), 1);

  auto N = input.numel();
  auto* output = Output(INDICES, input.sizes(), at::dtype<int32_t>());
  const auto* input_data = input.template data<float>();
  auto* output_data = output->template mutable_data<int32_t>();

  BucketizeOpKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      boundaries_device_.numel(),
      boundaries_device_.data<float>(),
      input_data,
      output_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
};

REGISTER_CUDA_OPERATOR(Bucketize, BucketizeOp<CUDAContext>);
} // namespace caffe2

using BucketizeCUDA = caffe2::BucketizeOp<caffe2::CUDAContext>;

C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(
    Bucketize,
    BucketizeCUDA);

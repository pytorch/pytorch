#include "caffe2/operators/mod_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void ModOpSimpleKernel(const int N, const int64_t divisor_,
                            const T* data_ptr, T* output_ptr) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    output_ptr[i] = data_ptr[i] % divisor_;
  }
}


template <typename T>
__global__ void ModOpKernel(const int N, const int64_t divisor_,
                            const T* data_ptr, T* output_ptr) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    output_ptr[i] = data_ptr[i] % divisor_;
    if (output_ptr[i] && ((output_ptr[i] > 0) != (divisor_ > 0))) {
      output_ptr[i] += divisor_;
    }
  }
}

}  // namespace

template <>
template <typename T>
bool ModOp<CUDAContext>::DoRunWithType() {
  auto& data = Input(DATA);
  auto N = data.numel();
  const auto* data_ptr = data.template data<T>();

  auto* output = Output(0, data.sizes(), at::dtype<T>());
  auto* output_ptr = output->template mutable_data<T>();

  if (sign_follow_divisor_) {
    ModOpKernel<<<
        CAFFE_GET_BLOCKS(N),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N, divisor_, data_ptr, output_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    ModOpSimpleKernel<<<
        CAFFE_GET_BLOCKS(N),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N, divisor_, data_ptr, output_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return true;

}

REGISTER_CUDA_OPERATOR(Mod, ModOp<CUDAContext>);

} // namespace caffe2

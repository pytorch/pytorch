#include <ATen/native/optimizers.h>

#include <cmath>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace at {
namespace native {

namespace {

constexpr int kCUDANumThreads = 128;

template <typename T>
__global__ void SparseAdamStepCUDAKernel(
    int64_t N,
    const T* grad,
    const T* moment1,
    const T* moment2,
    T beta1,
    T beta2,
    T step_size,
    T eps,
    T* adam_step,
    T* moment1_step,
    T* moment2_step) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    const T m = (grad[index] - moment1[index]) * (T(1) - beta1);
    const T v = (grad[index] * grad[index] - moment2[index]) * (T(1) - beta2);
    adam_step[index] = -step_size * (m + moment1[index]) /
        (c10::cuda::compat::sqrt(v + moment2[index]) + eps);
    moment1_step[index] = m;
    moment2_step[index] = v;
  }
}

template <typename T>
void SparseAdamStepKernelImpl(
    double alpha,
    double beta1,
    double beta2,
    double eps,
    int64_t step,
    TensorIterator* it) {
  TORCH_CHECK(it->is_contiguous());
  if (!it->can_use_32bit_indexing()) {
    for (auto& sub_it : it->with_32bit_indexing()) {
      SparseAdamStepKernelImpl<T>(alpha, beta1, beta2, eps, step, &sub_it);
    }
    return;
  }
  const T* grad_data = static_cast<T*>(it->data_ptr(3));
  const T* moment1_data = static_cast<T*>(it->data_ptr(4));
  const T* moment2_data = static_cast<T*>(it->data_ptr(5));
  T* adam_step_data = static_cast<T*>(it->data_ptr(0));
  T* moment1_step_data = static_cast<T*>(it->data_ptr(1));
  T* moment2_step_data = static_cast<T*>(it->data_ptr(2));
  const int64_t N = it->numel();
  if (N > 0) {
    const double bias_correction1 =
        1.0 - std::pow(beta1, static_cast<double>(step));
    const double bias_correction2 =
        1.0 - std::pow(beta2, static_cast<double>(step));
    const double step_size =
        alpha * std::sqrt(bias_correction2) / bias_correction1;
    const int64_t B = (N + kCUDANumThreads - 1) / kCUDANumThreads;
    const cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
    SparseAdamStepCUDAKernel<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
        N,
        grad_data,
        moment1_data,
        moment2_data,
        static_cast<T>(beta1),
        static_cast<T>(beta2),
        static_cast<T>(step_size),
        static_cast<T>(eps),
        adam_step_data,
        moment1_step_data,
        moment2_step_data);
  }
}

void SparseAdamStepKernel(
    double alpha,
    double beta1,
    double beta2,
    double eps,
    int64_t step,
    TensorIterator* it) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      it->dtype(), "SparseAdamStepKernel", [&]() {
        SparseAdamStepKernelImpl<scalar_t>(alpha, beta1, beta2, eps, step, it);
      });
}

} // namespace

REGISTER_DISPATCH(sparse_adam_step_stub, &SparseAdamStepKernel);

} // namespace native
} // namespace at

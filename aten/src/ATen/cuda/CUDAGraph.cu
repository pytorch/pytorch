#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>

namespace at::cuda {

namespace {

#if !(defined(USE_ROCM)) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
template <typename scalar_t>
__global__ void set_conditional_handle_kernel(
    cudaGraphConditionalHandle handle,
    const scalar_t* value) {
  cudaGraphSetConditional(handle, static_cast<unsigned int>(*value));
}

template <typename scalar_t>
__global__ void set_conditional_handle_clamped_kernel(
    cudaGraphConditionalHandle handle,
    const scalar_t* value,
    unsigned int num_cases) {
  const int64_t index = static_cast<int64_t>(*value);
  const int64_t max_case = static_cast<int64_t>(num_cases) - 1;
  const unsigned int clamped_index = index < 0
      ? 0
      : static_cast<unsigned int>(index > max_case ? max_case : index);
  cudaGraphSetConditional(handle, clamped_index);
}
#endif
}

void CUDAGraph::set_conditional_handle(
    cudaGraphConditionalHandle handle,
    const Tensor& scalar_cuda_value_tensor,
    std::optional<unsigned int> num_cases) {
#if !(defined(USE_ROCM)) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  if (num_cases.has_value()) {
    TORCH_CHECK(
        num_cases.value() > 0,
        "CUDA graph switch conditional nodes require at least one case");
    TORCH_CHECK(
        at::isIntegralType(
            scalar_cuda_value_tensor.scalar_type(), /*includeBool=*/false),
        "CUDA graph switch conditional values must have integer dtype");
    AT_DISPATCH_INTEGRAL_TYPES(
        scalar_cuda_value_tensor.scalar_type(),
        "set_conditional_handle_clamped",
        [&] {
          set_conditional_handle_clamped_kernel<<<
              1,
              1,
              0,
              getCurrentCUDAStream()>>>(
              handle,
              scalar_cuda_value_tensor.const_data_ptr<scalar_t>(),
              num_cases.value());
        });
  } else if (scalar_cuda_value_tensor.scalar_type() == at::ScalarType::Bool) {
    set_conditional_handle_kernel<<<1, 1, 0, getCurrentCUDAStream()>>>(
        handle, scalar_cuda_value_tensor.const_data_ptr<bool>());
  } else {
    TORCH_CHECK(
        scalar_cuda_value_tensor.scalar_type() == at::ScalarType::Long,
        "CUDA graph conditional values must have bool or int64 dtype");
    set_conditional_handle_kernel<<<1, 1, 0, getCurrentCUDAStream()>>>(
        handle, scalar_cuda_value_tensor.const_data_ptr<int64_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  AT_ERROR("not allowed");
  return;
#endif
}

} // namespace at::cuda

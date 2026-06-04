#ifdef LAE_USE_CUDA

#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

using torch::stable::Tensor;

__global__ void test_kernel_on_stream(int* output, int n, int fill_value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = fill_value;
  }
}

Tensor test_kernel_launch_on_stream(Tensor input, int fill_value) {
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Int,
      "input must be int32 dtype");

  const auto device_index = input.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);

  Tensor output = torch::stable::new_empty(input, input.sizes());

  cudaStream_t stream = static_cast<cudaStream_t>(
    torch::stable::accelerator::getCurrentStream(device_index).nativeHandle());

  const int64_t n = input.numel();
  const int block_size = 256;
  const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

  int* output_ptr = reinterpret_cast<int*>(output.data_ptr());
  test_kernel_on_stream<<<grid_size, block_size, 0, stream>>>(
      output_ptr, static_cast<int>(n), fill_value);

  return output;
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("test_kernel_launch_on_stream(Tensor input, int fill_value) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CUDA, m) {
  m.impl("test_kernel_launch_on_stream", TORCH_BOX(&test_kernel_launch_on_stream));
}

#endif // LAE_USE_CUDA

#ifdef LAE_USE_CUDA

#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

using torch::stable::Tensor;

__global__ void test_kernel_on_stream(int* output, int magic_value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = magic_value;
  }
}

// Get stream via nativeHandle(), launch kernel.
Tensor test_kernel_launch_on_stream(Tensor input, int magic_value) {
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Int,
      "input must be int32 dtype");

  const auto device_index = input.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);

  Tensor output = torch::stable::new_empty(input, {1});

  void* native_handle =
      torch::stable::accelerator::getCurrentStream(device_index).nativeHandle();
  cudaStream_t stream = static_cast<cudaStream_t>(native_handle);

  int* output_ptr = reinterpret_cast<int*>(output.data_ptr());
  test_kernel_on_stream<<<1, 32, 0, stream>>>(output_ptr, magic_value);

  cudaError_t err = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(err == cudaSuccess,
      "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return output;
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("test_kernel_launch_on_stream(Tensor input, int magic_value) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CUDA, m) {
  m.impl("test_kernel_launch_on_stream", TORCH_BOX(&test_kernel_launch_on_stream));
}

#endif // LAE_USE_CUDA

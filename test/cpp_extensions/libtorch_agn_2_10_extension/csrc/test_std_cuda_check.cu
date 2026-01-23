#include <cuda_runtime.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>

__global__ void dummy_kernel(int /*unused*/) {
  // Intentionally empty
}

__global__ void invalid_kernel(int /*unused*/) {
  // This kernel itself is fine, but we'll launch it with invalid config
}

int test_std_cuda_check_success() {
  // cudaGetDevice should succeed if CUDA is available
  int device;
  STD_CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

void test_std_cuda_check_error() {
  // cudaSetDevice with an invalid device ID should fail
  // Using 99999 as an invalid device ID to trigger an error
  STD_CUDA_CHECK(cudaSetDevice(99999));
}

void test_std_cuda_kernel_launch_check_success() {
  // Launch a simple kernel with valid configuration
  dummy_kernel<<<1, 1>>>(0);

  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void test_std_cuda_kernel_launch_check_error() {
  // Launch a kernel with invalid configuration
  // Using more blocks than allowed (2^31) will trigger a launch error
  invalid_kernel<<<2147483648, 1>>>(0);

  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("test_std_cuda_check_success() -> int");
  m.def("test_std_cuda_check_error() -> ()");
  m.def("test_std_cuda_kernel_launch_check_success() -> ()");
  m.def("test_std_cuda_kernel_launch_check_error() -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(
    libtorch_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl(
      "test_std_cuda_check_success", TORCH_BOX(&test_std_cuda_check_success));
  m.impl("test_std_cuda_check_error", TORCH_BOX(&test_std_cuda_check_error));
  m.impl(
      "test_std_cuda_kernel_launch_check_success",
      TORCH_BOX(&test_std_cuda_kernel_launch_check_success));
  m.impl(
      "test_std_cuda_kernel_launch_check_error",
      TORCH_BOX(&test_std_cuda_kernel_launch_check_error));
}

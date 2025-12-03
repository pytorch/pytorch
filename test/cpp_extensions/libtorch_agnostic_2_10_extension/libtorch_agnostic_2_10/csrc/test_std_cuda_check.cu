#include <cuda_runtime.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>

// Simple CUDA kernel that does nothing
// Takes a dummy parameter to ensure hipify correctly handles the kernel launch
__global__ void dummy_kernel(int /*unused*/) {
  // Intentionally empty
}

// Invalid CUDA kernel with too many threads to trigger launch error
// Takes a dummy parameter to ensure hipify correctly handles the kernel launch
__global__ void invalid_kernel(int /*unused*/) {
  // This kernel itself is fine, but we'll launch it with invalid config
}

// Test function that should successfully execute a CUDA operation
int test_std_cuda_check_success() {
  // cudaGetDevice should succeed if CUDA is available
  int device;
  STD_CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

// Test function that should throw due to invalid CUDA operation
void test_std_cuda_check_error() {
  // cudaSetDevice with an invalid device ID should fail
  // Using 99999 as an invalid device ID to trigger an error
  STD_CUDA_CHECK(cudaSetDevice(99999));
}

// Test function that successfully launches a kernel and checks for errors
void test_std_cuda_kernel_launch_check_success() {
  // Launch a simple kernel with valid configuration
  dummy_kernel<<<1, 1>>>(0);

  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

// Test function that should throw due to invalid kernel launch
void test_std_cuda_kernel_launch_check_error() {
  // Launch a kernel with invalid configuration
  // Using more blocks than allowed (2^31) will trigger a launch error
  invalid_kernel<<<(1U << 31), 1>>>(0);

  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic_2_10, m) {
  m.def("test_std_cuda_check_success() -> int");
  m.def("test_std_cuda_check_error() -> ()");
  m.def("test_std_cuda_kernel_launch_check_success() -> ()");
  m.def("test_std_cuda_kernel_launch_check_error() -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(
    libtorch_agnostic_2_10,
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

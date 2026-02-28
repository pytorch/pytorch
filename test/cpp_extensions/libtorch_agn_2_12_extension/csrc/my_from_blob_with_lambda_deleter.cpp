#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#ifdef LAE_USE_CUDA
#include <cuda_runtime.h>
#endif

using torch::stable::Tensor;

// Global counter to track lambda deleter calls for testing
static int64_t g_lambda_deleter_call_count = 0;

// Wrapper for from_blob with a capturing-lambda deleter.
// The lambda captures a pointer to the global counter and increments it,
// which exercises the torch_from_blob_v2 code path (deleter + context).
Tensor my_from_blob_with_lambda_deleter(
    int64_t data_ptr,
    torch::headeronly::HeaderOnlyArrayRef<int64_t> sizes,
    torch::headeronly::HeaderOnlyArrayRef<int64_t> strides,
    torch::stable::Device device,
    torch::headeronly::ScalarType dtype) {
  void* data = reinterpret_cast<void*>(data_ptr);
  int64_t* counter = &g_lambda_deleter_call_count;
  auto deleter = [counter](void* /*data*/) { (*counter)++; };
  return torch::stable::from_blob(data, sizes, strides, device, dtype, deleter);
}

int64_t get_lambda_deleter_call_count() {
  return g_lambda_deleter_call_count;
}

void reset_lambda_deleter_call_count() {
  g_lambda_deleter_call_count = 0;
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_from_blob_with_lambda_deleter(int data_ptr, int[] sizes, int[] strides, Device device, ScalarType dtype) -> Tensor");
  m.def("get_lambda_deleter_call_count() -> int");
  m.def("reset_lambda_deleter_call_count() -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(
    STABLE_LIB_NAME,
    CompositeExplicitAutograd,
    m) {
  m.impl(
      "my_from_blob_with_lambda_deleter",
      TORCH_BOX(&my_from_blob_with_lambda_deleter));
  m.impl(
      "get_lambda_deleter_call_count",
      TORCH_BOX(&get_lambda_deleter_call_count));
  m.impl(
      "reset_lambda_deleter_call_count",
      TORCH_BOX(&reset_lambda_deleter_call_count));
}

#ifdef LAE_USE_CUDA

// Same as my_from_blob_with_cuda_deleter (from 2.11) but uses a non-capturing
// lambda deleter, exercising the from_blob_v2 code path.
Tensor my_from_blob_with_cuda_lambda_deleter(
    int64_t numel,
    torch::stable::Device device) {
  size_t size_bytes = numel * sizeof(float);

  void* data = nullptr;
  cudaError_t err = cudaMalloc(&data, size_bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed");
  }

  // Zero the memory
  cudaMemset(data, 0, size_bytes);

  std::array<int64_t, 1> sizes = {numel};
  std::array<int64_t, 1> strides = {1};

  // This lambda doesn't capture anything, but capture is tested above in
  // my_from_blob_with_lambda_deleter
  auto deleter = [](void* data) { cudaFree(data); };
  return torch::stable::from_blob(
      data,
      torch::headeronly::HeaderOnlyArrayRef<int64_t>(sizes.data(), sizes.size()),
      torch::headeronly::HeaderOnlyArrayRef<int64_t>(strides.data(), strides.size()),
      device,
      torch::headeronly::ScalarType::Float,
      deleter);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_from_blob_with_cuda_lambda_deleter(int numel, Device device) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_from_blob_with_cuda_lambda_deleter", TORCH_BOX(&my_from_blob_with_cuda_lambda_deleter));
}

#endif  // LAE_USE_CUDA

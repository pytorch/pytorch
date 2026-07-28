#include <torch/headeronly/cuda/AtomicAdd.h>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>

using torch::stable::Tensor;

namespace {

template <typename scalar_t>
__global__ void fast_atomic_add_kernel(
    scalar_t* out,
    const int64_t* indices,
    const scalar_t* values,
    int64_t n,
    int64_t out_numel,
    bool fast_atomics) {
  for (int64_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    torch::headeronly::fastAtomicAdd(
        out, indices[i], out_numel, values[i], fast_atomics);
  }
}

} // namespace

Tensor my_fast_atomic_add(
    Tensor out,
    Tensor indices,
    Tensor values,
    bool fast_atomics) {
  STD_TORCH_CHECK(out.dim() == 1, "out must be 1D");
  STD_TORCH_CHECK(indices.dim() == 1, "indices must be 1D");
  STD_TORCH_CHECK(values.dim() == 1, "values must be 1D");
  STD_TORCH_CHECK(
      indices.size(0) == values.size(0),
      "indices and values must have the same length");
  STD_TORCH_CHECK(
      indices.scalar_type() == torch::headeronly::ScalarType::Long,
      "indices must be int64");
  STD_TORCH_CHECK(
      out.scalar_type() == values.scalar_type(),
      "out and values must have the same dtype");
  STD_TORCH_CHECK(
      out.device() == values.device(), "out and values must be on the same device");

  const auto device_index = out.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);

  const int64_t n = indices.size(0);
  if (n == 0) {
    return out;
  }

  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
#ifdef USE_ROCM
  auto stream = static_cast<hipStream_t>(raw_stream);
#else
  auto stream = static_cast<cudaStream_t>(raw_stream);
#endif

  const int64_t out_numel = out.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);

  THO_DISPATCH_V2(
      out.scalar_type(),
      "my_fast_atomic_add",
      AT_WRAP(([&]() {
        fast_atomic_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            reinterpret_cast<const int64_t*>(indices.data_ptr()),
            reinterpret_cast<const scalar_t*>(values.data_ptr()),
            n,
            out_numel,
            fast_atomics);
      })),
      AT_EXPAND(AT_FLOATING_TYPES),
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
  return out;
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_fast_atomic_add(Tensor out, Tensor indices, Tensor values, bool fast_atomics) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CUDA, m) {
  m.impl("my_fast_atomic_add", TORCH_BOX(&my_fast_atomic_add));
}

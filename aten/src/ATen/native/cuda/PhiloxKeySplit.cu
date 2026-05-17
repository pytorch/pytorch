#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/StatelessPhilox4x32.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_key_fold_in_native.h>
#include <ATen/ops/_philox_key_split_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at::native {

namespace {

using at::cuda::philox_4x32;

// Derive a new (seed, offset) key from 4 random uint32 values.
// Use 2 uint32s for the 64-bit seed and 2 for the 64-bit offset.
__device__ __forceinline__ void philox_derive_key(
    uint4 r,
    uint64_t* out_seed,
    uint64_t* out_offset) {
  *out_seed = static_cast<uint64_t>(r.x) | (static_cast<uint64_t>(r.y) << 32);
  *out_offset = static_cast<uint64_t>(r.z) | (static_cast<uint64_t>(r.w) << 32);
}

// Grid-stride loop over (split_idx, key_idx) pairs.
__global__ void philox_key_split_kernel(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    int64_t num_keys,
    int64_t num_splits) {
  int64_t total = num_keys * num_splits;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; tid < total; tid += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t split_idx = tid / num_keys;
    int64_t key_idx = tid % num_keys;

    uint64_t seed = input[key_idx * 2];
    uint64_t offset = input[key_idx * 2 + 1];

    // Sample randomness to get the next (seed, offset pair).
    uint4 r = philox_4x32(seed, offset + static_cast<uint64_t>(split_idx));
    int64_t out = (split_idx * num_keys + key_idx) * 2;
    philox_derive_key(r, &output[out], &output[out + 1]);
  }
}

__global__ void philox_key_fold_in_kernel(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    int64_t num_keys,
    int64_t data) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; idx < num_keys; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint64_t seed = input[idx * 2];
    uint64_t offset = input[idx * 2 + 1];

    // Sample randomness to get the next (seed, offset pair).
    uint4 r = philox_4x32(seed, offset + static_cast<uint64_t>(data));
    philox_derive_key(r, &output[idx * 2], &output[idx * 2 + 1]);
  }
}

} // anonymous namespace

Tensor _philox_key_split_cuda(const Tensor& key, int64_t num_splits) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_split: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_key_split: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(num_splits > 0,
      "_philox_key_split: num_splits must be positive, got ",
      num_splits);

  // Output shape: (num_splits, *key.shape)
  auto output_sizes = key.sizes().vec();
  output_sizes.insert(output_sizes.begin(), num_splits);
  Tensor output = at::empty(output_sizes, key.options());
  int64_t num_keys = key.numel() / 2;
  if (num_keys == 0) {
    return output;
  }

  int64_t total_threads = num_keys * num_splits;
  constexpr int block_size = 256;
  int num_blocks = std::min(
      static_cast<int>((total_threads + block_size - 1) / block_size),
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 4);

  auto key_contig = key.contiguous();
  philox_key_split_kernel<<<num_blocks, block_size, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      key_contig.data_ptr<uint64_t>(),
      output.data_ptr<uint64_t>(),
      num_keys, num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

Tensor _philox_key_fold_in_cuda(const Tensor& key, int64_t data) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_fold_in: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_key_fold_in: key must have dtype uint64, got ",
      key.scalar_type());

  Tensor output = at::empty_like(key);
  int64_t num_keys = key.numel() / 2;
  if (num_keys == 0) {
    return output;
  }

  constexpr int block_size = 256;
  int num_blocks = std::min(
      static_cast<int>((num_keys + block_size - 1) / block_size),
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 4);

  auto key_contig = key.contiguous();
  philox_key_fold_in_kernel<<<num_blocks, block_size, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      key_contig.data_ptr<uint64_t>(),
      output.data_ptr<uint64_t>(),
      num_keys, data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

} // namespace at::native

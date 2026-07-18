#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/StatelessPhilox4x32.cuh>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/DistributionTemplates.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/OpMathType.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/irange.h>
#include <limits>
#include <type_traits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_normal_flat_slice_native.h>
#include <ATen/ops/_philox_normal_indexed_native.h>
#include <ATen/ops/_philox_normal_native.h>
#include <ATen/ops/_philox_uniform_flat_slice_native.h>
#include <ATen/ops/_philox_uniform_indexed_native.h>
#include <ATen/ops/_philox_uniform_native.h>
#endif

namespace at::native {

namespace {

using at::cuda::philox_4x32;

// Elements produced per Philox 4x32 call: 4 for float/half/bfloat16, 2 for double.
// Note that we use a full float for each generated half/bfloat16 for better numerics.
template <typename scalar_t>
constexpr int elems_per_call = std::is_same_v<scalar_t, double> ? 2 : 4;

// Box-Muller: convert 4 uniform uint32 values into 4 standard normal floats.
__device__ __forceinline__ float4 box_muller_float(uint4 r) {
  constexpr float M = 2.3283064365386963e-10f; // 1/2^32
  constexpr float TWO_PI = 6.2831853071795864f;
  // Map to (0, 1] to avoid log(0).
  float u1 = fmaf(r.x, M, M * 0.5f);
  float u2 = fmaf(r.y, M, M * 0.5f);
  float u3 = fmaf(r.z, M, M * 0.5f);
  float u4 = fmaf(r.w, M, M * 0.5f);

  float radius1 = sqrtf(-2.0f * __logf(u1));
  float radius2 = sqrtf(-2.0f * __logf(u3));
  float s1, c1, s2, c2;
  __sincosf(TWO_PI * u2, &s1, &c1);
  __sincosf(TWO_PI * u4, &s2, &c2);
  return {radius1 * c1, radius1 * s1, radius2 * c2, radius2 * s2};
}

// Box-Muller: convert 4 uint32 values (packed into 2 uint64) into 2 standard
// normal doubles.
__device__ __forceinline__ double2 box_muller_double(uint4 r) {
  constexpr double M = 2.3283064365386963e-10; // 1/2^32
  constexpr double TWO_PI = 6.2831853071795864;
  // Pack pairs of uint32 for ~64 bits of uniform randomness.
  double u1 = fma(static_cast<double>(r.x), M,
                  static_cast<double>(r.y) * M * M + M * M * 0.5);
  double u2 = fma(static_cast<double>(r.z), M,
                  static_cast<double>(r.w) * M * M + M * M * 0.5);

  double radius = ::sqrt(-2.0 * ::log(u1));
  double s, c;
  ::sincos(TWO_PI * u2, &s, &c);
  return {radius * c, radius * s};
}

void validate_philox_index_blocks(
    const char* op_name,
    const Tensor& self,
    int64_t logical_numel,
    IntArrayRef start_indices,
    IntArrayRef block_sizes,
    IntArrayRef block_strides,
    IntArrayRef block_counts) {
  TORCH_CHECK(
      logical_numel >= 0,
      op_name,
      ": logical_numel must be non-negative, got ",
      logical_numel);
  TORCH_CHECK(
      start_indices.size() == block_sizes.size() &&
          start_indices.size() == block_strides.size() &&
          start_indices.size() == block_counts.size(),
      op_name,
      ": index block arrays must have the same length");
  TORCH_CHECK(
      self.numel() <= logical_numel,
      op_name,
      ": output numel ",
      self.numel(),
      " exceeds logical_numel ",
      logical_numel);

  int64_t mapped_numel = 0;
  int64_t previous_end = 0;
  for (const auto index : c10::irange(start_indices.size())) {
    const int64_t start_index = start_indices[index];
    const int64_t block_size = block_sizes[index];
    const int64_t block_stride = block_strides[index];
    const int64_t block_count = block_counts[index];
    TORCH_CHECK(
        start_index >= 0,
        op_name,
        ": start_index must be non-negative, got ",
        start_index);
    TORCH_CHECK(
        block_size > 0,
        op_name,
        ": block_size must be positive, got ",
        block_size);
    TORCH_CHECK(
        block_count > 0,
        op_name,
        ": block_count must be positive, got ",
        block_count);
    TORCH_CHECK(
        block_stride >= block_size,
        op_name,
        ": block_stride ",
        block_stride,
        " must be at least block_size ",
        block_size);
    TORCH_CHECK(
        start_index >= previous_end,
        op_name,
        ": index blocks must be ordered and non-overlapping; start_index ",
        start_index,
        " is before the previous end ",
        previous_end);
    TORCH_CHECK(
        start_index <= logical_numel,
        op_name,
        ": start_index ",
        start_index,
        " exceeds logical_numel ",
        logical_numel);
    TORCH_CHECK(
        block_size <= logical_numel - start_index,
        op_name,
        ": first block ends beyond logical_numel ",
        logical_numel);

    const int64_t remaining_after_first =
        logical_numel - start_index - block_size;
    TORCH_CHECK(
        block_count == 1 ||
            block_count - 1 <= remaining_after_first / block_stride,
        op_name,
        ": index blocks end beyond logical_numel ",
        logical_numel);
    TORCH_CHECK(
        block_size <= self.numel() - mapped_numel &&
            block_count <=
                (self.numel() - mapped_numel) / block_size,
        op_name,
        ": index blocks describe more than output numel ",
        self.numel());

    const int64_t local_numel = block_size * block_count;
    const int64_t end_index =
        start_index + (block_count - 1) * block_stride + block_size;
    mapped_numel += local_numel;
    previous_end = end_index;
  }
  TORCH_CHECK(
      mapped_numel == self.numel(),
      op_name,
      ": index blocks describe ",
      mapped_numel,
      " elements, expected output numel ",
      self.numel());
}

void validate_philox_flat_slice_args(
    const char* op_name,
    const Tensor& self,
    int64_t total_numel,
    IntArrayRef start_indices,
    IntArrayRef block_sizes,
    IntArrayRef block_strides,
    IntArrayRef num_blocks) {
  TORCH_CHECK(total_numel >= 0, op_name, ": total_numel must be non-negative");
  TORCH_CHECK(
      start_indices.size() == block_sizes.size() &&
          start_indices.size() == block_strides.size() &&
          start_indices.size() == num_blocks.size(),
      op_name,
      ": index block arrays must have the same length");
  TORCH_CHECK(
      self.numel() <= total_numel,
      op_name,
      ": output numel ",
      self.numel(),
      " exceeds total_numel ",
      total_numel);
  TORCH_CHECK(
      total_numel <= std::numeric_limits<int32_t>::max(),
      op_name,
      ": total_numel > INT_MAX is not supported yet");

  int64_t mapped_numel = 0;
  for (const auto index : c10::irange(start_indices.size())) {
    const int64_t start_index = start_indices[index];
    const int64_t block_size = block_sizes[index];
    const int64_t block_stride = block_strides[index];
    const int64_t block_count = num_blocks[index];
    TORCH_CHECK(
        start_index >= 0, op_name, ": start_index must be non-negative");
    TORCH_CHECK(
        start_index <= total_numel,
        op_name,
        ": start_index ",
        start_index,
        " exceeds total_numel ",
        total_numel);
    TORCH_CHECK(
        block_size >= 0, op_name, ": block_size must be non-negative");
    TORCH_CHECK(
        block_count >= 0, op_name, ": num_blocks must be non-negative");
    if (block_size == 0 || block_count == 0) {
      continue;
    }
    TORCH_CHECK(
        block_count <= total_numel / block_size,
        op_name,
        ": block_size * num_blocks exceeds total_numel");
    TORCH_CHECK(
        block_stride >= block_size,
        op_name,
        ": block_stride ",
        block_stride,
        " must be at least block_size ",
        block_size);
    TORCH_CHECK(
        block_stride <= total_numel,
        op_name,
        ": block_stride ",
        block_stride,
        " exceeds total_numel ",
        total_numel);
    const int64_t local_numel = block_size * block_count;
    TORCH_CHECK(
        local_numel <= self.numel() - mapped_numel,
        op_name,
        ": index blocks describe more than output numel ",
        self.numel());
    const int64_t end_index =
        start_index + (block_count - 1) * block_stride + block_size;
    TORCH_CHECK(
        end_index <= total_numel,
        op_name,
        ": output blocks end at ",
        end_index,
        ", beyond total_numel ",
        total_numel);
    mapped_numel += local_numel;
  }
  TORCH_CHECK(
      mapped_numel == self.numel(),
      op_name,
      ": index blocks describe ",
      mapped_numel,
      " elements, expected output numel ",
      self.numel());
}

void validate_philox_indexed_args(
    const char* op_name,
    const Tensor& self,
    const Tensor& key,
    int64_t logical_numel,
    IntArrayRef start_indices,
    IntArrayRef block_sizes,
    IntArrayRef block_strides,
    IntArrayRef block_counts) {
  TORCH_CHECK(
      self.is_floating_point(),
      op_name,
      ": self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(
      key.scalar_type() == kUInt64,
      op_name,
      ": key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(
      self.device() == key.device(),
      op_name,
      ": self and key must be on the same device, got ",
      self.device(),
      " and ",
      key.device());
  TORCH_CHECK(
      key.dim() == 1 && key.size(0) == 2,
      op_name,
      ": key must have shape (2,), got shape ",
      key.sizes());
  validate_philox_index_blocks(
      op_name,
      self,
      logical_numel,
      start_indices,
      block_sizes,
      block_strides,
      block_counts);
}

void validate_normal_std(double stddev) {
  TORCH_CHECK(
      stddev >= 0.0,
      "normal expects std >= 0.0, but found std ",
      stddev);
}

template <typename scalar_t>
void validate_uniform_bounds(const Tensor& self, double low, double high) {
  const auto min =
      static_cast<double>(std::numeric_limits<scalar_t>::lowest());
  const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
  TORCH_CHECK(low >= min && low <= max, "from is out of bounds for ", self.dtype());
  TORCH_CHECK(
      high >= min && high <= max, "to is out of bounds for ", self.dtype());
  TORCH_CHECK(
      low <= high,
      "uniform_ expects to return a [from, to) range, but found from=",
      low,
      " > to=",
      high);
  TORCH_CHECK(
      high - low <= max,
      "uniform_ expects to-from <= std::numeric_limits<",
      toString(self.scalar_type()),
      ">::max(), but found to=",
      high,
      " and from=",
      low,
      " which result in to-from to exceed the limit");
}

__device__ __forceinline__ int64_t decode_philox_logical_index(
    int64_t local_index,
    int64_t local_numel,
    int64_t start_index,
    int64_t block_size,
    int64_t block_stride) {
  return block_size == local_numel
      ? start_index + local_index
      : start_index + (local_index / block_size) * block_stride +
          local_index % block_size;
}

template <typename scalar_t, typename sample_t, typename param_t>
void philox_distribution_kernel(
    const char* op_name,
    Tensor& self,
    const Tensor& key,
    const sample_t& sample_func,
    const param_t& param_func);

template <typename scalar_t, typename sample_t, typename param_t>
__global__ void philox_indexed_distribution_kernel(
    scalar_t* __restrict__ output,
    const uint64_t* __restrict__ key,
    int64_t local_numel,
    int64_t start_index,
    int64_t block_size,
    int64_t block_stride,
    sample_t sample_func,
    param_t param_func) {
  constexpr int epc = elems_per_call<scalar_t>;
  const uint64_t seed = key[0];
  const uint64_t offset = key[1];
  const int64_t local_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t local_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  while (local_index < local_numel) {
    const int64_t logical_index = decode_philox_logical_index(
        local_index, local_numel, start_index, block_size, block_stride);
    const uint64_t counter = static_cast<uint64_t>(logical_index / epc);
    const int64_t lane = logical_index % epc;
    auto sample = sample_func(seed, offset + counter);
    output[local_index] = param_func((&sample.x)[lane]);
    if (local_numel - local_index <= local_stride) {
      break;
    }
    local_index += local_stride;
  }
}

template <typename scalar_t, typename sample_t, typename param_t>
void philox_indexed_distribution(
    Tensor& self,
    const Tensor& key,
    IntArrayRef start_indices,
    IntArrayRef block_sizes,
    IntArrayRef block_strides,
    IntArrayRef block_counts,
    const sample_t& sample_func,
    const param_t& param_func) {
  if (self.numel() == 0) {
    return;
  }

  // Snapshot the key before writing output because callers can construct a
  // floating-point output that aliases the uint64 key through a dtype view.
  auto key_snapshot =
      self.is_alias_of(key) ? key.clone().contiguous() : key.contiguous();
  if (start_indices.size() == 1 && start_indices[0] == 0 &&
      block_sizes[0] == self.numel() && block_counts[0] == 1) {
    philox_distribution_kernel<scalar_t>(
        "philox_indexed_distribution",
        self,
        key_snapshot,
        sample_func,
        param_func);
    return;
  }

  auto output = self.contiguous();
  constexpr int threads = 256;
  constexpr int max_blocks = 65535;
  int64_t local_start = 0;
  for (const auto index : c10::irange(start_indices.size())) {
    const int64_t block_size = block_sizes[index];
    const int64_t local_numel = block_size * block_counts[index];
    const int64_t required_blocks =
        (local_numel - 1) / threads + 1;
    const int blocks = static_cast<int>(
        required_blocks < max_blocks ? required_blocks : max_blocks);
    philox_indexed_distribution_kernel<scalar_t>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            output.mutable_data_ptr<scalar_t>() + local_start,
            key_snapshot.const_data_ptr<uint64_t>(),
            local_numel,
            start_indices[index],
            block_size,
            block_strides[index],
            sample_func,
            param_func);
    local_start += local_numel;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (output.data_ptr() != self.data_ptr()) {
    self.copy_(output);
  }
}

template <typename scalar_t, typename sample_t, typename param_t>
__global__ void distribution_flat_slice_kernel(
    scalar_t* __restrict__ output,
    PhiloxCudaState philox_args,
    int64_t local_numel,
    int64_t total_numel,
    int64_t start_index,
    int64_t block_size,
    int64_t block_stride,
    int64_t total_grid,
    sample_t sample_func,
    param_t param_func) {
  constexpr int unroll_factor = elems_per_call<scalar_t>;
  const int64_t local_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_idx >= local_numel) {
    return;
  }

  const int64_t logical_index = decode_philox_logical_index(
      local_idx, local_numel, start_index, block_size, block_stride);
  if (logical_index >= total_numel) {
    return;
  }

  const int64_t total_stride = blockDim.x * total_grid;
  const int64_t thread_index = logical_index % total_stride;
  const int64_t thread_iteration_and_lane = logical_index / total_stride;
  const int64_t lane = thread_iteration_and_lane % unroll_factor;
  const uint64_t thread_iteration =
      static_cast<uint64_t>(thread_iteration_and_lane / unroll_factor);

  auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  curand_init(
      seed,
      static_cast<uint64_t>(thread_index),
      offset + thread_iteration * max_generator_offsets_per_curand_call,
      &state);
  auto sample = sample_func(&state);
  output[local_idx] = param_func((&sample.x)[lane]);
}

template <typename scalar_t, typename sample_t, typename param_t>
void distribution_flat_slice(
    const char* op_name,
    Tensor& self,
    int64_t total_numel,
    IntArrayRef start_indices,
    IntArrayRef block_sizes,
    IntArrayRef block_strides,
    IntArrayRef num_blocks,
    std::optional<Generator> generator,
    const sample_t& sample_func,
    const param_t& param_func) {
  TORCH_CHECK(
      self.is_floating_point(),
      op_name,
      ": self must be a floating point tensor, got ",
      self.scalar_type());
  validate_philox_flat_slice_args(
      op_name,
      self,
      total_numel,
      start_indices,
      block_sizes,
      block_strides,
      num_blocks);
  if (total_numel == 0) {
    return;
  }

  constexpr int unroll_factor = elems_per_call<scalar_t>;
  auto [counter_offset, total_grid, block] =
      calc_execution_policy(total_numel, unroll_factor);
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(
      generator, cuda::detail::getDefaultCUDAGenerator());
  PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }
  if (self.numel() == 0) {
    return;
  }

  auto output = self.contiguous();
  int64_t local_start = 0;
  for (const auto index : c10::irange(start_indices.size())) {
    const int64_t start_index = start_indices[index];
    const int64_t block_size = block_sizes[index];
    const int64_t block_stride = block_strides[index];
    const int64_t local_numel = block_size * num_blocks[index];
    if (local_numel == 0) {
      continue;
    }
    const uint32_t local_grid =
        static_cast<uint32_t>((local_numel + block.x - 1) / block.x);
    distribution_flat_slice_kernel<scalar_t>
        <<<local_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.mutable_data_ptr<scalar_t>() + local_start,
        rng_engine_inputs,
        local_numel,
        total_numel,
        start_index,
        block_size,
        block_stride,
        total_grid.x,
        sample_func,
        param_func);
    local_start += local_numel;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (output.data_ptr() != self.data_ptr()) {
    self.copy_(output);
  }
}

// Single-key kernel: one thread per chunk of elements, where each chunk
// comes from a single Philox 4x32 call. Uses vectorized stores for full
// chunks and scalar writes for the tail.
template <typename scalar_t, typename sample_t, typename param_t>
__global__ void philox_single_key_kernel(
    scalar_t* __restrict__ output,
    const uint64_t* __restrict__ key,
    int64_t num_elems,
    sample_t sample_func,
    param_t param_func) {

  // Use vectorized load to get (seed, offset)
  auto key_vec = memory::ld_vec<16>(key);
  auto* key_vals = reinterpret_cast<const uint64_t*>(&key_vec);
  uint64_t seed = key_vals[0];
  uint64_t offset = key_vals[1];

  // Use vectorized stores for full chunks since they're aligned.
  constexpr int epc = elems_per_call<scalar_t>;
  int64_t num_full_chunks = num_elems / epc;
  int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (chunk < num_full_chunks) {
    auto sample = sample_func(seed, offset + static_cast<uint64_t>(chunk));
    constexpr int vec_bytes = epc * sizeof(scalar_t);
    memory::Vec<vec_bytes> v;
    auto* vals = reinterpret_cast<scalar_t*>(&v);
    #pragma unroll
    for (int j = 0; j < epc; j++) {
      vals[j] = param_func((&sample.x)[j]);
    }
    memory::st_vec<vec_bytes>(output + chunk * epc, v);
  }

  // Scalar tail for remaining elements.
  if (chunk == num_full_chunks) {
    int64_t tail_start = num_full_chunks * epc;
    auto sample = sample_func(seed, offset + static_cast<uint64_t>(num_full_chunks));
    for (int j = 0; j < num_elems - tail_start; j++) {
      output[tail_start + j] = param_func((&sample.x)[j]);
    }
  }
}

// Multi-key kernel: one thread per (key_idx, chunk) pair, where each chunk
// comes from a single Philox 4x32 call. Uses vectorized stores for full
// chunks and scalar writes for the tail.
template <typename scalar_t, typename sample_t, typename param_t>
__global__ void philox_multi_key_kernel(
    scalar_t* __restrict__ output,
    const uint64_t* __restrict__ keys,
    int64_t num_keys,
    int64_t elems_per_key,
    sample_t sample_func,
    param_t param_func,
    OffsetCalculator<1> key_offset_calc) {
  constexpr int epc = elems_per_call<scalar_t>;
  int64_t chunks_per_key = (elems_per_key + epc - 1) / epc;
  int64_t total_threads = num_keys * chunks_per_key;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_threads) return;

  // Determine correct (seed, offset) to use and sample.
  int64_t key_idx = tid / chunks_per_key;
  int64_t chunk = tid % chunks_per_key;
  auto elem_offset = key_offset_calc.get(key_idx)[0];
  uint64_t seed = keys[elem_offset];
  uint64_t offset = keys[elem_offset + 1];
  auto sample = sample_func(seed, offset + static_cast<uint64_t>(chunk));

  // Vectorized writes require aligned base addresses. This is guaranteed
  // when elems_per_key is a multiple of epc, since
  // base = key_idx * elems_per_key + chunk * epc.
  int64_t full_chunks_per_key = elems_per_key / epc;
  bool aligned = elems_per_key % epc == 0;
  int64_t base = key_idx * elems_per_key + chunk * epc;
  if (aligned && chunk < full_chunks_per_key) {
    constexpr int vec_bytes = epc * sizeof(scalar_t);
    memory::Vec<vec_bytes> v;
    auto* vals = reinterpret_cast<scalar_t*>(&v);
    #pragma unroll
    for (int j = 0; j < epc; j++) {
      vals[j] = param_func((&sample.x)[j]);
    }
    memory::st_vec<vec_bytes>(output + base, v);
  } else {
    for (int j = 0; j < epc && chunk * epc + j < elems_per_key; j++) {
      output[base + j] = param_func((&sample.x)[j]);
    }
  }
}

// Dispatches to single-key or multi-key kernels as needed.
template <typename scalar_t, typename sample_t, typename param_t>
void philox_distribution_kernel(
    const char* op_name,
    Tensor& self, const Tensor& key,
    const sample_t& sample_func, const param_t& param_func) {
  TORCH_CHECK(self.is_floating_point(),
      op_name, ": self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      op_name, ": key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(self.device() == key.device(),
      op_name, ": self and key must be on the same device, got ",
      self.device(), " and ", key.device());
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      op_name, ": key must have shape (2,) or (*batch, 2), got shape ",
      key.sizes());
  if (key.dim() > 1) {
    TORCH_CHECK(key.dim() == self.dim() + 1,
        op_name, ": batched key must have ndim == output ndim + 1, "
        "got key shape ", key.sizes(), " with output shape ", self.sizes());
    auto key_batch = key.sizes().slice(0, self.dim());
    TORCH_CHECK(is_expandable_to(key_batch, self.sizes()),
        op_name, ": key batch shape ", key_batch,
        " is not broadcastable with output shape ", self.sizes());
  }

  if (self.numel() == 0) {
    return;
  }

  // Ensure contiguous, aligned output for vectorized stores. Clone if needed
  // to ensure alignment; the result is copied back into self afterwards.
  constexpr int vec_bytes = elems_per_call<scalar_t> * sizeof(scalar_t);
  auto output = self.contiguous();
  if (reinterpret_cast<uintptr_t>(output.data_ptr()) % vec_bytes != 0) {
    output = output.clone();
  }

  constexpr int block_size = 256;

  if (key.dim() == 1) {
    // === Launch single key kernel ===
    constexpr int epc = elems_per_call<scalar_t>;
    int64_t num_chunks = (self.numel() + epc - 1) / epc;
    int num_blocks = static_cast<int>((num_chunks + block_size - 1) / block_size);

    auto key_contig = key.contiguous();
    philox_single_key_kernel<scalar_t>
        <<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.mutable_data_ptr<scalar_t>(),
        key_contig.data_ptr<uint64_t>(),
        self.numel(), sample_func, param_func);
  } else {
    // === Launch batched (multiple) key kernel ===
    // The kernel writes each key's output as a contiguous block of
    // elems_per_key elements. We determine elems_per_key by counting
    // trailing size-1 key dims; these are the output dimensions that a
    // single key generates over. For example, with key shape (4, 1, 1, 2)
    // and output shape (4, 10, 100): key_dims=1, elems_per_key=1000.
    int64_t elems_per_key = 1;
    int64_t key_dims = self.dim();
    for (int64_t i = self.dim() - 1; i >= 0; i--) {
      if (key.size(i) != 1) break;
      elems_per_key *= self.size(i);
      key_dims--;
    }
    int64_t num_keys = self.numel() / elems_per_key;

    // Handle key, self broadcasting via OffsetCalculator.
    c10::SmallVector<int64_t, MAX_DIMS> oc_sizes(key_dims);
    c10::SmallVector<int64_t, MAX_DIMS> oc_strides(key_dims);
    for (int64_t i = 0; i < key_dims; i++) {
      int64_t dim = key_dims - 1 - i;
      oc_sizes[i] = self.size(dim);
      oc_strides[i] = key.size(dim) > 1 ? key.stride(dim) : 0;
    }
    const int64_t* oc_strides_ptr = oc_strides.data();
    auto key_offset_calc = OffsetCalculator<1>(
        key_dims, oc_sizes.data(), &oc_strides_ptr);

    int64_t chunks_per_key =
        (elems_per_key + elems_per_call<scalar_t> - 1) / elems_per_call<scalar_t>;
    int64_t total_threads = num_keys * chunks_per_key;
    int num_blocks = static_cast<int>((total_threads + block_size - 1) / block_size);

    philox_multi_key_kernel<scalar_t>
        <<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.mutable_data_ptr<scalar_t>(),
        key.const_data_ptr<uint64_t>(),
        num_keys, elems_per_key,
        sample_func, param_func, key_offset_calc);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (output.data_ptr() != self.data_ptr()) {
    self.copy_(output);
  }
}

} // anonymous namespace

Tensor& _philox_uniform_cuda_(
    Tensor& self, const Tensor& key, double low, double high) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "_philox_uniform_", [&] {
    auto sample_func = []() {
      if constexpr (std::is_same_v<scalar_t, double>) {
        return [] __device__ (uint64_t seed, uint64_t offset) {
          uint4 r = philox_4x32(seed, offset);
          ulonglong2 packed;
          packed.x = (static_cast<unsigned long long>(r.x) << 32) | r.y;
          packed.y = (static_cast<unsigned long long>(r.z) << 32) | r.w;
          return packed;
        };
      } else {
        return [] __device__ (uint64_t seed, uint64_t offset) {
          return philox_4x32(seed, offset);
        };
      }
    }();

    auto lo = static_cast<scalar_t>(low);
    auto hi = static_cast<scalar_t>(high);
    auto param_func = [lo, hi] __device__ (auto rand) {
      return static_cast<scalar_t>(
          at::transformation::uniform_real(rand, lo, hi));
    };

    philox_distribution_kernel<scalar_t>(
        "_philox_uniform_", self, key, sample_func, param_func);
  });
  return self;
}

Tensor& _philox_normal_cuda_(
    Tensor& self, const Tensor& key, double mean, double stddev) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "_philox_normal_", [&] {
    using compute_t = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
    auto sample_func = []() {
      if constexpr (std::is_same_v<scalar_t, double>) {
        return [] __device__ (uint64_t seed, uint64_t offset) {
          return box_muller_double(philox_4x32(seed, offset));
        };
      } else {
        return [] __device__ (uint64_t seed, uint64_t offset) {
          return box_muller_float(philox_4x32(seed, offset));
        };
      }
    }();

    auto mu = static_cast<compute_t>(mean);
    auto sigma = static_cast<compute_t>(stddev);
    auto param_func = [mu, sigma] __device__ (compute_t rand) {
      return static_cast<scalar_t>(rand * sigma + mu);
    };

    philox_distribution_kernel<scalar_t>(
        "_philox_normal_", self, key, sample_func, param_func);
  });
  return self;
}

Tensor& _philox_uniform_indexed_cuda_(
    Tensor& self,
    const Tensor& key,
    int64_t logical_numel,
    IntArrayRef start_indices,
    IntArrayRef block_sizes,
    IntArrayRef block_strides,
    IntArrayRef block_counts,
    double low,
    double high) {
  validate_philox_indexed_args(
      "_philox_uniform_indexed_",
      self,
      key,
      logical_numel,
      start_indices,
      block_sizes,
      block_strides,
      block_counts);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "_philox_uniform_indexed_", [&] {
        validate_uniform_bounds<scalar_t>(self, low, high);
        auto sample_func = []() {
          if constexpr (std::is_same_v<scalar_t, double>) {
            return [] __device__(uint64_t seed, uint64_t offset) {
              uint4 r = philox_4x32(seed, offset);
              ulonglong2 packed;
              packed.x =
                  (static_cast<unsigned long long>(r.x) << 32) | r.y;
              packed.y =
                  (static_cast<unsigned long long>(r.z) << 32) | r.w;
              return packed;
            };
          } else {
            return [] __device__(uint64_t seed, uint64_t offset) {
              return philox_4x32(seed, offset);
            };
          }
        }();

        auto lo = static_cast<scalar_t>(low);
        auto hi = static_cast<scalar_t>(high);
        auto param_func = [lo, hi] __device__(auto rand) {
          return static_cast<scalar_t>(
              at::transformation::uniform_real(rand, lo, hi));
        };

        philox_indexed_distribution<scalar_t>(
            self,
            key,
            start_indices,
            block_sizes,
            block_strides,
            block_counts,
            sample_func,
            param_func);
      });
  return self;
}

Tensor& _philox_normal_indexed_cuda_(
    Tensor& self,
    const Tensor& key,
    int64_t logical_numel,
    IntArrayRef start_indices,
    IntArrayRef block_sizes,
    IntArrayRef block_strides,
    IntArrayRef block_counts,
    double mean,
    double stddev) {
  validate_philox_indexed_args(
      "_philox_normal_indexed_",
      self,
      key,
      logical_numel,
      start_indices,
      block_sizes,
      block_strides,
      block_counts);
  validate_normal_std(stddev);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "_philox_normal_indexed_", [&] {
        using compute_t =
            std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
        auto sample_func = []() {
          if constexpr (std::is_same_v<scalar_t, double>) {
            return [] __device__(uint64_t seed, uint64_t offset) {
              return box_muller_double(philox_4x32(seed, offset));
            };
          } else {
            return [] __device__(uint64_t seed, uint64_t offset) {
              return box_muller_float(philox_4x32(seed, offset));
            };
          }
        }();

        auto mu = static_cast<compute_t>(mean);
        auto sigma = static_cast<compute_t>(stddev);
        auto param_func = [mu, sigma] __device__(compute_t rand) {
          return static_cast<scalar_t>(rand * sigma + mu);
        };

        philox_indexed_distribution<scalar_t>(
            self,
            key,
            start_indices,
            block_sizes,
            block_strides,
            block_counts,
            sample_func,
            param_func);
      });
  return self;
}

Tensor& _philox_uniform_flat_slice_symint_cuda_(
    Tensor& self,
    c10::SymInt total_numel,
    c10::SymIntArrayRef start_indices,
    c10::SymIntArrayRef block_sizes,
    c10::SymIntArrayRef block_strides,
    c10::SymIntArrayRef num_blocks,
    double low,
    double high,
    std::optional<Generator> generator) {
  const int64_t total_numel_int =
      total_numel.guard_int(__FILE__, __LINE__);
  const auto start_indices_int = C10_AS_INTARRAYREF_SLOW_ALLOC(start_indices);
  const auto block_sizes_int = C10_AS_INTARRAYREF_SLOW_ALLOC(block_sizes);
  const auto block_strides_int = C10_AS_INTARRAYREF_SLOW_ALLOC(block_strides);
  const auto num_blocks_int = C10_AS_INTARRAYREF_SLOW_ALLOC(num_blocks);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      self.scalar_type(),
      "_philox_uniform_flat_slice_",
      [&] {
        validate_uniform_bounds<scalar_t>(self, low, high);
        using opmath_t = at::opmath_type<scalar_t>;
        auto lo = static_cast<scalar_t>(low);
        auto hi = static_cast<scalar_t>(high);
        auto range = static_cast<opmath_t>(hi - lo);
        auto sample_func = []() {
          if constexpr (std::is_same_v<scalar_t, double>) {
            return [] __device__ (curandStatePhilox4_32_10_t* state) {
              return curand_uniform2_double(state);
            };
          } else {
            return [] __device__ (curandStatePhilox4_32_10_t* state) {
              return curand_uniform4(state);
            };
          }
        }();
        auto param_func = [range, lo, hi] __device__ (opmath_t rand) {
          auto value = static_cast<scalar_t>(rand * range + lo);
          return value == hi ? lo : value;
        };
        distribution_flat_slice<scalar_t>(
            "_philox_uniform_flat_slice_",
            self,
            total_numel_int,
            start_indices_int,
            block_sizes_int,
            block_strides_int,
            num_blocks_int,
            generator,
            sample_func,
            param_func);
      });
  return self;
}

Tensor& _philox_normal_flat_slice_symint_cuda_(
    Tensor& self,
    c10::SymInt total_numel,
    c10::SymIntArrayRef start_indices,
    c10::SymIntArrayRef block_sizes,
    c10::SymIntArrayRef block_strides,
    c10::SymIntArrayRef num_blocks,
    double mean,
    double stddev,
    std::optional<Generator> generator) {
  const int64_t total_numel_int =
      total_numel.guard_int(__FILE__, __LINE__);
  const auto start_indices_int = C10_AS_INTARRAYREF_SLOW_ALLOC(start_indices);
  const auto block_sizes_int = C10_AS_INTARRAYREF_SLOW_ALLOC(block_sizes);
  const auto block_strides_int = C10_AS_INTARRAYREF_SLOW_ALLOC(block_strides);
  const auto num_blocks_int = C10_AS_INTARRAYREF_SLOW_ALLOC(num_blocks);
  validate_normal_std(stddev);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      self.scalar_type(),
      "_philox_normal_flat_slice_",
      [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto mu = static_cast<accscalar_t>(mean);
        auto sigma = static_cast<accscalar_t>(stddev);
        auto sample_func = []() {
          if constexpr (std::is_same_v<scalar_t, double>) {
            return [] __device__ (curandStatePhilox4_32_10_t* state) {
              return curand_normal2_double(state);
            };
          } else {
            return [] __device__ (curandStatePhilox4_32_10_t* state) {
              return curand_normal4(state);
            };
          }
        }();
        auto param_func = [mu, sigma] __device__ (accscalar_t rand) {
          return static_cast<scalar_t>(
              at::transformation::normal<accscalar_t>(rand, mu, sigma));
        };
        distribution_flat_slice<scalar_t>(
            "_philox_normal_flat_slice_",
            self,
            total_numel_int,
            start_indices_int,
            block_sizes_int,
            block_strides_int,
            num_blocks_int,
            generator,
            sample_func,
            param_func);
      });
  return self;
}

} // namespace at::native

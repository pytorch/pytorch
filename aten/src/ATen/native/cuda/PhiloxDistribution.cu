#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/StatelessPhilox4x32.cuh>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/PhiloxDistribution.h>
#include <ATen/native/cuda/DistributionTemplates.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <c10/util/irange.h>
#include <c10/util/safe_numerics.h>
#include <array>
#include <type_traits>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_normal_native.h>
#include <ATen/ops/_philox_uniform_native.h>
#endif

namespace at::native {

namespace {

void run_normal_distribution_shards(
    Tensor& self,
    IntArrayRef global_shape,
    IntArrayRef global_offsets,
    IntArrayRef local_offsets,
    IntArrayRef local_sizes,
    int64_t chunk_count,
    double mean,
    double stddev,
    std::optional<Generator> generator);

void philox_distribution_shards_cuda(
    Tensor& self,
    IntArrayRef global_shape,
    IntArrayRef global_offsets,
    IntArrayRef local_offsets,
    IntArrayRef local_sizes,
    int64_t chunk_count,
    PhiloxDistributionKind distribution,
    ArrayRef<Scalar> params,
    std::optional<Generator> generator) {
  switch (distribution) {
    case PhiloxDistributionKind::Normal:
      run_normal_distribution_shards(
          self,
          global_shape,
          global_offsets,
          local_offsets,
          local_sizes,
          chunk_count,
          params[0].toDouble(),
          params[1].toDouble(),
          generator);
      break;
  }
}

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

// A non-empty logical tensor with at most INT_MAX elements has at most 30
// dimensions whose size is greater than one. Size-one dimensions are folded
// into each chunk's base offsets before constructing this calculator.
constexpr int kMaxDistributionShardDims = 30;

struct DistributionShardOffsetCalculator {
  C10_HOST_DEVICE std::array<int64_t, 2> get(uint32_t linear_idx) const {
    std::array<int64_t, 2> offsets{0, 0};
    for (int dim = 0; dim < dims; ++dim) {
      const auto divmod = sizes[dim].divmod(linear_idx);
      linear_idx = divmod.div;
      offsets[0] += static_cast<int64_t>(divmod.mod) * global_strides[dim];
      offsets[1] += static_cast<int64_t>(divmod.mod) * local_strides[dim];
    }
    return offsets;
  }

  int dims{0};
  at::cuda::detail::IntDivider<uint32_t> sizes[kMaxDistributionShardDims];
  int64_t global_strides[kMaxDistributionShardDims];
  int64_t local_strides[kMaxDistributionShardDims];
};

struct DistributionShardLaunch {
  int64_t numel;
  int64_t global_base;
  int64_t local_base;
  bool is_contiguous;
  DistributionShardOffsetCalculator offset_calculator;
};

void append_shard_dimension(
    DistributionShardOffsetCalculator& calculator,
    int64_t size,
    int64_t global_stride,
    int64_t local_stride) {
  if (size <= 1) {
    return;
  }

  if (calculator.dims > 0) {
    const int previous = calculator.dims - 1;
    const int64_t inner_size = calculator.sizes[previous].divisor;
    int64_t expected_global_stride;
    int64_t expected_local_stride;
    const bool global_stride_overflow = c10::mul_overflows(
        inner_size,
        calculator.global_strides[previous],
        &expected_global_stride);
    const bool local_stride_overflow = c10::mul_overflows(
        inner_size, calculator.local_strides[previous], &expected_local_stride);
    if (!global_stride_overflow && !local_stride_overflow &&
        global_stride == expected_global_stride &&
        local_stride == expected_local_stride) {
      const auto combined_size = static_cast<uint32_t>(inner_size * size);
      calculator.sizes[previous] =
          at::cuda::detail::IntDivider<uint32_t>(combined_size);
      return;
    }
  }

  TORCH_CHECK(
      calculator.dims < kMaxDistributionShardDims,
      "_philox_distribution_shards_: too many non-trivial shard dimensions");
  const int index = calculator.dims++;
  calculator.sizes[index] =
      at::cuda::detail::IntDivider<uint32_t>(static_cast<uint32_t>(size));
  calculator.global_strides[index] = global_stride;
  calculator.local_strides[index] = local_stride;
}

std::vector<DistributionShardLaunch> build_shard_launches(
    const Tensor& self,
    IntArrayRef global_shape,
    IntArrayRef global_offsets,
    IntArrayRef local_offsets,
    IntArrayRef local_sizes,
    int64_t chunk_count,
    const detail::ValidatedPhiloxShardMetadata& metadata) {
  constexpr const char* op_name = "_philox_distribution_shards_";
  const size_t ndim = global_shape.size();
  if (metadata.global_numel == 0) {
    return {};
  }

  std::vector<int64_t> global_strides(ndim);
  int64_t stride = 1;
  for (size_t dim = ndim; dim > 0; --dim) {
    global_strides[dim - 1] = stride;
    stride *= global_shape[dim - 1];
  }

  std::vector<DistributionShardLaunch> launches;
  launches.reserve(static_cast<size_t>(chunk_count));
  for (const auto chunk : c10::irange(static_cast<size_t>(chunk_count))) {
    if (metadata.chunk_numels[chunk] == 0) {
      continue;
    }
    DistributionShardLaunch launch{};
    launch.numel = metadata.chunk_numels[chunk];
    for (const auto dim : c10::irange(ndim)) {
      const size_t index = chunk * ndim + dim;
      int64_t global_term;
      int64_t local_term;
      TORCH_CHECK(
          !c10::mul_overflows(
              global_offsets[index], global_strides[dim], &global_term) &&
              !c10::add_overflows(
                  launch.global_base, global_term, &launch.global_base),
          op_name,
          ": global shard offset overflow");
      TORCH_CHECK(
          !c10::mul_overflows(
              local_offsets[index],
              self.stride(static_cast<int64_t>(dim)),
              &local_term) &&
              !c10::add_overflows(
                  launch.local_base, local_term, &launch.local_base),
          op_name,
          ": local shard offset overflow");
    }
    for (size_t dim = ndim; dim > 0; --dim) {
      const size_t logical_dim = dim - 1;
      append_shard_dimension(
          launch.offset_calculator,
          local_sizes[chunk * ndim + logical_dim],
          global_strides[logical_dim],
          self.stride(static_cast<int64_t>(logical_dim)));
    }
    launch.is_contiguous = launch.offset_calculator.dims <= 1 &&
        (launch.offset_calculator.dims == 0 ||
         (launch.offset_calculator.global_strides[0] == 1 &&
          launch.offset_calculator.local_strides[0] == 1));
    launches.push_back(launch);
  }
  return launches;
}

template <typename scalar_t, typename sample_t, typename param_t>
__global__ void distribution_contiguous_shard_kernel(
    scalar_t* __restrict__ output,
    PhiloxCudaState philox_args,
    int64_t local_numel,
    int64_t global_base,
    int64_t local_base,
    int64_t total_grid,
    sample_t sample_func,
    param_t param_func) {
  constexpr int unroll_factor = elems_per_call<scalar_t>;
  const int64_t thread_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_stride = blockDim.x * total_grid;
  const int64_t global_end = global_base + local_numel;
  if (thread_index >= global_end) {
    return;
  }

  const int64_t first_q = global_base <= thread_index
      ? 0
      : (global_base - thread_index + total_stride - 1) / total_stride;
  const int64_t last_q = (global_end - 1 - thread_index) / total_stride;
  if (first_q > last_q) {
    return;
  }
  const int64_t first_call = first_q / unroll_factor;
  const int64_t last_call = last_q / unroll_factor;

  auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  curand_init(
      seed,
      static_cast<uint64_t>(thread_index),
      offset + static_cast<uint64_t>(first_call) *
          max_generator_offsets_per_curand_call,
      &state);
  for (int64_t call = first_call; call <= last_call; ++call) {
    auto sample = sample_func(&state);
#pragma unroll
    for (int lane = 0; lane < unroll_factor; ++lane) {
      const int64_t logical_index =
          thread_index +
          total_stride * (call * unroll_factor + static_cast<int64_t>(lane));
      if (logical_index >= global_base && logical_index < global_end) {
        output[local_base + logical_index - global_base] =
            param_func((&sample.x)[lane]);
      }
    }
  }
}

template <typename scalar_t, typename sample_t, typename param_t>
__global__ void distribution_shard_kernel(
    scalar_t* __restrict__ output,
    PhiloxCudaState philox_args,
    int64_t local_numel,
    int64_t global_base,
    int64_t local_base,
    DistributionShardOffsetCalculator offset_calculator,
    int64_t total_grid,
    sample_t sample_func,
    param_t param_func) {
  constexpr int unroll_factor = elems_per_call<scalar_t>;
  const int64_t local_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_idx >= local_numel) {
    return;
  }

  const auto offsets = offset_calculator.get(static_cast<uint32_t>(local_idx));
  const int64_t logical_index = global_base + offsets[0];
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
  output[local_base + offsets[1]] = param_func((&sample.x)[lane]);
}

template <typename scalar_t, typename sample_t, typename param_t>
void distribution_shards(
    Tensor& self,
    const detail::ValidatedPhiloxShardMetadata& metadata,
    const std::vector<DistributionShardLaunch>& launches,
    std::optional<Generator> generator,
    const sample_t& sample_func,
    const param_t& param_func) {
  constexpr int unroll_factor = elems_per_call<scalar_t>;
  auto [counter_offset, total_grid, block] =
      calc_execution_policy(metadata.global_numel, unroll_factor);
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

  auto* output = self.mutable_data_ptr<scalar_t>();
  for (const auto& launch : launches) {
    const int64_t total_stride = block.x * total_grid.x;
    if (launch.is_contiguous && launch.numel >= total_stride) {
      distribution_contiguous_shard_kernel<scalar_t>
          <<<total_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              output,
              rng_engine_inputs,
              launch.numel,
              launch.global_base,
              launch.local_base,
              total_grid.x,
              sample_func,
              param_func);
      continue;
    }
    const uint32_t local_grid =
        static_cast<uint32_t>((launch.numel + block.x - 1) / block.x);
    distribution_shard_kernel<scalar_t>
        <<<local_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            output,
            rng_engine_inputs,
            launch.numel,
            launch.global_base,
            launch.local_base,
            launch.offset_calculator,
            total_grid.x,
            sample_func,
            param_func);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_normal_distribution_shards(
    Tensor& self,
    IntArrayRef global_shape,
    IntArrayRef global_offsets,
    IntArrayRef local_offsets,
    IntArrayRef local_sizes,
    int64_t chunk_count,
    double mean,
    double stddev,
    std::optional<Generator> generator) {
  auto metadata = detail::validate_philox_shard_metadata(
      self,
      global_shape,
      global_offsets,
      local_offsets,
      local_sizes,
      chunk_count);
  auto launches = build_shard_launches(
      self,
      global_shape,
      global_offsets,
      local_offsets,
      local_sizes,
      chunk_count,
      metadata);
  if (metadata.global_numel == 0) {
    return;
  }
  if (launches.size() == 1 &&
      launches[0].numel == metadata.global_numel &&
      launches[0].global_base == 0 && launches[0].local_base == 0 &&
      self.numel() == metadata.global_numel && self.is_contiguous()) {
    auto gen = get_generator_or_default<CUDAGeneratorImpl>(
        generator, cuda::detail::getDefaultCUDAGenerator());
    templates::cuda::normal_kernel(self, mean, stddev, gen);
    return;
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      self.scalar_type(),
      "_philox_distribution_shards_",
      [&] {
        distribution_shards<scalar_t>(
            self,
            metadata,
            launches,
            generator,
            templates::cuda::NormalDistributionSampler<scalar_t>{},
            templates::cuda::NormalDistributionTransform<scalar_t>(
                mean, stddev));
      });
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

REGISTER_DISPATCH(
    philox_distribution_shards_stub,
    &philox_distribution_shards_cuda)

} // namespace at::native

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/StatelessPhilox4x32.cuh>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/core/TransformationHelper.h>
#include <type_traits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_normal_native.h>
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
        key.data_ptr<uint64_t>(),
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

} // namespace at::native

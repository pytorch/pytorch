#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <curand_kernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_uniform_native.h>
#endif

namespace at::native {

namespace {

template <typename scalar_t, int N>
struct alignas(sizeof(scalar_t) * N) AlignedVec {
  scalar_t val[N];
};

template <typename scalar_t>
__device__ void uniform_generate(
    scalar_t* output, int64_t base, int64_t elem, int64_t elem_end,
    curandStatePhilox4_32_10_t* state, double low, double high) {
  float flow = static_cast<float>(low);
  float frange = static_cast<float>(high - low);
  float4 u = curand_uniform4(state);
  float vals[4] = {
    flow + frange * u.x, flow + frange * u.y,
    flow + frange * u.z, flow + frange * u.w
  };
  #pragma unroll
  for (int j = 0; j < 4 && elem + j < elem_end; j++) {
    output[base + elem + j] = static_cast<scalar_t>(vals[j]);
  }
}
template <>
__device__ void uniform_generate<double>(
    double* output, int64_t base, int64_t elem, int64_t elem_end,
    curandStatePhilox4_32_10_t* state, double low, double high) {
  double range = high - low;
  double2 u = curand_uniform2_double(state);
  output[base + elem] = low + range * u.x;
  if (elem + 1 < elem_end) {
    output[base + elem + 1] = low + range * u.y;
  }
}

template <typename scalar_t>
__device__ void uniform_generate_vec(
    scalar_t* output, int64_t pos,
    curandStatePhilox4_32_10_t* state, double low, double high) {
  float flow = static_cast<float>(low);
  float frange = static_cast<float>(high - low);
  float4 u = curand_uniform4(state);
  AlignedVec<scalar_t, 4> v;
  v.val[0] = static_cast<scalar_t>(flow + frange * u.x);
  v.val[1] = static_cast<scalar_t>(flow + frange * u.y);
  v.val[2] = static_cast<scalar_t>(flow + frange * u.z);
  v.val[3] = static_cast<scalar_t>(flow + frange * u.w);
  *reinterpret_cast<AlignedVec<scalar_t, 4>*>(&output[pos]) = v;
}

template <>
__device__ void uniform_generate_vec<double>(
    double* output, int64_t pos,
    curandStatePhilox4_32_10_t* state, double low, double high) {
  double range = high - low;
  double2 u = curand_uniform2_double(state);
  AlignedVec<double, 2> v;
  v.val[0] = low + range * u.x;
  v.val[1] = low + range * u.y;
  *reinterpret_cast<AlignedVec<double, 2>*>(&output[pos]) = v;
}

template <typename scalar_t, bool single_key, typename key_offset_calc_t>
__global__ void philox_uniform_kernel(
    scalar_t* __restrict__ output,
    const uint64_t* __restrict__ keys,
    int64_t num_keys,
    int64_t event_numel,
    int64_t elems_per_thread,
    double low,
    double high,
    key_offset_calc_t key_offset_calc) {
  constexpr size_t compute_size =
      sizeof(scalar_t) < sizeof(float) ? sizeof(float) : sizeof(scalar_t);
  constexpr int outputs_per_value = compute_size / sizeof(float);
  constexpr int elems_per_call = 4 / outputs_per_value;

  extern __shared__ char philox_smem_[];

  // Generate elements in [elem_start, elem_end), handling 64-bit offset wrap.
  auto generate_range = [&](scalar_t* out, int64_t base, int64_t elem_start,
                            int64_t elem_end, uint64_t seed,
                            uint64_t key_offset, bool use_vec) {
    unsigned long long philox_offset = key_offset +
        static_cast<unsigned long long>(elem_start) * outputs_per_value;

    auto outputs_in_range =
        static_cast<unsigned long long>(elem_end - elem_start) * outputs_per_value;
    bool range_wraps = philox_offset != 0 &&
        (philox_offset + outputs_in_range < philox_offset);
    int64_t wrap_elem = range_wraps
        ? elem_start + static_cast<int64_t>(
              (0ULL - philox_offset) / outputs_per_value)
        : elem_end;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, /*subsequence=*/0, /*offset=*/philox_offset, &state);

    int64_t gen_end = min(wrap_elem, elem_end);
    int64_t elem = elem_start;

    int64_t full_end = elem + ((gen_end - elem) / elems_per_call) * elems_per_call;
    if (use_vec) {
      for (; elem < full_end; elem += elems_per_call) {
        uniform_generate_vec<scalar_t>(out, base + elem, &state, low, high);
      }
    } else {
      for (; elem < full_end; elem += elems_per_call) {
        uniform_generate<scalar_t>(out, base, elem, gen_end, &state, low, high);
      }
    }
    if (elem < gen_end) {
      uniform_generate<scalar_t>(out, base, elem, gen_end, &state, low, high);
    }

    if (range_wraps) {
      curand_init(seed, /*subsequence=*/0, /*offset=*/0ULL, &state);
      elem = wrap_elem;
      full_end = elem + ((elem_end - elem) / elems_per_call) * elems_per_call;
      for (; elem < full_end; elem += elems_per_call) {
        uniform_generate<scalar_t>(out, base, elem, elem_end, &state, low, high);
      }
      if (elem < elem_end) {
        uniform_generate<scalar_t>(out, base, elem, elem_end, &state, low, high);
      }
    }
  };

  if constexpr (single_key) {
    uint64_t seed = keys[0];
    uint64_t key_offset = keys[1];

    // For non-double types, use warp-cooperative tiled generation with
    // shared memory transpose for coalesced writes. Unlike the normal
    // kernel, uniform has no Box-Muller alignment constraint, so this
    // path works for any key_offset. Fall through to generate_range
    // when the offset could wrap past 2^64.
    bool could_wrap = key_offset != 0 &&
        (key_offset + static_cast<unsigned long long>(event_numel) *
         outputs_per_value < key_offset);
    if constexpr (elems_per_call == 4) {
    if (!could_wrap) {
      constexpr int K = 8;
      constexpr int EPT = elems_per_call * K;  // 32 elements per thread
      constexpr int TILE = 32 * EPT;            // 1024 elements per tile
      constexpr int PADDED = EPT + 1;            // 33, bank-conflict-free

      float* smem = reinterpret_cast<float*>(philox_smem_);
      int warp_id = threadIdx.x / 32;
      int lane = threadIdx.x % 32;
      float* warp_smem = smem + warp_id * 32 * PADDED;

      int global_warp =
          (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
      int num_warps = static_cast<int>(gridDim.x) * (blockDim.x / 32);

      float flow = static_cast<float>(low);
      float frange = static_cast<float>(high - low);

      for (int64_t tile = static_cast<int64_t>(global_warp) * TILE;
           tile < event_numel;
           tile += static_cast<int64_t>(num_warps) * TILE) {

        int64_t my_start = tile + static_cast<int64_t>(lane) * EPT;
        unsigned long long philox_offset = key_offset +
            static_cast<unsigned long long>(my_start) * outputs_per_value;

        curandStatePhilox4_32_10_t state;
        curand_init(seed, /*subsequence=*/0, /*offset=*/philox_offset, &state);

        #pragma unroll
        for (int k = 0; k < K; k++) {
          float4 u = curand_uniform4(&state);
          warp_smem[lane * PADDED + k * 4 + 0] = flow + frange * u.x;
          warp_smem[lane * PADDED + k * 4 + 1] = flow + frange * u.y;
          warp_smem[lane * PADDED + k * 4 + 2] = flow + frange * u.z;
          warp_smem[lane * PADDED + k * 4 + 3] = flow + frange * u.w;
        }
        __syncwarp();

        #pragma unroll
        for (int m = 0; m < EPT; m++) {
          int64_t pos = tile + static_cast<int64_t>(m) * 32 + lane;
          if (pos < event_numel) {
            output[pos] = static_cast<scalar_t>(warp_smem[m * PADDED + lane]);
          }
        }
        __syncwarp();
      }
      return;
    }
    }

    // Fallback for double and offset-wrapping cases: contiguous per thread.
    int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int64_t per_thread =
        ((event_numel + total_threads - 1) / total_threads +
         elems_per_call - 1) / elems_per_call * elems_per_call;

    int64_t elem_start = tid * per_thread;
    if (elem_start >= event_numel) return;
    int64_t elem_end = min(elem_start + per_thread, event_numel);

    generate_range(output, 0, elem_start, elem_end, seed, key_offset, true);
    return;
  }

  // Multi-key: each thread handles a fixed-size chunk for one key.
  int64_t num_chunks = (event_numel + elems_per_thread - 1) / elems_per_thread;
  int64_t total_threads = num_keys * num_chunks;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (; tid < total_threads; tid += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t key_idx = tid / num_chunks;
    int64_t chunk_idx = tid % num_chunks;
    auto key_elem_offset = key_offset_calc.get(key_idx)[0];
    uint64_t seed = keys[key_elem_offset];
    uint64_t key_offset = keys[key_elem_offset + 1];

    int64_t elem_start = chunk_idx * elems_per_thread;
    int64_t elem_end = min(elem_start + elems_per_thread, event_numel);
    int64_t base = key_idx * event_numel;

    generate_range(output, base, elem_start, elem_end, seed, key_offset,
                   (base % elems_per_call) == 0);
  }
}

} // anonymous namespace

Tensor& _philox_uniform_cuda_(Tensor& self, const Tensor& key, double low, double high) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_uniform: key must have shape (2,) or (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_uniform: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(key.is_cuda(),
      "_philox_uniform: key must be a CUDA tensor");
  TORCH_CHECK(self.is_cuda(),
      "_philox_uniform: self must be a CUDA tensor");
  TORCH_CHECK(self.is_floating_point(),
      "_philox_uniform: self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(self.device() == key.device(),
      "_philox_uniform: self and key must be on the same device, got ",
      self.device(), " and ", key.device());

  if (self.numel() == 0) {
    return self;
  }

  at::cuda::CUDAGuard device_guard(key.device());

  int64_t ndim = self.dim();
  int64_t elems_per_key = 1;
  int64_t key_dims = 0;

  if (key.dim() > 1) {
    TORCH_CHECK(key.dim() == ndim + 1,
        "_philox_uniform: batched key must have ndim == output ndim + 1, "
        "got key shape ", key.sizes(), " with output shape ", self.sizes());

    for (int64_t i = 0; i < ndim; i++) {
      TORCH_CHECK(key.size(i) == 1 || key.size(i) == self.size(i),
          "_philox_uniform: key dim ", i, " (size ", key.size(i),
          ") is not broadcastable with output dim ", i,
          " (size ", self.size(i), ")");
    }

    key_dims = ndim;
    for (int64_t i = ndim - 1; i >= 0; i--) {
      if (key.size(i) != 1) break;
      elems_per_key *= self.size(i);
      key_dims--;
    }
  } else {
    elems_per_key = self.numel();
  }

  int64_t num_keys = self.numel() / elems_per_key;
  auto output = self.contiguous();

  Tensor key_contig;
  if (num_keys == 1) {
    key_contig = key.contiguous();
  }
  const uint64_t* key_ptr = num_keys == 1
      ? key_contig.data_ptr<uint64_t>()
      : key.data_ptr<uint64_t>();

  std::vector<int64_t> oc_sizes(key_dims);
  std::vector<int64_t> oc_strides(key_dims);
  for (int64_t i = 0; i < key_dims; i++) {
    int64_t dim = key_dims - 1 - i;
    oc_sizes[i] = self.size(dim);
    oc_strides[i] = key.size(dim) > 1 ? key.stride(dim) : 0;
  }
  const int64_t* oc_strides_ptr = oc_strides.data();
  auto key_offset_calc = OffsetCalculator<1>(
      key_dims, oc_sizes.data(), &oc_strides_ptr);

  constexpr int block_size = 256;
  int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  int max_blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm;

  constexpr int64_t elems_per_thread = 16;
  int num_blocks;
  if (num_keys == 1) {
    num_blocks = std::min(
        static_cast<int>((elems_per_key + block_size - 1) / block_size),
        max_blocks);
  } else {
    int64_t num_chunks = (elems_per_key + elems_per_thread - 1) / elems_per_thread;
    int64_t total_threads = num_keys * num_chunks;
    num_blocks = std::min(
        static_cast<int>((total_threads + block_size - 1) / block_size),
        max_blocks);
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "_philox_uniform_cuda", [&] {
    size_t smem_bytes = 0;
    if (num_keys == 1 && sizeof(scalar_t) <= sizeof(float)) {
      constexpr int PADDED = 4 * 8 + 1;  // elems_per_call * K + 1 = 33
      smem_bytes = (block_size / 32) * 32 * PADDED * sizeof(float);
    }

    if (num_keys == 1) {
      philox_uniform_kernel<scalar_t, true><<<num_blocks, block_size, smem_bytes,
          at::cuda::getCurrentCUDAStream()>>>(
          output.mutable_data_ptr<scalar_t>(),
          key_ptr, num_keys, elems_per_key, elems_per_thread, low, high,
          key_offset_calc);
    } else {
      philox_uniform_kernel<scalar_t, false><<<num_blocks, block_size, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          output.mutable_data_ptr<scalar_t>(),
          key_ptr, num_keys, elems_per_key, elems_per_thread, low, high,
          key_offset_calc);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  if (output.data_ptr() != self.data_ptr()) {
    self.copy_(output);
  }

  return self;
}

} // namespace at::native

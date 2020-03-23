#pragma once

#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <c10/util/Half.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <cstdint>
#include <limits>
#include <utility>
#include <mutex>

namespace {

// launch bounds used for kernels utilizing TensorIterator
const uint32_t block_size_bound = 256;
const uint32_t grid_size_bound = 4;
// number of randoms given by distributions like curand_uniform4, curand_uniform2_double
// used in calculating philox offset.
const uint32_t curand4_engine_calls = 4;

// utility function that calculates proper philox_offset
// for distributions utilizing TensorIterator. For distributions using
// TensorIterator, we are using a grid-stride loop with each
// thread yielding one element per thread. For the edge of the grid-stride
// loop, if the tensor size is large, the unroll loop will kick in and the float4
// from curand4 will start getting utilized (for common tensor sizes, we end up
// using rand.x from each thread). Hence, the philox_offset is
// (number of elements per thread * number of engine calls), which makes
// sure that philox offset increment is not less than the number of randoms used
// in each thread.
std::tuple<uint64_t, dim3, dim3> calc_execution_policy(int64_t total_elements) {
  const uint64_t numel = static_cast<uint64_t>(total_elements);
  const uint32_t block_size = block_size_bound;
  const uint32_t unroll = curand4_engine_calls;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  uint32_t blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  grid.x = std::min(
      static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm,
      grid.x);
  //number of times random will be generated per thread, to offset philox counter in thc random state
  uint64_t counter_offset = ((numel - 1) / (block_size * grid.x * unroll) + 1)
                                * curand4_engine_calls;
  return std::make_tuple(counter_offset, grid, dim_block);
}

// grid stride loop kernel for distributions
template<typename accscalar_t, int unroll_factor, typename dist_t, typename transform_t>
C10_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel(int numel,
                                                            std::pair<uint64_t, uint64_t> seeds,
                                                            const dist_t dist_func,
                                                            const transform_t transform_func) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(
      seeds.first,
      idx,
      seeds.second,
      &state);
  int rounded_size = ((numel - 1)/(blockDim.x * gridDim.x * unroll_factor)+1) *
      blockDim.x * gridDim.x * unroll_factor;
  for(int linear_index = idx; linear_index < rounded_size; linear_index += blockDim.x * gridDim.x * unroll_factor) {
    auto rand = dist_func(&state);
    #pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        transform_func(li, static_cast<accscalar_t>((&rand.x)[ii]));
      }
    }
    __syncthreads();
  }
}

/**
 * distribution_nullary_kernel is analogous to gpu_kernel in
 * ATen/native/cuda/Loops.cuh. Like gpu_kernel, it uses
 * TensorIterator to launch a kernel. However, the differences are
 *   - it launches a grid-stride loop based kernel. The kernel is not
 *     generic like elementwise_kernel in Loops.cuh and is specialized
 *     for the distribution kernels here.
 *   - For big size tensors, we can launch multiple kernels recursively
 *     (i.e. if (!iter.can_use_32bit_indexing())) and hence, the philox
 *     offset calculation is done in this function.
 *
 * FIXME: Can we specialize elementwise_kernel and launch_kernel in Loops.cuh
 * to have grid-stride loop kernel and then use that to launch our distribution
 * kernels? Note that we need a grid-stride loop kernel because, we found by testing
 * that it achieves peak effective bandwidth.
 */
template<typename scalar_t,
         typename accscalar_t,
         int unroll_factor,
         typename RNG,
         typename dist_t,
         typename transform_t>
void distribution_nullary_kernel(at::TensorIterator& iter,
                                 RNG gen,
                                 const dist_t& dist_func,
                                 const transform_t transform_func) {
  static_assert(unroll_factor >= 1, "unroll_factor must be >= 1.");
  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  auto execution_policy = calc_execution_policy(numel);
  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_nullary_kernel<scalar_t, accscalar_t, unroll_factor>(sub_iter,
        gen, dist_func, transform_func);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);

  auto stream = at::cuda::getCurrentCUDAStream();
  if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
      numel,
      rng_engine_inputs,
      dist_func,
      [=]__device__(int idx, accscalar_t rand) {
        scalar_t* out = (scalar_t*)&out_data[stride0 * idx];
        *out = transform_func(rand);
      }
    );
  } else {
    auto offset_calc = at::native::legacy::make_offset_calculator<1>(iter);
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
      numel,
      rng_engine_inputs,
      dist_func,
      [=]__device__(int idx, accscalar_t rand) {
        auto offsets = offset_calc.get(idx);
        scalar_t* out = (scalar_t*)&out_data[offsets[0]];
        *out = transform_func(rand);
      }
    );
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace at {
namespace native {
namespace templates {
namespace cuda {

template<typename RNG>
void random_from_to_kernel(TensorIterator& iter, uint64_t range, int64_t base, RNG gen) {
#ifdef _WIN32
  // TODO: https://github.com/pytorch/pytorch/issues/33793
  if (iter.dtype() == ScalarType::BFloat16) {
    TORCH_CHECK(false, "random_() is not supported for bfloat16 CUDA tensors on Windows. Please see https://github.com/pytorch/pytorch/issues/33793");
  }
#endif
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_from_to_kernel_cuda", [&] {
    if ((
      std::is_same<scalar_t, int64_t>::value ||
      std::is_same<scalar_t, double>::value ||
      std::is_same<scalar_t, float>::value ||
      std::is_same<scalar_t, at::BFloat16>::value) && range >= 1ULL << 32)
    {
      // define lambda to mod with range and add base
      auto random_func = [range, base] __device__ (uint64_t rand) {
        return static_cast<scalar_t>(static_cast<int64_t>(rand % range + base));
      };
      distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
          ulonglong2 ret;
          uint4 rand_val = curand4(state);
          ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
          ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
          return ret;
        },
        random_func);
    } else {
      auto random_func = [range, base] __device__ (uint32_t rand) {
        return static_cast<scalar_t>(static_cast<int64_t>(rand % range + base));
      };
      distribution_nullary_kernel<scalar_t, uint32_t, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand4(state);
        },
        random_func);
    }
   });
}

// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
template<typename RNG>
void random_full_64_bits_range_kernel(TensorIterator& iter, RNG gen) {
#ifdef _WIN32
  // TODO: https://github.com/pytorch/pytorch/issues/33793
  if (iter.dtype() == ScalarType::BFloat16) {
    TORCH_CHECK(false, "random_() is not supported for bfloat16 CUDA tensors on Windows. Please see https://github.com/pytorch/pytorch/issues/33793");
  }
#endif
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel_cuda", [&] {
    if (std::is_same<scalar_t, int64_t>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, at::BFloat16>::value) {
      auto random_func = [] __device__ (uint64_t rand) {
        return static_cast<scalar_t>(static_cast<int64_t>(rand));
      };
      distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
          ulonglong2 ret;
          uint4 rand_val = curand4(state);
          ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
          ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
          return ret;
        },
        random_func);
    } else {
      TORCH_CHECK(false, "random_full_64_bits_range_kernel_cuda handles only int64, double, float and bfloat16");
    }
  });
}

template<typename RNG>
struct RandomFromToKernel {
  void operator()(TensorIterator& iter, uint64_t range, int64_t base, RNG gen) {
    random_from_to_kernel(iter, range, base, gen);
  }
  void operator()(TensorIterator& iter, RNG gen) {
    random_full_64_bits_range_kernel(iter, gen);
  }
};

template<typename RNG>
void random_kernel(TensorIterator& iter, RNG gen) {
#ifdef _WIN32
  // TODO: https://github.com/pytorch/pytorch/issues/33793
  if (iter.dtype() == ScalarType::BFloat16) {
    TORCH_CHECK(false, "random_() is not supported for bfloat16 CUDA tensors on Windows. Please see https://github.com/pytorch/pytorch/issues/33793");
  }
#endif
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_kernel_fp_cuda", [&] {
      if (std::is_same<scalar_t, double>::value) {
        auto random_func = [] __device__ (uint64_t rand) {
          return static_cast<scalar_t>(rand % static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1));
        };
        distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter,
          gen,
          [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
            ulonglong2 ret;
            uint4 rand_val = curand4(state);
            ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
            ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
            return ret;
          },
          random_func);
      } else {
        auto random_func = [] __device__ (uint32_t rand) {
          return static_cast<scalar_t>(rand % static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1));
        };
        distribution_nullary_kernel<scalar_t, uint32_t, curand4_engine_calls>(iter,
          gen,
          [] __device__ (curandStatePhilox4_32_10_t* state) {
            return curand4(state);
          },
          random_func);
      }
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, iter.dtype(), "random_kernel_int_cuda", [&] {
      if (std::is_same<scalar_t, int64_t>::value) {
        auto random_func = [] __device__ (uint64_t rand) {
          return static_cast<scalar_t>(rand % (static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1));
        };
        distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter,
          gen,
          [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
            ulonglong2 ret;
            uint4 rand_val = curand4(state);
            ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
            ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
            return ret;
          },
          random_func);
      } else if (std::is_same<scalar_t, bool>::value) {
        auto random_func = [] __device__ (uint32_t rand) {
          return static_cast<scalar_t>(rand & 1);
        };
        distribution_nullary_kernel<scalar_t, uint32_t, curand4_engine_calls>(iter,
          gen,
          [] __device__ (curandStatePhilox4_32_10_t* state) {
            return curand4(state);
          },
          random_func);
      } else {
        auto random_func = [] __device__ (uint32_t rand) {
          return static_cast<scalar_t>(rand % (static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1));
        };
        distribution_nullary_kernel<scalar_t, uint32_t, curand4_engine_calls>(iter,
          gen,
          [] __device__ (curandStatePhilox4_32_10_t* state) {
            return curand4(state);
          },
          random_func);
      }
    });
  } else {
    TORCH_CHECK(false, "random_kernel_cuda handles only integral, floating-point and boolean types");
  }
}

template<typename RNG>
struct RandomKernel {
  void operator()(TensorIterator& iter, RNG gen) {
    random_kernel(iter, gen);
  }
};

}}}}

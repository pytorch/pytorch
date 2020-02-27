#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/CUDAGenerator.h>
#include <ATen/native/UnaryOps.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <utility>
#include <functional>

#include <ATen/native/Distributions.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/LegacyTHFunctionsCUDA.h>

#include <THC/THCGeneral.h>
#include <THC/THCApply.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>

namespace {

void random_kernel_cuda(TensorIterator& iter, uint64_t range, int64_t base, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_cuda", [&] {
    if (std::is_same<scalar_t, double>::value || std::is_same<scalar_t, int64_t>::value) {
      // define lambda to mod with range and add base
      auto random_func = [range, base] __device__ (uint64_t rand) {
        return static_cast<int64_t>(rand % range + base);
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
        return static_cast<int32_t>(rand % static_cast<uint32_t>(range) + static_cast<int32_t>(base));
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

} // namespace

namespace at { namespace native {

Tensor& random_cuda_(Tensor& self, Generator* gen) {
  auto iter = TensorIterator::nullary_op(self);
  uint64_t range;
  auto iter_scalar_type = iter.dtype();
  if (isFloatingType(iter_scalar_type)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter_scalar_type, "random_cuda_range_calc", [&] {
      range = static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1);
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter_scalar_type, "random_cuda_range_calc", [&] {
      range = static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1;
    });
  }
  random_kernel_cuda(iter, range, 0, gen);
  return self;
}

Tensor& clamped_random_cuda_(Tensor& self, int64_t from, int64_t to, Generator* gen) {
  TORCH_CHECK(from < to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
  auto iter = TensorIterator::nullary_op(self);
  uint64_t range = to - from;
  random_kernel_cuda(iter, range, from, gen);
  return self;
}

Tensor& capped_random_cuda_(Tensor& self, int64_t to, Generator* gen) {
  return clamped_random_cuda_(self, 0, to, gen);
}

}} // namespace at::native

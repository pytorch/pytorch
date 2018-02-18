#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <utility>

#include <TH/THAtomic.h>

#include <THC/THCGeneral.h>
#include <THC/THCHalf.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorRandom.h>

THCGenerator* THCRandom_getGenerator(THCState* state);

namespace at {
namespace native {

namespace dist {
  std::pair<uint64_t, uint64_t> next_philox_seed(Generator *gen) {
    auto gen_ = THCRandom_getGenerator(at::globalContext().thc_state);
    uint64_t offset = THAtomicAddLong(&gen_->philox_seed_offset, 1);
    return std::make_pair(gen_->initial_seed, offset);
  }

  template <typename scalar>
  struct PoissonOpCUDA {
    static void apply(Tensor& ret, const Tensor& lambda, std::pair<uint64_t, uint64_t> seeds) {
      at::cuda::CUDA_tensor_apply2<scalar, float>(ret, lambda,
        [seeds] __device__ (scalar& ret_val, const float& lambda, bool early_exit) {
          curandStatePhilox4_32_10_t state;
          curand_init(seeds.first, blockIdx.x * blockDim.x + threadIdx.x, seeds.second, &state);
          ret_val = scalar_cast<scalar>(curand_poisson(&state, lambda));
        }
      );
    }
  };

} // at::native::dist

Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen) {
  Tensor ret = lambda.type().tensor(lambda.sizes());
  auto lambda_ = lambda.toType(ScalarType::Float);
  dispatch_floating_types<void, dist::PoissonOpCUDA>(ret.type(), "poisson", ret, lambda_, dist::next_philox_seed(gen));
  return ret;
}

} // at::native
} // at

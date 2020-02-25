#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/CUDAGenerator.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/macros/Macros.h>
#include <curand_kernel.h>

#include <THC/THCGeneral.h>

namespace at{
namespace native{

namespace {

// philox generates 128 bits of randomness at a time. Kernel uses this explicitly by putting suitably transformed result into float4
// for all members of float4 to be consumed UNROLL has to be 4. Don't change!
const int UNROLL = 4;

template <
          typename scalar_t,
          typename accscalar_t,
          typename IndexType,
          int ADims>
#if __CUDA_ARCH__ >= 350
C10_LAUNCH_BOUNDS_2(256, 8)
#elif defined (__HIP_PLATFORM_HCC__)
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void
fused_dropout_kernel(cuda::detail::TensorInfo<scalar_t, IndexType> a,
                      cuda::detail::TensorInfo<scalar_t, IndexType> b,
                      cuda::detail::TensorInfo<uint8_t, IndexType> c,
                      IndexType totalElements, accscalar_t p, std::pair<uint64_t, uint64_t> seeds
                      ) {

  accscalar_t pinv = accscalar_t(1)/p;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
    curand_init(
        seeds.first,
        idx,
        seeds.second,
        &state);
  IndexType rounded_size = ((totalElements - 1)/(blockDim.x * gridDim.x * UNROLL)+1) *
        blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx;
       linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x*UNROLL) {
//curand_uniform_double was pure evil anyway, not doing what it promises, and there's nothing for halfs, so generate float for everything
       float4 rand = curand_uniform4(&state);
       scalar_t src[UNROLL];
       rand.x = rand.x < p;
       rand.y = rand.y < p;
       rand.z = rand.z < p;
       rand.w = rand.w < p;
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
    // Convert `linearIndex` into an offset of `a`
               const IndexType aOffset =
                   cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, a);
               src[ii] = a.data[aOffset];
           }
       }
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
    // Convert `linearIndex` into an offset of `b`
               const IndexType bOffset =
                   cuda::detail::IndexToOffset<scalar_t, IndexType, 1>::get(li, b);
               b.data[bOffset] = src[ii]*(&rand.x)[ii]*pinv;
               c.data[bOffset] = (uint8_t)(&rand.x)[ii];
           }
       }
       __syncthreads();
  }
}

template<typename scalar_t, typename accscalar_t>
void masked_scale_kernel(at::Tensor& ret, const at::Tensor src, const at::Tensor mask, accscalar_t scale){
   at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, uint8_t>(ret, src, mask, [scale]__device__(scalar_t& ret_val, const scalar_t& src_val, const uint8_t mask_val){
       ret_val = (float)mask_val * src_val * scale;
  });
}
} //anonymous namespace

std::tuple<Tensor,Tensor>
fused_dropout_cuda(const Tensor& self, double p, Generator * gen_){
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  Tensor ret = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor mask = at::empty(self.sizes(), self.options().dtype(kByte));
  const int64_t nelem = self.numel();
//empty tensors should not get here, but just in case, avoid FPE
  if (nelem==0) return std::tuple<Tensor,Tensor>(self, mask);
  const int64_t block_size = 256;
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  dim3 dim_block(block_size);
  dim3 grid((nelem + block_size -1)/block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
//number of times random will be generated per thread, to offset philox counter in thc random state
  int64_t counter_offset = ((nelem - 1)/(block_size*grid.x*UNROLL)+1)*UNROLL;
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
  }
  if (cuda::detail::canUse32BitIndexMath(self)){
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "fused_dropout", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      accscalar_t pa = (accscalar_t)(p);
      auto self_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(self);
      auto ret_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(ret);
      auto mask_info = cuda::detail::getTensorInfo<uint8_t, unsigned int>(mask);
      self_info.collapseDims();
      ret_info.collapseDims();
      mask_info.collapseDims(); //ret and mask are collapsed to 1d contiguous tensor
      switch (self_info.dims) {
        case 1:
            fused_dropout_kernel<scalar_t, accscalar_t, unsigned int, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(self_info, ret_info, mask_info, nelem, pa, rng_engine_inputs);
            break;
        default:
            fused_dropout_kernel<scalar_t, accscalar_t, unsigned int, -1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(self_info, ret_info, mask_info, nelem, pa, rng_engine_inputs);
      }
   });
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "fused_dropout", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      accscalar_t pa = (accscalar_t)(p);
      auto self_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(self);
      auto ret_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(ret);
      auto mask_info = cuda::detail::getTensorInfo<uint8_t, uint64_t>(mask);
      self_info.collapseDims();
      ret_info.collapseDims();
      mask_info.collapseDims(); //ret and mask are collapsed to 1d contiguous tensor
      switch (self_info.dims) {
        case 1:
            fused_dropout_kernel<scalar_t, accscalar_t, uint64_t, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(self_info, ret_info, mask_info, nelem, pa, rng_engine_inputs);
            break;
        default:
            fused_dropout_kernel<scalar_t, accscalar_t, uint64_t, -1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(self_info, ret_info, mask_info, nelem, pa, rng_engine_inputs);
      }
   });
  }
  THCudaCheck(cudaGetLastError());
  return std::tuple<Tensor,Tensor>(ret, mask);
}

Tensor masked_scale_cuda(const Tensor& self, const Tensor& mask, double scale){
   Tensor ret = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
   TORCH_CHECK(mask.scalar_type() == at::ScalarType::Byte, "mask should be torch.uint8 dtype");
   AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.scalar_type(), "masked_scale", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      accscalar_t pa = (accscalar_t)(scale);
    masked_scale_kernel<scalar_t>(ret, self, mask, pa);
  });
  return ret;
}

}
}

#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include "ATen/cuda/detail/TensorInfo.cuh"
#include "ATen/cuda/PhiloxRNGEngine.h"

#include <THC/THCGeneral.h>

namespace at{
namespace native{

namespace {

const int UNROLL = 4;

template <
          typename scalar_t,
          typename accscalar_t,
          typename IndexType,
          int ADims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(256,8)
#endif
__global__ void
fused_dropout_kernel(cuda::detail::TensorInfo<scalar_t, IndexType> a,
                      cuda::detail::TensorInfo<scalar_t, IndexType> b,
                      cuda::detail::TensorInfo<uint8_t, IndexType> c,
                      IndexType totalElements, accscalar_t p, std::pair<uint64_t, uint64_t> seeds
                      ) {

  accscalar_t pinv = accscalar_t(1)/p;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  at::cuda::Philox4_32_10 engine(seeds.first, idx, seeds.second);
  IndexType rounded_size = ((totalElements - 1)/(blockDim.x * gridDim.x * UNROLL)+1) * 
        blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx;
       linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x*UNROLL) {
       scalar_t src[UNROLL];
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
               float randn = at::cuda::standard_uniform_distribution(engine);
               randn = randn < p;
    // Convert `linearIndex` into an offset of `b`
               const IndexType bOffset =
                   cuda::detail::IndexToOffset<scalar_t, IndexType, 1>::get(li, b);
               b.data[bOffset] = src[ii]*randn*pinv;
               c.data[bOffset] = (uint8_t)randn;
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
fused_dropout_cuda(const Tensor& self, double p, Generator * gen){
  Tensor ret = at::empty_like(self);
  Tensor mask = at::empty(self.sizes(), self.options().dtype(kByte));
  const int64_t nelem = self.numel();
  const int64_t block_size = 256;
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  dim3 dim_block(block_size);
  dim3 grid((nelem + block_size -1)/block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
  auto gen_ = detail::checkGeneratorWithDefault(gen, &detail::getDefaultGenerator(kCUDA));
  if (cuda::detail::canUse32BitIndexMath(self)){
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "fused_dropout", [&] {
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
            fused_dropout_kernel<scalar_t, accscalar_t, unsigned int, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                self_info, 
                ret_info, 
                mask_info, 
                nelem, 
                pa, 
                gen_->incrementPhiloxOffset(nelem, grid.x, block_size, 4)); /* Loop unrolling 4 and engine call 1*/
            break;
        default:
            fused_dropout_kernel<scalar_t, accscalar_t, unsigned int, -1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                self_info, 
                ret_info, 
                mask_info, 
                nelem, 
                pa, 
                gen_->incrementPhiloxOffset(nelem, grid.x, block_size, 4));
      }
   });
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "fused_dropout", [&] {
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
            fused_dropout_kernel<scalar_t, accscalar_t, uint64_t, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                self_info, 
                ret_info, 
                mask_info, 
                nelem, 
                pa, 
                gen_->incrementPhiloxOffset(nelem, grid.x, block_size, 4));
            break;
        default:
            fused_dropout_kernel<scalar_t, accscalar_t, uint64_t, -1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                self_info, 
                ret_info, 
                mask_info, 
                nelem, 
                pa, 
                gen_->incrementPhiloxOffset(nelem, grid.x, block_size, 4));
      }
   });
  }
  THCudaCheck(cudaGetLastError());
  return std::tuple<Tensor,Tensor>(ret, mask);
}

Tensor masked_scale_cuda(const Tensor& self, const Tensor& mask, double scale){
   Tensor ret = at::empty_like(self);
   AT_CHECK(mask.type().scalarType() == at::ScalarType::Byte, "mask should be torch.uint8 dtype");
   AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.type(), "masked_scale", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      accscalar_t pa = (accscalar_t)(scale);
    masked_scale_kernel<scalar_t>(ret, self, mask, pa);
  });
  return ret;
}

}
}

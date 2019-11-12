#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/ScatterGather.h>
#include <ATen/native/cuda/ScatterGather.cuh>

namespace at { namespace native {
namespace {

void gather_kernel_cuda(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) {
  int64_t numel = index.numel();
  int64_t block = 512;
  int64_t grid = std::min<int64_t>((numel + block - 1) / block, 2048L);

  if (numel > 0) {
    AT_DISPATCH_ALL_TYPES_AND2(ScalarType::Bool, ScalarType::Half, self.scalar_type(), "gather_out_cuda", [&](){
      auto stream = at::cuda::getCurrentCUDAStream();
      if (cuda::detail::canUse32BitIndexMath(result) &&
          cuda::detail::canUse32BitIndexMath(self) &&
          cuda::detail::canUse32BitIndexMath(index)) {
        auto result_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(result);
        auto self_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(self);
        auto index_info = cuda::detail::getTensorInfo<int64_t, unsigned int>(index);

        // Specialize for a small number of dimensions.
        switch (index_info.dims) {
          case 1:
            gather_kernel<unsigned int, scalar_t, 1><<<grid, block, stream>>>(
              result_info, self_info, index_info, dim, static_cast<unsigned int>(numel));
            break;
          case 2:
            gather_kernel<unsigned int, scalar_t, 2><<<grid, block, stream>>>(
              result_info, self_info, index_info, dim, static_cast<unsigned int>(numel));
            break;
          case 3:
            gather_kernel<unsigned int, scalar_t, 3><<<grid, block, stream>>>(
              result_info, self_info, index_info, dim, static_cast<unsigned int>(numel));
            break;
          default:
            gather_kernel<unsigned int, scalar_t, -1><<<grid, block, stream>>>(
              result_info, self_info, index_info, dim, static_cast<unsigned int>(numel));
            break;
        }
      } else {
        auto result_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(result);
        auto self_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(self);
        auto index_info = cuda::detail::getTensorInfo<int64_t, uint64_t>(index);
        gather_kernel<uint64_t, scalar_t, -1><<<grid, block, stream>>>(
          result_info, self_info, index_info, dim, static_cast<uint64_t>(numel));
      }
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(gather_stub, &gather_kernel_cuda);

}}  // namespace at::native

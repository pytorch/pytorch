#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/cuda/ScatterGatherKernel.cu>

namespace at { namespace native {
namespace {
  void scatter_cuda_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    int64_t numel = index.numel();
    int64_t block = 512;
    int64_t grid = std::min<int64_t>((numel + block - 1) / block, 2048L);

    if (self.is_non_overlapping_and_dense()) {
      std::cout << "self is non overlapping and dense.\n";
    }
    if (numel > 0) {
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::Bool,
        ScalarType::Half,
        self.scalar_type(),
        "scatter_cuda_", [&]() {
        auto stream = at::cuda::getCurrentCUDAStream();
        if (cuda::detail::canUse32BitIndexMath(self) &&
          cuda::detail::canUse32BitIndexMath(src) &&
          cuda::detail::canUse32BitIndexMath(index)) {
          auto self_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(self);
          auto index_info = cuda::detail::getTensorInfo<int64_t, unsigned int>(index);
          auto src_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(src);

          switch(index_info.dims) {
          case 1:
            scatter_kernel<unsigned int, scalar_t, 1><<<grid, block, 0, stream>>>(
              self_info, src_info, index_info, dim, numel);
            break;
          case 2:
            scatter_kernel<unsigned int, scalar_t, 2><<<grid, block, 0, stream>>>(
              self_info, src_info, index_info, dim, numel);
            break;
          case 3:
            scatter_kernel<unsigned int, scalar_t, 3><<<grid, block, 0, stream>>>(
              self_info, src_info, index_info, dim, numel);
            break;
          default:
            scatter_kernel<unsigned int, scalar_t, -1><<<grid, block, 0, stream>>>(
              self_info, src_info, index_info, dim, numel);
          }
        }
        else {
          auto self_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(self);
          auto index_info = cuda::detail::getTensorInfo<int64_t, uint64_t>(index);
          auto src_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(src);
          scatter_kernel<uint64_t, scalar_t, -1><<<grid, block, 0, stream>>>(
            self_info, src_info, index_info, dim, numel);
        }
      });
    }
  }

  void scatter_fill_cuda_(Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    int64_t numel = index.numel();
    int64_t block = 512;
    int64_t grid = std::min<int64_t>((numel + block - 1) / block, 2048L);

    if (numel > 0) {
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::Bool,
        ScalarType::Half,
        self.scalar_type(),
        "scatter_fill_cuda_", [&]() {
        auto stream = at::cuda::getCurrentCUDAStream();
        if (cuda::detail::canUse32BitIndexMath(self) &&
          cuda::detail::canUse32BitIndexMath(index)) {
          auto self_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(self);
          auto index_info = cuda::detail::getTensorInfo<int64_t, unsigned int>(index);
          auto src = value.to<scalar_t>();

          switch(index_info.dims) {
          case 1:
            scatter_fill_kernel<unsigned int, scalar_t, 1><<<grid, block, 0, stream>>>(
              self_info, src, index_info, dim, numel);
            break;
          case 2:
            scatter_fill_kernel<unsigned int, scalar_t, 2><<<grid, block, 0, stream>>>(
              self_info, src, index_info, dim, numel);
            break;
          case 3:
            scatter_fill_kernel<unsigned int, scalar_t, 3><<<grid, block, 0, stream>>>(
              self_info, src, index_info, dim, numel);
            break;
          default:
            scatter_fill_kernel<unsigned int, scalar_t, -1><<<grid, block, 0, stream>>>(
              self_info, src, index_info, dim, numel);
          }
        }
        else {
          auto self_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(self);
          auto index_info = cuda::detail::getTensorInfo<int64_t, uint64_t>(index);
          auto src = value.to<scalar_t>();
          scatter_fill_kernel<uint64_t, scalar_t, -1><<<grid, block, 0, stream>>>(
            self_info, src, index_info, dim, numel);
        }
      });
    }
  }
} // anonymous namespace
    
REGISTER_DISPATCH(scatter_stub, &scatter_cuda_);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cuda_);
}}                            // namespace at::native


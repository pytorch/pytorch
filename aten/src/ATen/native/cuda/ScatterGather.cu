#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/cuda/ScatterGatherKernel.cuh>

namespace at { namespace native {
namespace {
  void scatter_cuda_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    int64_t numel = index.numel();
    int64_t block = 512;
    int64_t grid = std::min<int64_t>((numel + block - 1) / block, 2048L);

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
        }
      });
    }
  }

  void scatter_fill_cuda_(Tensor& self, int64_t dim, const Tensor& index, Scalar src) {
  }
} // anonymous namespace
    
REGISTER_DISPATCH(scatter_stub, &scatter_cuda_);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cuda_);
}}                            // namespace at::native


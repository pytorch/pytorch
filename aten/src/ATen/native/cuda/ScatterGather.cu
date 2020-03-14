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
      
    }
  }

  void scatter_fill_cuda_(Tensor& self, int64_t dim, const Tensor& index, Scalar src) {
  }
} // anonymous namespace
    
REGISTER_DISPATCH(scatter_stub, &scatter_cuda_);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cuda_);
}}                            // namespace at::native


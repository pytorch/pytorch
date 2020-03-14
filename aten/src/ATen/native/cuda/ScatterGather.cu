#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorAdvancedIndexing.h>

namespace at { namespace native {
namespace {
  void scatter_cuda_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  }

  void scatter_fill_cuda_(Tensor& self, int64_t dim, const Tensor& index, Scalar src) {
  }
} // anonymous namespace
    
REGISTER_DISPATCH(scatter_stub, &scatter_cuda_);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cuda_);
}}                            // namespace at::native


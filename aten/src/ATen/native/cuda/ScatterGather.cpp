#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCUDA.h>

namespace at { namespace native {

Tensor & scatter__cuda(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  return legacy::cuda::_th_scatter_(self, dim, index, src);
}

Tensor & scatter__cuda(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return legacy::cuda::_th_scatter_(self, dim, index, value);
}

Tensor & scatter_add__cuda(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  return legacy::cuda::_th_scatter_add_(self, dim, index, src);
}

Tensor & scatter_add__cuda(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return scatter_add__cuda(self, dim, index, at::tensor(value, self.options()));
}

Tensor & gather_out_cuda(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cuda::_th_gather_out(result, self, dim, index);
}

Tensor gather_cuda(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cuda::_th_gather(self, dim, index);
}

}} // at::native
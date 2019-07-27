#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCPU.h>

namespace at { namespace native {

Tensor & scatter__cpu(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  return legacy::cpu::_th_scatter_(self, dim, index, src);
}

Tensor & scatter__cpu(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return legacy::cpu::_th_scatter_(self, dim, index, value);
}

Tensor & scatter_add__cpu(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  return legacy::cpu::_th_scatter_add_(self, dim, index, src);
}

Tensor & scatter_add__cpu(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return scatter_add__cpu(self, dim, index, at::full({}, value, self.options()));
}

Tensor & gather_out_cpu(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cpu::_th_gather_out(result, self, dim, index);
}

Tensor gather_cpu(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cpu::_th_gather(self, dim, index);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone().scatter_(dim, index, source);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return self.clone().scatter_(dim, index, value);
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone().scatter_add_(dim, index, source);
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return self.clone().scatter_add_(dim, index, value);
}

Tensor _gather_sparse_backward(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& grad){
// special case scalar input and/or index
    if (self.ndimension() == 0) return at::_sparse_coo_tensor_unsafe(at::empty({0,grad.numel()}, index.options()), grad, self.sizes());
    if (grad.ndimension() == 0) return at::_sparse_coo_tensor_unsafe(index.view({1,1}), grad, self.sizes());
    Tensor sparse_ind = at::empty({self.ndimension(), grad.numel()}, self.options().dtype(at::kLong));
    int64_t n_above = grad.numel();
    int64_t n_below = 1;
    if (dim < 0) dim += self.ndimension();
    for (int i=0; i<self.ndimension(); i++) {
        n_above /= grad.size(i);
        if (i == dim) {
            sparse_ind[i] = index.reshape(-1);
        } else {
            sparse_ind[i] = at::arange(grad.size(i),self.options().dtype(at::kLong)).unsqueeze(1).expand({grad.size(i), n_above}).reshape(-1).repeat(n_below);
        }
        n_below *= grad.size(i);
    }
    return at::_sparse_coo_tensor_unsafe(sparse_ind, grad.reshape(-1), self.sizes());
}

}} // at::native
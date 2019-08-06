#include <tuple>
#include <algorithm>
#include <ATen/ATen.h>
#include <c10/core/WrapDimMinimal.h>

namespace at { namespace native {

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone().scatter_(dim, index, source);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return self.clone().scatter_(dim, index, value);
}

Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  if (index.numel() == 0) {
    return self;
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return at::_scatter_add_(self, dim, index, source);
  }
  if (source.dim() == 0) {
    dim = c10::maybe_wrap_dim(dim, index.dim());
    std::vector<int64_t> source_sizes = self.sizes().vec();
    source_sizes[dim] = index.size(dim);
    return at::_scatter_add_(self, dim, index, source.expand(source_sizes));
  }
  return at::_scatter_add_(self, dim, index, source);
}

Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return self.scatter_add_(dim, index, at::full({}, value, self.options()));
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone().scatter_add_(dim, index, source);
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return self.scatter_add(dim, index, at::full({}, value, self.options()));
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

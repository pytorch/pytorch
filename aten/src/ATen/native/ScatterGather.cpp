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
  // The legacy scatter_add_ does not support scalar source, so scalar source should be manually
  // broadcasted to the correct shape, so that it could be handled by legacy scatter_add.
  // The general rule of broadcasting source is, for dimenisons d!=dim, the sizes of source should
  // match with self and for dimension d==dim, the size of source should match with index.
  //
  // The general rule does not always apply. There are special cases that could not be treated as usual.
  // Things to worry are: self, index, source could be scalar tensor, and index could be empty.
  //
  // When one of self or index is scalar, then the other and source needs to be either scalar
  // or shape (1,) vector. Also, dim has to be 0. These are all handled well in legacy scatter_add_
  //
  // Empty index when source is not scalar is also handled well in legacy scatter_add_, but need special
  // treatment when source is scalar.
  if (source.dim() > 0 || self.dim() == 0 || index.dim() == 0) {
    return at::_legacy_scatter_add_(self, dim, index, source);
  }
  if (index.numel() == 0) {
    return self;
  }
  dim = c10::maybe_wrap_dim(dim, index.dim());
  std::vector<int64_t> source_sizes = self.sizes().vec();
  source_sizes[dim] = index.size(dim);
  return at::_legacy_scatter_add_(self, dim, index, source.expand(source_sizes));
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

#include <tuple>
#include <algorithm>
#include <ATen/ATen.h>
#include <c10/core/WrapDimMinimal.h>

namespace {

inline void expand_size(int64_t dim, int64_t &size1, int64_t &size2) {
  TORCH_CHECK(size1 == 1 || size2 == 1, "Size mismatch at dim=", dim, ", get: ", size1, " and ", size2);
  size1 = size2 = size1 + size2 - 1;
}

std::tuple<std::vector<int64_t>, at::Tensor, at::Tensor>
inline expand3(const at::Tensor &self, int64_t &dim, at::Tensor index, const at::Tensor &src) {
  std::vector<int64_t> self_sizes = self.sizes().vec();
  std::vector<int64_t> index_sizes = index.sizes().vec();
  std::vector<int64_t> src_sizes = src.sizes().vec();
  if (src_sizes.size() == 0) {  // when src is a scalar tensor
    src_sizes = std::vector<int64_t>(self_sizes.size());
    std::fill(src_sizes.begin(), src_sizes.end(), 1);
  }
  TORCH_CHECK(self_sizes.size() == src_sizes.size(), "torch.scatter requires src and dest to have the same number of dimensions");
  TORCH_CHECK(index_sizes.size() <= src_sizes.size(), "torch.scatter requires src to have more or equal dimensions than index");
  dim = c10::maybe_wrap_dim(dim, index_sizes.size());
  for (int64_t i = 0; i < self_sizes.size(); i++) {
    if (i == dim) {
      if (src_sizes[i] != index_sizes[i]) {
        expand_size(i, index_sizes[i], src_sizes[i]);
      }
    } else if (i < index_sizes.size()) {
      int64_t expanded_size = 1;
      for (int64_t s : {self_sizes[i], index_sizes[i], src_sizes[i]}) {
        if (s != 1) {
          if (expanded_size == 1) {
            expanded_size = s;
          } else {
            AT_CHECK(expanded_size == s, "Size mismatch at dim=", dim, ", get: ", self_sizes[i], ", ", index_sizes[i], " and ", src_sizes[i]);
          }
        }
      }
      self_sizes[i] = index_sizes[i] = src_sizes[i] = expanded_size;
    } else {
      if (src_sizes[i] != self_sizes[i]) {
        expand_size(i, src_sizes[i], self_sizes[i]);
      }
      index = index.unsqueeze(-1);
      index_sizes.push_back(src_sizes[i]);
    }
  }
  return std::make_tuple(self_sizes, index.expand(index_sizes), src.expand(src_sizes));
}

std::tuple<std::vector<int64_t>, at::Tensor>
inline expand2(const at::Tensor &self, int64_t &dim, at::Tensor index) {
  std::vector<int64_t> self_sizes = self.sizes().vec();
  std::vector<int64_t> index_sizes = index.sizes().vec();
  TORCH_CHECK(self_sizes.size() >= index_sizes.size(), "requires input to have more or equal dimensions than index");
  dim = c10::maybe_wrap_dim(dim, index_sizes.size());
  for(int64_t i = 0; i < self_sizes.size(); i++) {
    if (i == dim) {
      continue;
    } else if (i < index_sizes.size()) {
      if (self_sizes[i] != index_sizes[i]) {
        expand_size(i, index_sizes[i], self_sizes[i]);
      }
    } else {
      index = index.unsqueeze(-1);
      index_sizes.push_back(self_sizes[i]);
    }
  }
  return std::make_tuple(self_sizes, index.expand(index_sizes));
}

}  // namespace

namespace at { namespace native {

Tensor & gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  if (self.dim() == 0 || index.dim() == 0) {
    return at::_gather_out(result, self, dim, index, sparse_grad);
  }
  std::vector<int64_t> self_sizes;
  Tensor expanded_index;
  std::tie(self_sizes, expanded_index) = expand2(self, dim, index);
  return at::_gather_out(result, self.expand(self_sizes), dim, expanded_index, sparse_grad);
}

Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  if (self.dim() == 0 || index.dim() == 0) {
    return at::_gather(self, dim, index, sparse_grad);
  }
  std::vector<int64_t> self_sizes;
  Tensor expanded_index;
  std::tie(self_sizes, expanded_index) = expand2(self, dim, index);
  return at::_gather(self.expand(self_sizes), dim, expanded_index, sparse_grad);
}

Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  if (index.numel() == 0) {
    return self;
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return at::_scatter_(self, dim, index, source);
  }
  Tensor expanded_source, expanded_index;
  std::vector<int64_t> self_sizes;
  std::tie(self_sizes, expanded_index, expanded_source) = expand3(self, dim, index, source);
  TORCH_CHECK(self_sizes == self.sizes(), "broadcasting change the shape of self");
  return at::_scatter_(self, dim, expanded_index, expanded_source);
}

Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  if (index.numel() == 0) {
    return self;
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return at::_scatter_(self, dim, index, value);
  }
  Tensor expanded_index;
  std::vector<int64_t> result_sizes;
  std::tie(result_sizes, expanded_index) = expand2(self, dim, index);
  TORCH_CHECK(result_sizes == self.sizes(), "broadcasting change the shape of self");
  return at::_scatter_(self, dim, expanded_index, value);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  if (index.numel() == 0) {
    return self.clone();
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return self.clone().scatter_(dim, index, source);
  }
  Tensor expanded_source, expanded_index;
  std::vector<int64_t> self_sizes;
  std::tie(self_sizes, expanded_index, expanded_source) = expand3(self, dim, index, source);
  Tensor ret = self.clone().expand(self_sizes).contiguous();
  return at::_scatter_(ret, dim, expanded_index, expanded_source);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  if (index.numel() == 0) {
    return self.clone();
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return self.clone().scatter_(dim, index, value);
  }
  Tensor expanded_index;
  std::vector<int64_t> result_sizes;
  std::tie(result_sizes, expanded_index) = expand2(self, dim, index);
  Tensor ret = self.clone().expand(result_sizes).contiguous();
  return at::_scatter_(ret, dim, expanded_index, value);
}

Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  if (index.numel() == 0) {
    return self;
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return at::_scatter_add_(self, dim, index, source);
  }
  Tensor expanded_source, expanded_index;
  std::vector<int64_t> self_sizes;
  std::tie(self_sizes, expanded_index, expanded_source) = expand3(self, dim, index, source);
  TORCH_CHECK(self_sizes == self.sizes(), "broadcasting change the shape of self");
  return at::_scatter_add_(self, dim, expanded_index, expanded_source);
}

Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return self.scatter_add_(dim, index, at::full({}, value, self.options()));
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  if (index.numel() == 0) {
    return self.clone();
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return self.clone().scatter_add_(dim, index, source);
  }
  Tensor expanded_source, expanded_index;
  std::vector<int64_t> self_sizes;
  std::tie(self_sizes, expanded_index, expanded_source) = expand3(self, dim, index, source);
  Tensor ret = self.clone().expand(self_sizes).contiguous();
  return at::_scatter_add_(ret, dim, expanded_index, expanded_source);
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return at::scatter_add(self, dim, index, at::full({}, value, self.options()));
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

#include <ATen/ATen.h>
#include <ATen/native/ScatterGather.h>

namespace at { namespace native {

DEFINE_DISPATCH(gather_stub);

Tensor & gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool) {
  int64_t num_dims = std::max<int64_t>(self.dim(), 1);
  TORCH_CHECK(std::max<int64_t>(index.dim(), 1) == num_dims, "Index tensor must have same dimensions as input tensor");
  dim = c10::maybe_wrap_dim(dim, self.dim());

  std::vector<int64_t> self_sizes = self.sizes().vec();
  std::vector<int64_t> index_sizes = index.sizes().vec();
  ensure_nonempty(self_sizes);
  ensure_nonempty(index_sizes);

  for(int64_t i = 0; i < num_dims; i++) {
    if(i != dim) {
      TORCH_CHECK(index_sizes[i] == self_sizes[i], "Size does not match at dimension ", i, " get ", self_sizes[i], " vs ", index_sizes[i]);
    }
  }
  if (result.defined()) {
    result.resize_as_(index);
  } else {
    result = at::empty(index.sizes(), self.options());
  }
  gather_stub(result.device().type(), result, self, dim, index);
  return result;
}

Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  Tensor result;
  return at::gather_out(result, self, dim, index, sparse_grad);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).scatter_(dim, index, source);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar source) {
  return self.clone(at::MemoryFormat::Preserve).scatter_(dim, index, source);
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).scatter_add_(dim, index, source);
}


}}  // namespace at::native

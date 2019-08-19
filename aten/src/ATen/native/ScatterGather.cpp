#include <ATen/ATen.h>

namespace at { namespace native {

template <typename scalar_t>
Tensor & gather_out_cpu(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  int64_t num_dims = std::max<int64_t>(self.dim(), 1);
  TORCH_CHECK(std::max<int64_t>(index.dim(), 1) == num_dims, "Index tensor must have same dimensions as input tensor");
  TORCH_CHECK(dim >= 0 && dim < num_dims, "Index dimension is out of bounds");
  TORCH_CHECK(std::max<int64_t>(result.dim(), 1) == num_dims, "Input tensor must have same dimensions as output tensor");

  int64_t elems_per_row = (index.dim() == 0 ? 1 : index.size(dim));
  int64_t src_dim_size = src.size(dim);
  int64_t outer_size = 1;
  for(int64_t i = 0; i < num_dims; i++) {
    if(i != dim) {
      AT_CHECK(index.size(i) == self.size(i), "Size does not match at dimension ", i, " get ", self.size(i), " vs ", index.size(i));
      outer_size *= index.size(i);
    }
  }
  result.resize_as_(index);
  scalar_t *result_data = result.data<scalar_t>();
  scalar_t *self_data = self.data<scalar_t>();
  int64_t *index_data = index.data<int64_t>();
  int64_t result_dim_stride = result.stride(dim);
  int64_t index_dim_stride = index.stride(dim);
  int64_t self_dim_stride = self.stride(dim);

  at::parallel_for(0, outer_size, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for(int64_t i = begin; i < end; i++) {
      scalar_t *result_base = result_data;
      int64_t *index_base = index_data;
      scalar_t *self_base = self_data;
      for(int64_t k = 0; k < num_dims; k++) {
        if(dim != k) {
          int64_t index_at_k = i % result.size(k);
          result_base += result.stride(k) * index_at_k;
          index_base += index.stride(k) * index_at_k;
          self_base += self.stride(k) * index_at_k;
          i /= result.size(k);
        }
      }
      for(int64_t j = 0; j < elems_per_row; j++) {
        AT_CHECK(j >= 0 && j < src_dim_size, "Invalid index in gather: out of range");
        int64_t index = *(index_base + j * index_dim_stride);
        *(result_base + j * result_dim_stride) = *(self_base + index * self_dim_stride);
      }
    }
  });
  return result;
}

Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  Tensor result = at::empty({}, self.options());
  return at::gather_out(result, self, dim, index, sparse_grad);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone().scatter_(dim, index, source);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar source) {
  return self.clone().scatter_(dim, index, source);
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone().scatter_add_(dim, index, source);
}

}}  // namespace at::native

#include "ATen/ATen.h"

namespace at { namespace native {

Tensor & gather_out_cpu(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool) {
  int64_t num_dims = std::max<int64_t>(self.dim(), 1);
  TORCH_CHECK(std::max<int64_t>(index.dim(), 1) == num_dims, "Index tensor must have same dimensions as input tensor");
  dim = c10::maybe_wrap_dim(dim, self.dim());

  int64_t elems_per_row = (index.dim() == 0 ? 1 : index.size(dim));
  int64_t self_dim_size = self.size(dim);
  int64_t outer_size = 1;
  for(int64_t i = 0; i < num_dims; i++) {
    if(i != dim) {
      TORCH_CHECK(index.size(i) == self.size(i), "Size does not match at dimension ", i, " get ", self.size(i), " vs ", index.size(i));
      outer_size *= index.size(i);
    }
  }
  result.resize_as_(index);

  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "gather_out_cpu", [&](){
    scalar_t *result_data = result.data<scalar_t>();
    scalar_t *self_data = self.data<scalar_t>();
    int64_t *index_data = index.data<int64_t>();
    if (result.numel() == 0) {
      return;
    }
    bool finished = false;
    std::vector<int64_t> counter(num_dims, 0);

    int64_t result_dim_stride = result.stride(dim);
    int64_t index_dim_stride = index.stride(dim);
    int64_t self_dim_stride = self.stride(dim);

    while(!finished) {
      for(int64_t j = 0; j < elems_per_row; j++) {
        int64_t index_value = *(index_data + j * index_dim_stride);
        TORCH_CHECK(index_value >= 0 && index_value < self_dim_size, "Invalid index in gather: out of range");
        *(result_data + j * result_dim_stride) = *(self_data + index_value * self_dim_stride);
      }
      if(num_dims == 1) {
        break;
      }
      for(int64_t i = 0; i < num_dims; i++) {
        if(i == dim) {
          if(i == num_dims - 1) {
            finished = true;
            break;
          }
          continue;
        }
        counter[i]++;
        result_data += result.stride(i);
        self_data += self.stride(i);
        index_data += index.stride(i);
        if(counter[i] == result.size(i)) {
          if(i == num_dims - 1) {
            finished = true;
            break;
          }
          int64_t size = result.size(i);
          result_data -= size * result.stride(i);
          self_data -= size * self.stride(i);
          index_data -= size * index.stride(i);
          counter[i] = 0;
        } else {
          break;
        }
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

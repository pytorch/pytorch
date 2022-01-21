#include <ATen/ATen.h>

namespace at {
namespace native {

 // Check to see if the shape of tensors is compatible
 // for being concatenated along a given dimension.
inline void check_cat_shape_except_dim(const Tensor & first, const Tensor & second, int64_t dimension, int64_t index) {
   int64_t first_dims = first.dim();
   int64_t second_dims = second.dim();
   TORCH_CHECK(first_dims == second_dims, "Tensors must have same number of dimensions: got ",
               first_dims, " and ", second_dims);
   for (int64_t dim = 0; dim < first_dims; dim++) {
     if (dim == dimension) {
       continue;
     }
     int64_t first_dim_size = first.sizes()[dim];
     int64_t second_dim_size = second.sizes()[dim];
     TORCH_CHECK(first_dim_size == second_dim_size, "Sizes of tensors must match except in dimension ",
                 dimension, ". Expected size ", static_cast<long long>(first_dim_size), " but got size ", static_cast<long long>(second_dim_size), " for tensor number ", index, " in the list.");
   }
 }

inline void check_cat_no_zero_dim(at::ArrayRef<Tensor> tensors) {
  for(const auto i : c10::irange(tensors.size())) {
    auto& t = tensors[i];
    TORCH_CHECK(t.dim() > 0,
             "zero-dimensional tensor (at position ", i, ") cannot be concatenated");
  }
}

}} // namespace at::native

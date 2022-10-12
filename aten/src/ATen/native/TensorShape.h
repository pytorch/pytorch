#pragma once
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <ATen/core/IListRef.h>

namespace at {
namespace native {
inline bool cat_should_skip_tensor(const Tensor& t) {
  return t.numel() == 0 && t.dim() == 1;
}

 // Check to see if the shape of tensors is compatible
 // for being concatenated along a given dimension.
inline void check_cat_shape_except_dim(const Tensor & first, const Tensor & second, int64_t dimension, int64_t index) {
   int64_t first_dims = first.dim();
   int64_t second_dims = second.dim();
   TORCH_CHECK(first_dims == second_dims, "Tensors must have same number of dimensions: got ",
               first_dims, " and ", second_dims);
   for (const auto dim : c10::irange(first_dims)) {
     if (dim == dimension) {
       continue;
     }
     int64_t first_dim_size = first.sizes()[dim];
     int64_t second_dim_size = second.sizes()[dim];
     TORCH_CHECK(first_dim_size == second_dim_size, "Sizes of tensors must match except in dimension ",
                 dimension, ". Expected size ", static_cast<long long>(first_dim_size), " but got size ", static_cast<long long>(second_dim_size), " for tensor number ", index, " in the list.");
   }
 }

inline void check_cat_no_zero_dim(const MaterializedITensorListRef& tensors) {
  int64_t i = 0;
  for(const Tensor& t : tensors) {
    TORCH_CHECK(t.dim() > 0,
             "zero-dimensional tensor (at position ", i, ") cannot be concatenated");
    i++;
  }
}

inline int64_t get_num_splits(const Tensor& self, int64_t split_size, int64_t dim) {
  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  TORCH_CHECK(split_size >= 0,  "split expects split_size be non-negative, but got split_size=", split_size);
  int64_t dim_size = self.size(dim);
  TORCH_CHECK(split_size > 0 || dim_size == 0,
           "split_size can only be 0 if dimension size is 0, "
           "but got dimension size of ", dim_size);
  // if split_size is 0 and dimension size is 0, there is 1 split.
  int64_t num_splits = 1;
  if (split_size != 0) {
    // ensuring num_splits is at least 1 makes consistent the case where split_size > dim_size
    // (returns a single split).  We might want to error here, but keep it for BC.
    num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  }
  return num_splits;
}

///
/// For more information, see
/// https://pytorch.org/docs/master/generated/torch.Tensor.unfold.html#torch.Tensor.unfold
///

Tensor unfold(const Tensor& self, int64_t dimension, int64_t size, int64_t step);

}} // namespace at::native

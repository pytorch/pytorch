#pragma once

#include <c10/macros/Macros.h>
#include <ATen/NestedTensorImpl.h>

#include <vector>

namespace at {
namespace native {
struct NestedTensorImpl;

// TODO: cache this and only do it once per NestedTensor
int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt);

at::Tensor wrap_buffer(at::Tensor buffer, at::Tensor nested_size_tensor);

at::Tensor wrap_buffer(at::Tensor buffer, at::Tensor nested_size_tensor, at::Tensor nested_stride_tensor, const std::vector<int64_t>& offsets);

inline const at::Tensor& get_buffer(const at::Tensor& tensor) {
  return get_nested_tensor_impl(tensor)->get_buffer();
}

// The sizes of the underlying tensors
inline std::vector<IntArrayRef> NestedTensor_get_sizes(const NestedTensorImpl* self_ptr) {
  int64_t ntensors = self_ptr->size(0);
  std::vector<IntArrayRef> sizes(ntensors);
  if (ntensors == 0) {
    return sizes;
  }
  const Tensor& sizemat = self_ptr->get_nested_size_tensor();
  int64_t orig_dim = sizemat.size(1);
  // nesting scalars has empty sizes
  if (orig_dim == 0) {
    return sizes;
  }
  const int64_t* sizemat_ptr = sizemat.data_ptr<int64_t>();

  for(const auto i: c10::irange(ntensors)){
    sizes[i] = IntArrayRef(sizemat_ptr, sizemat_ptr + orig_dim);
    sizemat_ptr += orig_dim;
  }
  return sizes;
}

inline std::vector<IntArrayRef> NestedTensor_get_sizes(const at::Tensor& self) {
  const NestedTensorImpl* self_ptr = get_nested_tensor_impl(self);
  return NestedTensor_get_sizes(self_ptr);
}

// The strides of the underlying tensors
inline std::vector<IntArrayRef> NestedTensor_get_strides(const NestedTensorImpl* self_ptr) {
  int64_t ntensors = self_ptr->size(0);
  std::vector<IntArrayRef> strides(ntensors);
  if (ntensors == 0) {
    return strides;
  }
  const Tensor& stridemat = self_ptr->get_nested_stride_tensor();
  int64_t orig_dim = stridemat.size(1);
  // nesting scalars has empty strides
  if (orig_dim == 0) {
    return strides;
  }
  const int64_t* stridemat_ptr = stridemat.data_ptr<int64_t>();
  for(const auto i: c10::irange(ntensors)) {
    strides[i] = IntArrayRef(stridemat_ptr, stridemat_ptr + orig_dim);
    stridemat_ptr += orig_dim;
  }
  return strides;
}

inline std::vector<IntArrayRef> NestedTensor_get_strides(const at::Tensor& self) {
  const NestedTensorImpl* self_ptr = get_nested_tensor_impl(self);
  return NestedTensor_get_strides(self_ptr);
}

TORCH_API std::vector<int64_t> NestedTensor_get_max_size(const NestedTensorImpl& nt);

TORCH_API Tensor NestedTensor_to_padded_tensor_generic(const Tensor& t, double padding, OptionalIntArrayRef output_size);

} // namespace native
} // namespace at

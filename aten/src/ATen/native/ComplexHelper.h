#pragma once

#include <ATen/ATen.h>

// WARNING: this header contains non-inline functions and should be only
// included from ONE cpp file

namespace at { namespace native {

inline std::vector<int64_t> computeStrideForViewAsReal(IntArrayRef oldstride) {
  auto res = oldstride.vec();
  for(size_t i = 0; i < res.size(); i++) {
    res[i] = res[i] * 2;
  }
  res.emplace_back(1);
  return res;
}

// expects as input a complex tensor and returns back a tensor
// with corresponding real dtype containing the complex values
// in the last two dimensions
Tensor view_as_real(const Tensor& self) {
  TORCH_CHECK(!self.is_conj(), "view_as_real doesn't work on unresolved conjugated tensors.  To resolve the conjugate tensor so you can view it as real, use self.resolve_conj(); however, be warned that the resulting tensor will NOT alias the original.  To view the true underlying representation, use view_as_real_physical(); this will return a real tensor and a boolean telling you whether or not you should negate the imaginary component of the tensor.");
  return native::view_as_real_physical(self);
}

Tensor view_as_real_physical(const Tensor& self) {
  TORCH_CHECK(self.is_complex(), "view_as_real is only supported for complex tensors");
  auto new_sizes = self.sizes().vec();
  // last dimension will always have two elements containing the real and imag vals
  new_sizes.emplace_back(2);
  auto new_strides = computeStrideForViewAsReal(self.strides());
  auto new_storage_offset = 2 * self.storage_offset();
  const auto float_type = c10::toValueType(self.scalar_type());
  return at::empty({0}, self.options().dtype(float_type)).set_(self.storage(), new_storage_offset, new_sizes, new_strides);
}

bool is_conj(const Tensor& self) {
  return self.is_conj();
}

inline std::vector<int64_t> computeStrideForViewAsComplex(IntArrayRef oldstride) {
  auto res = oldstride.vec();
  int dim = res.size();

  TORCH_CHECK(res[dim-1] == 1, "Tensor must have a last dimension with stride 1");
  res.pop_back();

  for (auto i = decltype(res.size()){0}; i < res.size(); i++) {
    TORCH_CHECK(res[i] % 2 == 0, "Tensor must have a stride divisible by 2 for all but last dimension");
    res[i] = res[i] / 2;
  }
  return res;
}

// expects as input a float or double tensor with last dimension of size 2
// and returns back a tensor with corresponding complex dtype
Tensor view_as_complex(const Tensor& self) {
  TORCH_CHECK(
    self.scalar_type() == kFloat || self.scalar_type() == kDouble || self.scalar_type() == kHalf,
    "view_as_complex is only supported for half, float and double tensors, but got a tensor of scalar type: ", self.scalar_type());

  TORCH_CHECK(self.dim() != 0, "Input tensor must have one or more dimensions");
  auto new_sizes = self.sizes().vec();
  TORCH_CHECK(new_sizes[self.dim()-1] == 2, "Tensor must have a last dimension of size 2");
  new_sizes.pop_back();

  const auto new_strides = computeStrideForViewAsComplex(self.strides());
  const auto complex_type = c10::toComplexType(self.scalar_type());

  TORCH_CHECK(self.storage_offset() % 2 == 0, "Tensor must have a storage_offset divisible by 2");
  const auto new_storage_offset = self.storage_offset() / 2;

  return at::empty({0}, self.options().dtype(complex_type)).set_(self.storage(), new_storage_offset, new_sizes, new_strides);
}

}} // namespace at::native

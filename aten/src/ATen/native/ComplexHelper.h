#pragma once

#include <ATen/ATen.h>

namespace at { namespace native {

inline std::vector<int64_t> computeStrideForComplex(IntArrayRef oldstride) {
  auto res = oldstride.vec();
  for(size_t i = 0; i < res.size(); i++) {
    res[i] = res[i] * 2;
  }
  res.emplace_back(1);
  return res;
}

inline Tensor from_empty(
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options = {}) {
  AutoNonVariableTypeMode guard;
  auto storage = Storage(
      options.dtype(),
      detail::computeStorageSize(sizes, strides),
      DataPtr(nullptr, options.device()),
      /*allocator=*/nullptr,
      /*resizable=*/false);
  return at::empty({0}, options).set_(storage, 0, sizes, strides);
}

// expects as input a complex tensor and returns back a float tensor
// containing the complex values in the last two dimensions
inline Tensor view_complex_as_float(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.is_complex());
  auto new_sizes = self.sizes().vec();
  // last dimension will always have two elements containing the real and imag vals
  new_sizes.emplace_back(2);
  auto new_strides = computeStrideForComplex(self.strides());
  if (self.numel() == 0) {
    return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "view_complex_as_float_empty", [&] {
      auto value_dtype = c10::toValueType(self.scalar_type());
      return from_empty(new_sizes, new_strides, self.options().dtype(value_dtype));
    });
  } else {
    return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "view_complex_as_float", [&] {
      auto value_dtype = c10::toValueType(self.scalar_type());
      auto data = self.data_ptr<scalar_t>();
      return at::from_blob(data, new_sizes, new_strides, self.options().dtype(value_dtype));
    });
  }
}

}} // namespace at::native

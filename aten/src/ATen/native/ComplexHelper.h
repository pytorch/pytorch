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

// expects as input a complex tensor and returns back a float tensor
// containing the complex values in the last two dimensions
inline Tensor view_complex_as_float(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.is_complex());
  auto new_sizes = self.sizes().vec();
  // last dimension will always have two elements containing the real and imag vals
  new_sizes.emplace_back(2);
  auto new_strides = computeStrideForComplex(self.strides());
  if(self.scalar_type() == at::kComplexFloat) {
    float* data = reinterpret_cast<float*>(self.data_ptr<std::complex<float>>());
    return at::from_blob(data, new_sizes, new_strides, dtype(at::kFloat));
  } else {
    double* data = reinterpret_cast<double*>(self.data_ptr<std::complex<double>>());
    return at::from_blob(data, new_sizes, new_strides, dtype(at::kDouble));
  }
}

}} // namespace at::native

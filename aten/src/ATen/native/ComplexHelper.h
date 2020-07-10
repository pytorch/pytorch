#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { struct TensorIterator; }

namespace at { namespace native {

inline std::vector<int64_t> computeStrideForViewAsReal(IntArrayRef oldstride) {
  auto res = oldstride.vec();
  for(size_t i = 0; i < res.size(); i++) {
    res[i] = res[i] * 2;
  }
  res.emplace_back(1);
  return res;
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

using binary_fn = void(*)(TensorIterator&);

DECLARE_DISPATCH(binary_fn, complex_stub);
DECLARE_DISPATCH(binary_fn, complex_polar_stub);

}} // namespace at::native

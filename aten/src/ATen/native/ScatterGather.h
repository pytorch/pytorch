#pragma once
#include <vector>
#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {
namespace {

using gather_fn = void (*)(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);

inline void ensure_nonempty(std::vector<int64_t> &vec) {
  if(vec.size() == 0) {
    vec.push_back(1);
  }
}

}  // namespace

DECLARE_DISPATCH(gather_fn, gather_stub);

}  // namespace native
}  // namespace at

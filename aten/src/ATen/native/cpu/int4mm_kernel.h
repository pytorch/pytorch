#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using weight_to_int4pack_fn = void(*)(const Tensor&, const Tensor&);
using int4pack_mm_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, int64_t, const Tensor&);
DECLARE_DISPATCH(weight_to_int4pack_fn, weight_to_int4pack_stub);
DECLARE_DISPATCH(int4pack_mm_fn, int4pack_mm_stub);

inline bool is_block_start(int index, int BLOCK_SIZE) {
  return !(index & (BLOCK_SIZE -1));
}

} // at::native

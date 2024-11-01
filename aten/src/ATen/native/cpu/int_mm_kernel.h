#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using weight_to_int4pack_fn = void(*)(const Tensor&, const Tensor&, int, int);
using int4pack_mm_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, int, const Tensor&, int, int);
using int8pack_mm_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&);

DECLARE_DISPATCH(weight_to_int4pack_fn, weight_to_int4pack_stub)
DECLARE_DISPATCH(int4pack_mm_fn, int4pack_mm_stub)
DECLARE_DISPATCH(int8pack_mm_fn, int8pack_mm_stub)

} // namespace at::native

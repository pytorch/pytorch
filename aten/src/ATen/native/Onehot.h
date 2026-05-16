#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using one_hot_check_bounds_fn = void(*)(const Tensor&, int64_t);
DECLARE_DISPATCH(one_hot_check_bounds_fn, one_hot_check_bounds_stub)

} // namespace at::native

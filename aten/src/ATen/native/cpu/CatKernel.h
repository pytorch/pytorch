#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using cat_serial_fn = void(*)(const Tensor &, ITensorList, int64_t);
DECLARE_DISPATCH(cat_serial_fn, cat_serial_stub);

}}  // namespace at::native

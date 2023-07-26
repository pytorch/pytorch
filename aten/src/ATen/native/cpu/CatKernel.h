#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/core/IListRef.h>

namespace at { namespace native {

using cat_serial_fn = void(*)(const Tensor &, const MaterializedITensorListRef&, int64_t);
using cat_fast_fn = void(*)(const Tensor &, const MaterializedITensorListRef&, int64_t);

DECLARE_DISPATCH(cat_serial_fn, cat_serial_stub);
DECLARE_DISPATCH(cat_fast_fn, cat_fast_stub);

}}  // namespace at::native

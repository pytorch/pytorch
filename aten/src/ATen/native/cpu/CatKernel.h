#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/core/IListRef.h>

namespace at::native {

using cat_serial_fn = void(*)(const Tensor &, const MaterializedITensorListRef&, int64_t);
DECLARE_DISPATCH(cat_serial_fn, cat_serial_stub);

} // namespace at::native

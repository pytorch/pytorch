#pragma once

#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using cat_serial_fn =
    void (*)(const Tensor&, const MaterializedITensorListRef&, int64_t);
DECLARE_DISPATCH(cat_serial_fn, cat_serial_stub);

} // namespace native
} // namespace at

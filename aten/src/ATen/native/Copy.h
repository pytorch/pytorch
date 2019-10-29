#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

struct TensorIterator;

namespace native {

using copy_fn = void (*)(TensorIterator&, bool non_blocking);

DECLARE_DISPATCH(copy_fn, copy_stub);

} // namespace native
} // namespace at

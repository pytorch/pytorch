#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

struct TensorIterator;

namespace native {

using copy_fn = void (*)(TensorIterator&, bool non_blocking);
using transpose_copy_fn = void (*)(Tensor&, const Tensor&);

DECLARE_DISPATCH(copy_fn, copy_stub);
DECLARE_DISPATCH(transpose_copy_fn, transpose_copy_stub);

} // namespace native
} // namespace at

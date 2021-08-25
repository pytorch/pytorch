#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

struct TensorIterator;

namespace native {

using copy_fn = void (*)(TensorIterator&, bool non_blocking);

DECLARE_DISPATCH(copy_fn, copy_stub);

TORCH_API void copy_ignoring_overlaps(const Tensor &dst, const Tensor &src);

} // namespace native
} // namespace at

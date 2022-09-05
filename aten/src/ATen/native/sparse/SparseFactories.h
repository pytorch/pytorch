#pragma once
#include <ATen/native/DispatchStub.h>

namespace at {
struct TensorIterator;
class TensorBase;

namespace native {

using spdiags_kernel_fn_t =
    void (*)(TensorIterator&, const TensorBase&, TensorBase&, TensorBase&);

DECLARE_DISPATCH(spdiags_kernel_fn_t, spdiags_kernel_stub);
} // namespace native
} // namespace at

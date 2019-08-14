#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
struct TensorIterator;
namespace native {

using amp_pointwise_fn = void (*)(TensorIterator&, double);

DECLARE_DISPATCH(amp_pointwise_fn, amp_unscale_inf_check_stub);

} // namespace at
} // namespace native

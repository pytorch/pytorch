#pragma once

#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using renorm_scale_factor_fn = void (*) (TensorIteratorBase& iter, double maxnorm);
DECLARE_DISPATCH(renorm_scale_factor_fn, renorm_scale_factor_stub);

}}  // namespace at::native

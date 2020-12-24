#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/Optional.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

using addr_fn = void (*)(TensorIterator &, Scalar beta, Scalar alpha);
DECLARE_DISPATCH(addr_fn, addr_stub);

}} // namespace at::native

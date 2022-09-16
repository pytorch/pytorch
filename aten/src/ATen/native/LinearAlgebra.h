#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/util/Optional.h>

namespace c10 {
class Scalar;
}

namespace at {
struct TensorIterator;
}

namespace at { namespace native {

using addr_fn = void (*)(TensorIterator &, const Scalar& beta, const Scalar& alpha);
DECLARE_DISPATCH(addr_fn, addr_stub);
}} // namespace at::native

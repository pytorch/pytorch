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

using linalg_vector_norm_fn = void(*)(TensorIterator &, Scalar);
DECLARE_DISPATCH(linalg_vector_norm_fn, linalg_vector_norm_stub);

using unpack_pivots_fn = void(*)(
  TensorIterator& iter,
  int64_t dim_size
);
DECLARE_DISPATCH(unpack_pivots_fn, unpack_pivots_stub);


}} // namespace at::native

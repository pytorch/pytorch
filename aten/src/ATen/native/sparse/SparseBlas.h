#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using addmv_out_sparse_csr_fn = void (*)(
    const Tensor& /*mat*/,
    const Tensor& /*vec*/,
    const Scalar& /*beta*/,
    const Scalar& /*alpha*/,
    const Tensor& /*result*/);
DECLARE_DISPATCH(addmv_out_sparse_csr_fn, addmv_out_sparse_csr_stub);

} // namespace native
} // namespace at

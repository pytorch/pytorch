#pragma once

#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>

namespace at {
struct TensorIterator;
struct TensorIteratorBase;
}

namespace at::native {

using structured_beta_fn = void(*)(TensorIteratorBase&);

DECLARE_DISPATCH(structured_beta_fn, betainc_stub)

TORCH_API std::tuple<Tensor, Tensor, Tensor> _special_betainc_partials(
    const Tensor& a, const Tensor& b, const Tensor& x);

TORCH_API std::tuple<Tensor, Tensor, Tensor> _special_betaincinv_partials(
    const Tensor& a, const Tensor& b, const Tensor& y);

} // namespace at::native

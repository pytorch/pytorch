#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

enum class NESTED_DENSE_OP : uint8_t { ADD, MUL };

using nested_dense_elementwise_fn = void (*)(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    const NESTED_DENSE_OP& op);

DECLARE_DISPATCH(nested_dense_elementwise_fn, nested_dense_elementwise_stub);

} // namespace at::native

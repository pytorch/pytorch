#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using mul_sparse_sparse_out_fn = void (*)(Tensor&, const Tensor&, const Tensor&);

DECLARE_DISPATCH(mul_sparse_sparse_out_fn, mul_sparse_sparse_out_stub);

}}

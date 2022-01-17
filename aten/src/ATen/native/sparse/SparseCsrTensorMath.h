#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using linalg_solve_fn = void(*)(
  const at::Tensor&, const at::Tensor&, const at::Tensor&, int&
);
DECLARE_DISPATCH(linalg_solve_fn, linalg_solve_sparse_csr_stub);

}} // namespace at::native

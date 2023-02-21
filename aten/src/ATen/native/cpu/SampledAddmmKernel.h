#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using sampled_addmm_sparse_csr_fn = void(*)(const Tensor&, const Tensor&, const Scalar&, const Scalar&, const Tensor&);

DECLARE_DISPATCH(sampled_addmm_sparse_csr_fn, sampled_addmm_sparse_csr_stub);

}} // at::native

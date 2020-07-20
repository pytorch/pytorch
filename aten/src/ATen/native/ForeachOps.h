#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

DECLARE_DISPATCH(std::vector<Tensor>(*)(TensorList, Scalar alpha), foreach_tensor_add_scalar_stub);

}} // namespace at::native

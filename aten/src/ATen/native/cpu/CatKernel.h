#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/core/IListRef.h>

namespace at { namespace native {

using cat_kernel_fn = void(*)(const Tensor &, const MaterializedITensorListRef&, int64_t);
DECLARE_DISPATCH(cat_kernel_fn, cat_serial_stub);
DECLARE_DISPATCH(cat_kernel_fn, cat_parallel_stub);

}}  // namespace at::native

#pragma once
#include <ATen/native/DispatchStub.h>

namespace at {
class Tensor;

namespace native {

using max_unpooling_fn = void(*)(Tensor&, const Tensor&, const Tensor&);

DECLARE_DISPATCH(max_unpooling_fn, max_unpool2d_kernel)
DECLARE_DISPATCH(max_unpooling_fn, max_unpool3d_kernel)

}} // at::native

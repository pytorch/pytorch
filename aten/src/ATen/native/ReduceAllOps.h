#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using reduce_all_fn = void (*)(Tensor & result, const Tensor & self);
DECLARE_DISPATCH(reduce_all_fn, min_all_stub);
DECLARE_DISPATCH(reduce_all_fn, max_all_stub);

}}

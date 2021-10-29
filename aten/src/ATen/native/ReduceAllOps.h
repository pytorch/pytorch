#pragma once

#include <ATen/native/DispatchStub.h>

namespace at {
class Tensor;
}

namespace at { namespace native {

using reduce_all_fn = void (*)(const Tensor & result, const Tensor & self);
DECLARE_DISPATCH(reduce_all_fn, min_all_stub);
DECLARE_DISPATCH(reduce_all_fn, max_all_stub);

}}

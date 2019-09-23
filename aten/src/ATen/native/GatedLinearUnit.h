#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { struct TensorIterator; }

namespace at { namespace native {

using glu_fn = void(*)(TensorIterator&);

DECLARE_DISPATCH(glu_fn, glu_stub);

}} // namespace at::native

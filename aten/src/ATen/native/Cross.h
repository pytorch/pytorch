#pragma once

#include <ATen/native/DispatchStub.h>

namespace at {
class Tensor;

namespace native {

using cross_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const int64_t d);

DECLARE_DISPATCH(cross_fn, cross_stub);

}} // namespace at::native

#pragma once
#include <ATen/native/DispatchStub.h>

namespace at {
class Tensor;

namespace native {

using channel_shuffle_fn = void(*)(Tensor&, const Tensor&, int64_t);
DECLARE_DISPATCH(channel_shuffle_fn, channel_shuffle_kernel);

}} // at::native

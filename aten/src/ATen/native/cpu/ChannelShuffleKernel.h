#pragma once
#include <ATen/native/DispatchStub.h>
#include <cstdint>

namespace at {
class TensorBase;
}

namespace at::native {

using channel_shuffle_fn = void(*)(TensorBase&, const TensorBase&, int64_t);
DECLARE_DISPATCH(channel_shuffle_fn, channel_shuffle_kernel);

} // at::native

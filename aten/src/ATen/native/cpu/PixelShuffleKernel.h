#pragma once
#include <ATen/native/DispatchStub.h>

namespace at {
class TensorBase;
}

namespace at::native {

using pixel_shuffle_fn = void(*)(TensorBase&, const TensorBase&, int64_t);
DECLARE_DISPATCH(pixel_shuffle_fn, pixel_shuffle_kernel)
DECLARE_DISPATCH(pixel_shuffle_fn, pixel_unshuffle_kernel)

} // at::native

#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using lerp_fn = void (*)(
      at::Tensor& ret,
      const at::Tensor& self,
      const at::Tensor& end,
      Scalar weight);

DECLARE_DISPATCH(lerp_fn, lerp_stub);

} // namespace native
} // namespace at

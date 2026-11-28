// Demonstrates the gen_backend_stubs structured-kernel + out-as-primary codegen
// path for a PrivateUse1 backend (see csrc/aten/openreg_native_functions.yaml).
//
// maximum.out and minimum.out are registered as `structured: true`, so torchgen
// generates (into the build tree) RegisterPrivateUse1.cpp +
// PrivateUse1NativeFunctions.h whose functional / inplace / out wrappers drive
// aten's op.meta() and the backend's op.impl(). The backend only has to provide
// impl(), which we delegate to CPU here (openreg tensors are host-backed),
// mirroring abs_out in native/Extra.cpp. div.out and angle.out are plain .out
// kernels; see their comments for what torchgen generates around them.
#include <ATen/native/Resize.h>
#include <ATen/ops/angle.h>
#include <ATen/ops/div.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/minimum.h>

#include <ATen/native/PrivateUse1NativeFunctionsHelper.h>

#include "PrivateUse1NativeFunctions.h"

namespace at::native::openreg {

// structured: true, native meta -- only impl() is needed.
TORCH_PRIVATEUSE1_IMPL_FUNC(maximum_out)
(const at::Tensor& self, const at::Tensor& other, const at::Tensor& out) {
  out.copy_(at::maximum(self.cpu(), other.cpu()));
}

// structured: true, native meta -- only impl() is needed.
TORCH_PRIVATEUSE1_IMPL_FUNC(minimum_out)
(const at::Tensor& self, const at::Tensor& other, const at::Tensor& out) {
  out.copy_(at::minimum(self.cpu(), other.cpu()));
}

// Registered as a plain .out kernel. aten defines div as structured, so torchgen
// generates nothing for div / div_: both stay with aten's composite, which runs
// meta() and then calls this kernel with a sized out. Only a user out= call
// reaches it without meta(), so it sizes the output itself.
at::Tensor& TORCH_PRIVATEUSE1_FUNC(div_out)(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  const auto result = at::div(self.cpu(), other.cpu());
  out.resize_(result.sizes());
  out.copy_(result);
  return out;
}

// Registered as a plain .out kernel for an op aten has no structured or
// composite kernel for, so torchgen derives angle() from it: the derived
// functional passes an undefined out, and only the kernel knows the output
// dtype (angle of a complex or integer input is a floating tensor).
at::Tensor& TORCH_PRIVATEUSE1_FUNC(angle_out)(
    const at::Tensor& self,
    at::Tensor& out) {
  const auto result = at::angle(self.cpu());
  if (!out.defined()) {
    out = at::empty(result.sizes(), self.options().dtype(result.scalar_type()));
  } else {
    at::native::resize_output(out, result.sizes());
  }
  out.copy_(result);
  return out;
}

} // namespace at::native::openreg

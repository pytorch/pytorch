// Demonstrates the gen_backend_stubs structured-kernel + out-as-primary codegen
// path for a PrivateUse1 backend (see csrc/aten/openreg_native_functions.yaml).
//
// maximum.out is registered as `structured: true`, so torchgen generates (into
// the build tree) RegisterPrivateUse1.cpp + PrivateUse1NativeFunctions.h: the
// out / functional wrappers drive op.meta() (reusing the native meta) and
// op.impl(), and -- with use_out_as_primary: true -- the functional variant
// allocates an empty out and reuses it instead of doing a temp-copy. The backend
// only has to provide impl(), which we delegate to CPU here (openreg tensors are
// host-backed), mirroring abs_out in native/Extra.cpp.
#include <ATen/ops/maximum.h>

#include <ATen/native/PrivateUse1NativeFunctionsHelper.h>

#include "PrivateUse1NativeFunctions.h"

namespace at::native::openreg {

TORCH_PRIV1_IMPL_FUNC(maximum_out)
(const at::Tensor& self, const at::Tensor& other, const at::Tensor& out) {
  out.copy_(at::maximum(self.cpu(), other.cpu()));
}

} // namespace at::native::openreg

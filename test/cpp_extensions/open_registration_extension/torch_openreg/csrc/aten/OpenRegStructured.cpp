// Demonstrates the gen_backend_stubs structured-kernel + out-as-primary codegen
// path for a PrivateUse1 backend (see csrc/aten/openreg_native_functions.yaml).
//
// Both maximum.out and minimum.out are registered as `structured: true`, so
// torchgen generates (into the build tree) RegisterPrivateUse1.cpp +
// PrivateUse1NativeFunctions.h: the out / functional wrappers drive op.meta()
// and op.impl(), and -- with use_out_as_primary: true -- the functional variant
// allocates an empty out and reuses it instead of doing a temp-copy. The backend
// only has to provide impl(), which we delegate to CPU here (openreg tensors are
// host-backed), mirroring abs_out in native/Extra.cpp.
//
// openreg deliberately does NOT set `define_meta: true`. define_meta emits a
// TORCH_LIBRARY_IMPL(aten, Meta) that overrides aten's meta on the device-agnostic,
// process-global Meta key -- as an autoloaded in-tree test fixture, openreg must not
// mutate global aten state. The define_meta codegen is covered by the gen_backend_stubs
// unit tests; the runtime Meta-override mechanism is exercised RAII-scoped in tests/test_ops.py.
#include <ATen/ops/div.h>
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

// Non-structured out-as-primary: no native meta is run, so the backend's
// op_out must size the output itself. torchgen routes the functional and
// inplace variants of div through this single function.
at::Tensor& TORCH_PRIVATEUSE1_FUNC(div_out)(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  const auto result = at::div(self.cpu(), other.cpu());
  out.resize_(result.sizes());
  out.copy_(result);
  return out;
}

} // namespace at::native::openreg

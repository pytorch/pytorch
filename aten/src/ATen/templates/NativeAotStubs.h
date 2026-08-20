#pragma once
// ${generated_comment}

// DispatchStubs for AOT-compiled DSL kernels embedded into ATen ops, one
// per (op, dispatch key) declared by a torch/_native/ops/<op>/aot.py
// declaration module. The generated structured-kernel wrappers
// (Register<Key>.cpp) consult the stub between op.meta() and op.impl():
// outputs are already allocated/validated, so an AOT kernel only has to
// fill them and return true; returning false falls through to the stock
// impl. The stub signature is the op's structured impl signature.
//
// No kernel is registered here (REGISTER_NO_CPU_DISPATCH in the .cpp).
// The AOT kernels -- built separately from the same declarations via
// tools/native_aot; needs the DSL toolchain but not this build -- are
// linked into libtorch_cuda and register with set_<device>_dispatch_ptr()
// from static initializers.
// The whole path is gated on at::globalContext().allowNativeAot().

// TensorBase.h, not Tensor.h: every stub signature takes `const at::Tensor &`,
// so a declaration is all this header needs -- and Tensor.h drags in the
// generated TensorBody.h, which makes every includer rebuild on any
// native_functions.yaml change.
#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

${native_aot_stub_declarations}

} // namespace at::native

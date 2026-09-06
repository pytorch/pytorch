#pragma once
// ${generated_comment}

// DispatchStubs for AOT-compiled DSL kernels embedded into ATen ops, one per
// (op, dispatch key) declared by a torch/_native/ops/<op>/aot.py module. The
// generated structured wrappers consult the stub between op.meta() and op.impl():
// the outputs are already allocated, so an AOT kernel fills them and returns true,
// or returns false to fall through to the stock impl. The stub signature is the
// op's structured impl signature, and the path is gated on
// at::globalContext().allowNativeAot().
//
// No kernel is registered here. The AOT kernels are built separately from the same
// declarations by tools/native_aot, linked into libtorch_cuda, and registered with
// set_<device>_dispatch_ptr() from static initializers.

// TensorBase.h, not Tensor.h: a declaration is all this header needs, and Tensor.h
// drags in the generated TensorBody.h, rebuilding every includer on any
// native_functions.yaml change.
#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

${native_aot_stub_declarations}

} // namespace at::native

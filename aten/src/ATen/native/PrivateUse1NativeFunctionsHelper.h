#pragma once

/*
 * PrivateUse1NativeFunctionsHelper.h
 *
 * This header serves as a dedicated extension utility for out-of-tree backends.
 * It provides specialized macros to streamline kernel implementations for the
 * PrivateUse1 ecosystem.
 *
 * These macros are designed to align with and extend the structured kernel
 * architecture defined in ATen/TensorMeta.h. While the core macros in TensorMeta.h
 * assume a global or anonymous namespace, this helper resolves definitions into
 * the backend-specific dispatch class (default: PrivateUse1NativeFunctions)
 * required by gen_backend_stubs.py.
 *
 */

// TORCH_PRIVATEUSE1_CLASS resolves the target dispatch class for PrivateUse1 kernels.
// It aligns with the 'class_name' attribute used by gen_backend_stubs.py.
#ifndef TORCH_PRIVATEUSE1_CLASS
#define TORCH_PRIVATEUSE1_CLASS PrivateUse1NativeFunctions
#endif

// TORCH_PRIVATEUSE1_FUNC is a specialized macro for non-structured backend operators.
// Example: Tensor TORCH_PRIVATEUSE1_FUNC(add)(const Tensor& self, const Tensor& other)
#define TORCH_PRIVATEUSE1_FUNC(name) TORCH_PRIVATEUSE1_CLASS::name

/*
 * Non-structured .out kernels under use_out_as_primary
 * ----------------------------------------------------------------------------
 * When a backend registers only `op.out` (structured: false), torchgen fills in
 * the other variants:
 *   - If aten already provides the functional or inplace (a structured op, or a
 *     composite kernel), nothing is generated for it: aten's kernel runs meta()
 *     and then calls this .out kernel with a defined, correctly sized out.
 *   - Otherwise the variant is derived from this .out kernel. A derived inplace
 *     passes `self` as `out`. A derived functional passes an UNDEFINED `out`,
 *     because only the kernel knows the output dtype and shape; the kernel must
 *     allocate it, and the wrapper checks that it did.
 * So an .out kernel reached by a derived functional must handle both cases:
 *
 *   Tensor& TORCH_PRIVATEUSE1_FUNC(op_out)(const Tensor& self, Tensor& out) {
 *     if (!out.defined()) {
 *       out = at::empty(sizes, self.options().dtype(result_dtype));
 *     } else {
 *       at::native::resize_output(out, sizes);
 *     }
 *     ...
 *   }
 *
 * A user `out=` call reaches the kernel directly, without meta(). For an op aten
 * defines as structured, prefer `structured: true` so every variant, out=
 * included, goes through aten's meta().
 * ----------------------------------------------------------------------------
 */

/*
 * ----------------------------------------------------------------------------
 * PrivateUse1 Structured Kernel Macros
 * ----------------------------------------------------------------------------
 * A backend op registered with `structured: true` reuses aten's meta() and only
 * implements impl(). The macro names the struct torchgen generates for the
 * backend's .out kernel, so it is "struct-explicit": pass the kernel name as it
 * appears in the generated header, overload suffix included.
 *
 * Example: maximum.out -> TORCH_PRIVATEUSE1_IMPL_FUNC(maximum_out)
 *          div.out_mode -> TORCH_PRIVATEUSE1_IMPL_FUNC(div_out_mode)
 * ----------------------------------------------------------------------------
 */

// TORCH_PRIVATEUSE1_IMPL_FUNC defines the hardware-specific kernel implementation
// inside the backend's dispatch class scope.
// Expansion: void <class_name>::structured_<kernel>::impl
#define TORCH_PRIVATEUSE1_IMPL_FUNC(name) \
  void TORCH_PRIVATEUSE1_CLASS::structured_##name::impl

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

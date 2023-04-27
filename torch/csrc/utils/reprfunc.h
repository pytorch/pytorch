/// Type-safe-ish utilities for defining __repr__ on Python types.
///
/// The classic Python way of handling this is c-style casts
/// transmuting one function type to another. This is very sketchy and
/// violates a modern Clang warning "-Wcast-function-type-strict". A
/// somewhat less sketchy approach is to preserve the function
/// signature and provide adapters that reinterpret_cast arguments as
/// necessary. This is more principled, but isn't fully type-safe
/// since nothing prevents you from giving Python an adapter that
/// takes the wrong argument.

#pragma once

#include <Python.h>

#include <string>
#include <type_traits>

/// Adapts an existing type-safe function to the reprfunc signature.
///
/// This is provided as an alternative to using torch::as_reprfunc()
/// directly. The advantage of the macro is that the type of the
/// object is inferred.
///
/// @example
/// ```
/// PyTypeObject type = {
///   ...
///   /*tp_basicsize=*/sizeof(CppType)
///   // Note that it is very important that you put the + in front of
///   // the lambda, and also that the lambda is stateless. This is a
///   // technical requirement of the design of this feature.
///   /*tp_reprfunc=*/TORCH_AS_REPRFUNC(CppType_repr);
/// };
/// ```
#define TORCH_AS_REPRFUNC(func) \
  ::torch::as_reprfunc<decltype(::torch::detail::get_self((func))), (func)>()

namespace torch::detail {

/// Typed alias of a reprfunc.
template <typename Self>
using ReprFunc = PyObject* (*)(Self* self);

} // namespace torch::detail

namespace torch {

/// Adapts an existing type-safe function to the reprfunc signature.
///
/// This is available if you prefer redundantly specifying the type of
/// the object to the ickiness of a macro.
///
/// @example
/// ```
/// PyTypeObject type = {
///   ...
///   /*tp_basicsize=*/sizeof(CppType)
///   // Note that it is very important that you put the + in front of
///   // the lambda, and also that the lambda is stateless. This is a
///   // technical requirement of the design of this feature.
///   /*tp_reprfunc=*/torch::as_reprfunc<CppType, CppType_repr>();
/// };
/// ```
///
/// Note that in C++ 20, we will be able to infer the type using
/// "template <auto impl>" and the macro will no longer provide any
/// value.
template <typename Self, detail::ReprFunc<Self> impl>
reprfunc as_reprfunc();

} // namespace torch

// The rest of the file is implementation details. Review at your own peril.

namespace torch::detail {

// Extracts the object type of a ReprFunc.
//
// Example:
//   using Self = decltype(get_self(some_typed_reprfunc));
template <typename Self>
Self get_self(ReprFunc<Self> func) {
  return std::declval<Self>();
}

} // namespace torch::detail

namespace torch {

template <typename Self, detail::ReprFunc<Self> impl>
reprfunc as_reprfunc() {
  return +[](PyObject* self) { return impl(reinterpret_cast<Self*>(self)); };
}

} // namespace torch

#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>

namespace torch {
namespace jit {

TORCH_API at::TypePtr getCustomClass(const std::string& name);

TORCH_API bool isCustomClass(const c10::IValue& v);

using GetCustomClassFnType = at::TypePtr (*)(const std::string&);
// Use this to set the function for retrieving custom classes
//
// This is necessary because the custom classes implementation
// is not in ATen core, but the schema type parser is, which
// can resolve custom classes as type expressions.
TORCH_API void setGetCustomClassFn(GetCustomClassFnType fn);

TORCH_API int register_custom_class_handler();

} // namespace jit
} // namespace torch

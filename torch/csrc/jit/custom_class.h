#pragma once

#include <ATen/core/jit_type.h>

namespace torch {
namespace jit {

at::TypePtr getCustomClass(const std::string& name);

using GetCustomClassFnType = at::TypePtr (*)(const std::string&);
// Use this to set the function for retrieving custom classes
//
// This is necessary because the custom classes implementation
// is not in ATen core, but the schema type parser is, which
// can resolve custom classes as type expressions.
void setGetCustomClassFn(GetCustomClassFnType fn);

} // namespace jit
} // namespace torch

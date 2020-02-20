#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <torch/custom_class.h>

namespace torch {
namespace jit {

TORCH_API at::TypePtr getCustomClass(const std::string& name);

TORCH_API bool isCustomClass(const c10::IValue& v);

TORCH_API std::shared_ptr<script::CompilationUnit>& classCU();

} // namespace jit
} // namespace torch

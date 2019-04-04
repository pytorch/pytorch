#pragma once

#include <ATen/core/function_schema.h>
#include <c10/macros/Macros.h>
#include <string>

namespace torch {
namespace jit {

CAFFE2_API ::c10::FunctionSchema parseSchema(const std::string& schema);

} // namespace jit
} // namespace torch

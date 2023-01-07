#pragma once

#include <ATen/core/function_schema.h>
#include <c10/macros/Macros.h>
#include <c10/util/either.h>
#include <string>

namespace torch {
namespace jit {

TORCH_API c10::either<c10::OperatorName, c10::FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName);
TORCH_API c10::FunctionSchema parseSchema(const std::string& schema);
TORCH_API c10::OperatorName parseName(const std::string& name);

} // namespace jit
} // namespace torch

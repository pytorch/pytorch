#pragma once

#include <ATen/core/function_schema.h>
#include <c10/macros/Macros.h>
#include <string>
#include <variant>

namespace torch::jit {

// allow_typevars: If true, we assume that lowercase types that we don't
// understand are type variables. This is only needed for TorchScript (and not
// not needed for custom ops).
// If false, we disallow typevars, except in certain cases for BC reason (i.e.
// your op is in the aten or prim namespace).
TORCH_API std::variant<c10::OperatorName, c10::FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName,
    bool allow_typevars = true);
TORCH_API c10::FunctionSchema parseSchema(
    const std::string& schema,
    bool allow_typevars = true);
TORCH_API c10::OperatorName parseName(const std::string& name);

} // namespace torch::jit

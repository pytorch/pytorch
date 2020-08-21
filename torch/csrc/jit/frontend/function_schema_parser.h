#pragma once

#include <ATen/core/Macros.h>
#include <ATen/core/function_schema.h>
#include <c10/util/either.h>
#include <string>

namespace torch {
namespace jit {

CAFFE2_API c10::either<c10::OperatorName, c10::FunctionSchema> parseSchemaOrName(
    c10::string_view schemaOrName);
CAFFE2_API c10::FunctionSchema parseSchema(c10::string_view schema);
CAFFE2_API c10::OperatorName parseName(c10::string_view name);

} // namespace jit
} // namespace torch

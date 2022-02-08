#pragma once

#include <ATen/core/Macros.h>
#include <ATen/core/alias_info.h>
#include <ATen/core/jit_type.h>
#include <c10/util/FunctionRef.h>
#include <torch/csrc/jit/frontend/lexer.h>

namespace torch {
namespace jit {

using TypePtr = c10::TypePtr;

struct TORCH_API SchemaTypeParser {
  TypePtr parseBaseType();
  c10::optional<c10::AliasInfo> parseAliasAnnotation();
  std::pair<TypePtr, c10::optional<c10::AliasInfo>> parseType();
  c10::optional<at::ScalarType> parseTensorDType(const std::string& dtype);
  TypePtr parseRefinedTensor();

  SchemaTypeParser(Lexer& L, bool parse_complete_tensor_types) : L(L) {
    complete_tensor_types = parse_complete_tensor_types;
  }

 private:
  c10::optional<bool> tryToParseRequiresGrad();
  c10::optional<c10::Device> tryToParseDeviceType();
  void parseList(
      int begin,
      int sep,
      int end,
      c10::function_ref<void()> callback);

  bool complete_tensor_types;
  Lexer& L;
  size_t next_id = 0;
};
} // namespace jit
} // namespace torch

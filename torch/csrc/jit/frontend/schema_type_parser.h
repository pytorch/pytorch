#pragma once

#include <ATen/core/jit_type.h>
#include <ATen/core/alias_info.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <ATen/core/Macros.h>

namespace torch {
namespace jit {

using TypePtr = c10::TypePtr;

struct CAFFE2_API SchemaTypeParser {
  TypePtr parseBaseType();
  c10::optional<c10::AliasInfo> parseAliasAnnotation();
  std::pair<TypePtr, c10::optional<c10::AliasInfo>> parseType();
  c10::optional<at::ScalarType> parseTensorDType(const std::string& dtype);
  TypePtr parseRefinedTensor();

  SchemaTypeParser(Lexer& L, bool parse_complete_tensor_types) : L(L) {
    complete_tensor_types = parse_complete_tensor_types;
  }

 private:
  void parseList(
      int begin,
      int sep,
      int end,
      const std::function<void()>& callback);

  bool complete_tensor_types;
  Lexer& L;
  size_t next_id = 0;
};
} // namespace jit
} // namespace torch

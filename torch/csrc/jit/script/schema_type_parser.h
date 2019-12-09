#pragma once

#include <ATen/core/jit_type.h>
#include <ATen/core/alias_info.h>
#include <torch/csrc/jit/script/lexer.h>
#include <ATen/core/Macros.h>

namespace torch {
namespace jit {
namespace script {

using TypePtr = c10::TypePtr;
using TypeAndAlias = std::pair<TypePtr, c10::optional<c10::AliasInfo>>;

struct CAFFE2_API SchemaTypeParser {
  TypeAndAlias parseBaseType();
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
} // namespace script
} // namespace jit
} // namespace torch

#pragma once

#include <ATen/core/Macros.h>
#include <ATen/core/alias_info.h>
#include <ATen/core/jit_type.h>
#include <c10/util/string_view.h>
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

  void setStringView(c10::string_view text) {
    strView = text;
  }

  c10::string_view textForToken(const Token& t) {
    AT_ASSERT(!strView.empty()); // XXX breaks other use cases but should work
                                 // for our prototype
    return strView.substr(t.range.start(), t.range.end() - t.range.start());
  }

 private:
  static std::shared_ptr<Source> newSource(c10::string_view v) {
    return std::make_shared<Source>(std::string(v.begin(), v.end()));
  }

  Token withSource(Token t) {
    auto result = std::move(t);
    result.range = SourceRange(
        newSource(strView), result.range.start(), result.range.end());
    return result;
  }

  c10::optional<bool> tryToParseRequiresGrad();
  c10::optional<c10::Device> tryToParseDeviceType();
  void parseList(
      int begin,
      int sep,
      int end,
      const std::function<void()>& callback);

  bool complete_tensor_types;
  Lexer& L;
  size_t next_id = 0;
  c10::string_view strView;
};
} // namespace jit
} // namespace torch

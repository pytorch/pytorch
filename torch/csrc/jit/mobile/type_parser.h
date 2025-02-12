#pragma once

#include <ATen/core/dynamic_type.h>
#include <ATen/core/jit_type.h>
#include <unordered_set>

namespace c10 {

class TORCH_API TypeParser {
 public:
  explicit TypeParser(std::string pythonStr);
  explicit TypeParser(std::vector<std::string>& pythonStrs);

  TypePtr parse();
  std::vector<TypePtr> parseList();
  static const std::unordered_set<std::string>& getNonSimpleType();
  static const std::unordered_set<std::string>& getCustomType();
  std::unordered_set<std::string> getContainedTypes();

 private:
  TypePtr parseNamedTuple(const std::string& qualified_name);
  TypePtr parseCustomType();
  TypePtr parseTorchbindClassType();
  TypePtr parseNonSimple(const std::string& token);

  void expect(const char* s);
  void expectChar(char c);
  template <typename T>
  TypePtr parseSingleElementType();

  void lex();

  std::string next();
  std::string_view nextView();
  void advance();
  [[nodiscard]] std::string_view cur() const;

  std::string pythonStr_;
  size_t start_;
  std::string_view next_token_;

  // Used for parsing string list
  std::vector<std::string> pythonStrs_;
  std::unordered_map<std::string, c10::TypePtr> str_type_ptr_map_;

  // Store all contained types when parsing a string
  std::unordered_set<std::string> contained_types_;
};

TORCH_API TypePtr parseType(const std::string& pythonStr);

TORCH_API std::vector<TypePtr> parseType(std::vector<std::string>& pythonStr);

} // namespace c10

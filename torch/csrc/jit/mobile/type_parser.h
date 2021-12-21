#pragma once

#include <type_traits>

#include <ATen/core/dynamic_type.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/frontend/string_to_type.h>

namespace c10 {

class TORCH_API TypeParser {
  template <typename T>
  struct TypeFactory {};

 public:
  explicit TypeParser(std::string pythonStr);
  explicit TypeParser(std::vector<std::string>& pythonStrs);

  template <typename T>
  TypePtr parse();
  std::vector<TypePtr> parseList();
  static std::unordered_set<std::string> getNonSimpleType();
  static std::unordered_set<std::string> getCustomType();
  std::unordered_set<std::string> getContainedTypes();

 private:
  // Torchbind custom class always starts with the follow prefix, so use it as
  // an identifier for torchbind custom class type
  static constexpr const char* kTypeTorchbindCustomClass =
      "__torch__.torch.classes";
  static constexpr const char* kTypeNamedTuple = "NamedTuple";

  template <typename T>
  TypePtr parseNamedTuple(const std::string& qualified_name);
  template <typename T>
  TypePtr parseCustomType();
  TypePtr parseTorchbindClassType();
  template <typename T>
  TypePtr parseNonSimple(const std::string& token);

  void expect(const char* s);
  void expectChar(char c);
  template <typename M, typename T>
  TypePtr parseSingleElementType();

  void lex();

  std::string next();
  c10::string_view nextView();
  void advance();
  C10_NODISCARD c10::string_view cur() const;

  std::string pythonStr_;
  size_t start_;
  c10::string_view next_token_;

  // Used for parsing string list
  std::vector<std::string> pythonStrs_;
  std::unordered_map<std::string, c10::TypePtr> str_type_ptr_map_;

  // Store all contained types when parsing a string
  std::unordered_set<std::string> contained_types_;
};

template <typename T = c10::Type>
TORCH_API TypePtr parseType(const std::string& pythonStr) {
  TypeParser parser(pythonStr);
  return parser.parse<T>();
}

TORCH_API std::vector<TypePtr> parseType(std::vector<std::string>& pythonStr);

} // namespace c10

#include <torch/csrc/jit/mobile/type_parser_inl.h>

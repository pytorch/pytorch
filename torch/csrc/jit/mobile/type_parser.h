#include <ATen/core/jit_type.h>

namespace c10 {
class TypeParser {
 public:
  explicit TypeParser(std::string pythonStr);

  TypePtr parse();
  static std::unordered_set<std::string> getNonSimpleType();
  static std::unordered_set<std::string> getCustomType();
  std::unordered_set<std::string> getContainedTypes();
  TypePtr parseNonSimple(const std::string& token);

 private:
  TypePtr parseTorchbindClassType();

  void expect(const std::string& s);

  template <class T>
  TypePtr CreateSingleElementType();

  void lex();

  std::string next();

  std::string& cur();

  std::string pythonStr_;
  size_t start_;
  std::string next_token_;

  // Store all contained types when parsing a string
  std::unordered_set<std::string> contained_types_;
};

TORCH_API TypePtr parseType(const std::string& pythonStr);
} // namespace c10

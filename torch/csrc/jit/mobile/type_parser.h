#include <ATen/core/jit_type.h>

namespace c10 {
class TypeParser {
 public:
  explicit TypeParser(std::string pythonStr);

  TypePtr parse();

 private:
  TypePtr parseClassType();

  void expect(const std::string& s);

  template <class T>
  TypePtr CreateSingleElementType();

  void lex();

  std::string next();

  std::string& cur();

  std::string pythonStr_;
  size_t start_;
  std::string next_token_;
};

TORCH_API TypePtr parseType(const std::string& pythonStr);
} // namespace c10

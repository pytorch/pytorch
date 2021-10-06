#include <ATen/core/jit_type.h>

namespace c10 {
class TypeParser {
 public:
  explicit TypeParser(std::string pythonStr);

  TypePtr parse();

 private:
  TypePtr parseClassType();

  void expect(const char* s);
  void expectChar(char c);
  template <class T>
  TypePtr CreateSingleElementType();

  void lex();

  std::string next();
  void advance();
  C10_NODISCARD c10::string_view cur() const;

  std::string pythonStr_;
  size_t start_;
  c10::string_view next_token_;
};

TORCH_API TypePtr parseType(const std::string& pythonStr);
} // namespace c10

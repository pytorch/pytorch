#include <ATen/ATen.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/alias_info.h>
#include <torch/csrc/jit/script/lexer.h>

namespace torch {
namespace jit {
namespace script {

using TypePtr = c10::TypePtr;
using TypeAndAlias = std::pair<TypePtr, c10::optional<AliasInfo>>;

struct SchemaTypeParser {
  TypeAndAlias parseBaseType();
  c10::optional<AliasInfo> parseAliasAnnotation();
  std::pair<TypePtr, c10::optional<AliasInfo>> parseType();
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

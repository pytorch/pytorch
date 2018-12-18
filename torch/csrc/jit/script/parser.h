#pragma once
#include <memory>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/script/tree.h>

namespace torch {
namespace jit {
namespace script {

struct Decl;
struct ParserImpl;
struct Lexer;

TORCH_API Decl mergeTypesFromTypeComment(const Decl& decl, const Decl& type_annotation_decl, bool is_method);

struct TORCH_API Parser {
  explicit Parser(const std::string& str);
  TreeRef parseFunction(bool is_method);
  Decl parseTypeComment();
  Lexer& lexer();
  ~Parser();
private:
  std::unique_ptr<ParserImpl> pImpl;
};

} // namespace script
} // namespace jit
} // namespace torch

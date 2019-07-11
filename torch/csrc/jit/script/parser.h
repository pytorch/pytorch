#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/script/tree.h>
#include <torch/csrc/jit/script/tree_views.h>
#include <memory>

namespace torch {
namespace jit {
namespace script {

struct Decl;
struct ParserImpl;
struct Lexer;

TORCH_API Decl mergeTypesFromTypeComment(
    const Decl& decl,
    const Decl& type_annotation_decl,
    bool is_method);

struct TORCH_API Parser {
  explicit Parser(const std::string& str);
  TreeRef parseFunction(bool is_method);
  TreeRef parseClass();
  Decl parseTypeComment();
  Expr parseExp();
  Lexer& lexer();
  ~Parser();

 private:
  std::unique_ptr<ParserImpl> pImpl;
};

} // namespace script
} // namespace jit
} // namespace torch

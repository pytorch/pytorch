#pragma once
#include <ATen/core/jit_type.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/script/resolver.h>
#include <torch/csrc/jit/script/tree_views.h>

namespace torch {
namespace jit {
namespace script {

/**
 * class ScriptTypeParser
 *
 * Parses expressions in our typed AST format (TreeView) into types and
 * typenames.
 */
class TORCH_API ScriptTypeParser {
 public:
  explicit ScriptTypeParser() {}
  explicit ScriptTypeParser(ResolverPtr resolver)
      : resolver_(std::move(resolver)) {}
  c10::optional<std::string> parseBaseTypeName(const Expr& expr) const;

  c10::TypePtr parseTypeFromExpr(const Expr& expr) const;

  c10::optional<std::pair<c10::TypePtr, int32_t>> parseBroadcastList(
      const Expr& expr) const;

  c10::TypePtr parseType(const std::string& str);

 private:
  at::TypePtr subscriptToType(
      const std::string& typeName,
      const Subscript& subscript) const;
  ResolverPtr resolver_ = nullptr;
};
} // namespace script
} // namespace jit
} // namespace torch

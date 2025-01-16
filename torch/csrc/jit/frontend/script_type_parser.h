#pragma once
#include <ATen/core/jit_type.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/tree_views.h>

namespace torch::jit {

/**
 * class ScriptTypeParser
 *
 * Parses expressions in our typed AST format (TreeView) into types and
 * typenames.
 */
class TORCH_API ScriptTypeParser {
 public:
  explicit ScriptTypeParser() = default;
  explicit ScriptTypeParser(ResolverPtr resolver)
      : resolver_(std::move(resolver)) {}

  c10::TypePtr parseTypeFromExpr(const Expr& expr) const;

  std::optional<std::pair<c10::TypePtr, int32_t>> parseBroadcastList(
      const Expr& expr) const;

  c10::TypePtr parseType(const std::string& str);

  FunctionSchema parseSchemaFromDef(const Def& def, bool skip_self);

  c10::IValue parseClassConstant(const Assign& assign);

 private:
  c10::TypePtr parseTypeFromExprImpl(const Expr& expr) const;

  std::optional<std::string> parseBaseTypeName(const Expr& expr) const;
  at::TypePtr subscriptToType(
      const std::string& typeName,
      const Subscript& subscript) const;
  std::vector<IValue> evaluateDefaults(
      const SourceRange& r,
      const std::vector<Expr>& default_types,
      const std::vector<Expr>& default_exprs);
  std::vector<Argument> parseArgsFromDecl(const Decl& decl, bool skip_self);

  std::vector<Argument> parseReturnFromDecl(const Decl& decl);

  ResolverPtr resolver_ = nullptr;

  // Need to use `evaluateDefaults` in serialization
  friend struct ConstantTableValue;
  friend struct SourceImporterImpl;
};
} // namespace torch::jit
